from scipy.interpolate import interp1d
import pandas as pd
import numpy as np

def preprocessing(uploaded_file):
    # Detect name/extension whether this is a Streamlit UploadedFile or a path
    fname = uploaded_file.name.lower() if hasattr(uploaded_file, "name") else str(uploaded_file).lower()
    file_obj = uploaded_file

    # --- Load file into dataframe ---
    if fname.endswith(".csv"):
        df = pd.read_csv(file_obj)
    elif fname.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_obj)
    elif fname.endswith(".json"):
        df = pd.read_json(file_obj)
    elif fname.endswith(".txt"):
        try:
            df = pd.read_csv(file_obj, sep="\t")
        except Exception:
            df = pd.read_csv(file_obj, delim_whitespace=True)
    else:
        raise ValueError(f"Unsupported file format: {fname}")

    # --- Normalize/rename columns to x (energy) and y (intensity) ---
    rename_map = {}
    for col in df.columns:
        col_norm = str(col).strip().lower().replace(" ", "")
        if col_norm in {"energy","e","ev","energy(ev)","energy(eV)".lower()}:
            rename_map[col] = "x"
        elif col_norm in {"intensity","counts","absorbance","mu","y"}:
            rename_map[col] = "y"
    df = df.rename(columns=rename_map)

    # --- If no energy column, assume evenly spaced 8330→8370 ---
    if "y" in df.columns and "x" not in df.columns:
        # use the length of y to set grid
        n = len(df["y"])
        if n < 2:
            raise ValueError("Need at least 2 intensity points to build an energy grid.")
        df["x"] = np.linspace(8330.0, 8370.0, n)

    # --- Sanity checks ---
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Could not find both energy and intensity columns. "
                         f"Available columns: {df.columns.tolist()}")

    # Drop NaNs and sort by energy (interp1d requires increasing x)
    df = df[["x","y"]].dropna()
    df = df.sort_values("x", kind="mergesort")
    # If there are duplicate x values, average their y's
    df = df.groupby("x", as_index=False)["y"].mean()

    if len(df) < 2:
        raise ValueError("Not enough points after cleaning to interpolate (need ≥ 2).")

    # --- Interpolate onto a standard grid 8330→8370 (100 points) ---
    new_x = np.linspace(8330.0, 8370.0, 100)
    f = interp1d(df["x"].to_numpy(), df["y"].to_numpy(),
                 kind="linear", bounds_error=False, fill_value="extrapolate")
    interpolated_y = f(new_x)

    # --- CDF normalization ---
    cdf = np.cumsum(interpolated_y)
    denom = (cdf.max() - cdf.min())
    norm_cdf = (cdf - cdf.min()) / denom if denom != 0 else np.zeros_like(cdf)

    return pd.DataFrame({"x": new_x, "y": interpolated_y, "cdf": norm_cdf})
