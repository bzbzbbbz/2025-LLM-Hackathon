import numpy as np
import pickle
import shap
import numpy as np
np.float = float
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.util.plotting import pretty_plot
import pickle
from pymatgen.analysis.xas.spectrum import XAS
from pathlib import Path


# === Constants ===
ENERGY_MIN = 8330
ENERGY_MAX = 8370
TOP_K = 20

# === Getting Data ===

#def load_input(uploaded_file_or_df, limit: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
#    """
#    Load the user data. Accepts either:
#      - a file-like object (Streamlit uploader) OR
#      - a pandas DataFrame

#    Returns (df_input, X_exp) where X_exp is the numpy array from 'cdf' column.
#    """
#    if isinstance(uploaded_file_or_df, pd.DataFrame):
#        df_input = uploaded_file_or_df
#    else:
#        # file-like object (e.g., from st.file_uploader)
#        df_input = pd.read_json(uploaded_file_or_df)

#    if limit is not None:
#        df_input = df_input.iloc[:limit, :]

#    X_exp = np.array(df_input["cdf"].to_list())
#    return df_input, X_exp

def load_input(uploaded_file_or_df, limit: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    if isinstance(uploaded_file_or_df, pd.DataFrame):
        df_input = uploaded_file_or_df
    else:
        file_name = getattr(uploaded_file_or_df, "name", "uploaded")
        if file_name.endswith(".csv"):
            df_input = pd.read_csv(uploaded_file_or_df)
        elif file_name.endswith((".xlsx", ".xls")):
            df_input = pd.read_excel(uploaded_file_or_df)
        elif file_name.endswith(".json"):
            df_input = pd.read_json(uploaded_file_or_df)
        elif file_name.endswith(".txt"):
            df_input = pd.read_csv(uploaded_file_or_df, delim_whitespace=True)
        else:
            raise ValueError(f"Unsupported file format: {file_name}")

    if limit is not None:
        df_input = df_input.iloc[:limit, :]

    # Case 1: already has cdf
    if "cdf" in df_input.columns:
        X_exp = np.array(df_input["cdf"].to_list())
        return df_input, X_exp

    # Case 2: try to detect energy + intensity
    lower_cols = [c.lower() for c in df_input.columns]
    try:
        energy_col = next(c for c in df_input.columns if "energy" in c.lower())
        intensity_col = next(c for c in df_input.columns if "intensity" in c.lower() or "value" in c.lower() or "abs" in c.lower())
    except StopIteration:
        raise KeyError(f"Could not detect energy/intensity columns in {df_input.columns}")

    # Interpolate + build cdf
    # Interpolate + build cdf
    from scipy.interpolate import interp1d
    f = interp1d(df_input[energy_col], df_input[intensity_col], fill_value="extrapolate")
    x_new = np.linspace(ENERGY_MIN, ENERGY_MAX, 100)
    y_new = f(x_new)
    cdf = np.cumsum(y_new)
    norm_cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())

    # Build a new dataframe with one row containing the cdf list
    df_processed = pd.DataFrame({"cdf": [norm_cdf]})

    X_exp = np.array(df_processed["cdf"].to_list())
    return df_processed, X_exp


# === Loading Models ===
def resolve_models_dir(models_dir: str | Path | None = None) -> Path:
    """
    Resolve the Models directory. By default, looks for a sibling 'Models' folder
    one level above this file (project root / Models).
    """
    if models_dir is None:
        base_dir = Path(__file__).resolve().parent.parent  # backend/.. → project root
        return base_dir / "Models"
    return Path(models_dir)

def load_models(models_dir: str | Path | None = None):
    """
    Load oxidation and bond length models from the Models directory.
    Also builds SHAP explainers (same as your current behavior).
    """
    models_path = resolve_models_dir(models_dir)
    ox_path = models_path / "oxidation_regressor.pkl"
    bl_path = models_path / "bondlength_regressor.pkl"

    with open(ox_path, "rb") as f:
        ox_model = pickle.load(f)
    with open(bl_path, "rb") as f:
        bl_model = pickle.load(f)

    explainer_ox = shap.TreeExplainer(ox_model)
    explainer_bl = shap.TreeExplainer(bl_model)
    return ox_model, bl_model, explainer_ox, explainer_bl

# === Helper Functions ===
def top_contributors(expected, pred, shap_vals, energy_bins, top_k: int = TOP_K):
    direction = np.sign(pred - expected)
    mask = shap_vals * direction > 0
    shap_filtered = shap_vals * mask
    idx_sorted = np.argsort(-np.abs(shap_filtered))[:top_k]
    return [(float(energy_bins[j]), shap_vals[j]) for j in idx_sorted if shap_filtered[j] != 0]

def rf_uncertainty(rf_model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_preds = np.stack([tree.predict(X) for tree in rf_model.estimators_], axis=0)
    return all_preds.mean(axis=0), all_preds.std(axis=0)

# === Core Functions ===
def run_predictions(X_exp: np.ndarray,
                    ox_model,
                    bl_model,
                    explainer_ox,
                    explainer_bl,
                    energy_min: float = ENERGY_MIN,
                    energy_max: float = ENERGY_MAX):
    """
    Runs predictions, SHAP, and uncertainty for each row in X_exp.
    Prints the same text as your script AND returns a structured dict.
    """
    n_bins = X_exp.shape[1]
    energy_bins = np.linspace(energy_min, energy_max, n_bins)

    # Predictions
    ox_preds = ox_model.predict(X_exp)
    bl_preds = bl_model.predict(X_exp)

    # SHAP values
    shap_values_ox = explainer_ox.shap_values(X_exp)
    shap_values_bl = explainer_bl.shap_values(X_exp)

    # Handle expected_value being scalar or array
    def _exp_val(ev):
        try:
            return ev[0]
        except Exception:
            return ev

    results = []
    for i in range(len(X_exp)):
        ox_pred = ox_preds[i]
        bl_pred = bl_preds[i]

        # Uncertainty
        _, ox_std = rf_uncertainty(ox_model, X_exp[i].reshape(1, -1))
        _, bl_std = rf_uncertainty(bl_model, X_exp[i].reshape(1, -1))

        # SHAP top features
        ox_exp_val = _exp_val(explainer_ox.expected_value)
        bl_exp_val = _exp_val(explainer_bl.expected_value)

        top_ox = top_contributors(ox_exp_val, ox_pred, shap_values_ox[i], energy_bins)
        top_bl = top_contributors(bl_exp_val, bl_pred, shap_values_bl[i], energy_bins)

        # --- Print (preserve your current console output) ---
        print(f"\n=== Spectrum {i} ===")
        print(f"Predicted Oxidation State: {ox_pred:.3f} ± {ox_std[0]:.3f}")
        print("Top 20 energy bins driving oxidation prediction:")
        for e, s in top_ox:
            print(f"  {e:.2f} eV (SHAP: {s:.4f})")

        print(f"Predicted Bond Length: {bl_pred:.3f} ± {bl_std[0]:.3f} Å")
        print("Top 20 energy bins driving bond length prediction:")
        for e, s in top_bl:
            print(f"  {e:.2f} eV (SHAP: {s:.4f})")

        # Collect a row for programmatic use
        results.append({
            "index": i,
            "ox_pred": float(ox_pred),
            "ox_std": float(ox_std[0]),
            "bl_pred": float(bl_pred),
            "bl_std": float(bl_std[0]),
            "top_ox": [(float(e), float(s)) for e, s in top_ox],
            "top_bl": [(float(e), float(s)) for e, s in top_bl],
        })

    return {
        "energy_bins": energy_bins.tolist(),
        "results": results,
    }

# === Pipeline ===
def run_pipeline(uploaded_file_or_df,
                 models_dir: str | Path | None = None,
                 limit: int | None = None):
    """
    End-to-end runner:
      1) read user data (JSON with 'cdf')
      2) load models from Models/
      3) predict + SHAP + uncertainty
      4) print same output and return a structured dict
    """
    df_input, X_exp = load_input(uploaded_file_or_df, limit=limit)
    ox_model, bl_model, explainer_ox, explainer_bl = load_models(models_dir)
    return run_predictions(X_exp, ox_model, bl_model, explainer_ox, explainer_bl)