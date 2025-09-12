import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === BACKEND IMPORTS ===
from Backend.Modeling import run_pipeline
from Backend.Standardization import preprocessing

st.set_page_config(page_title="XAS Chatbot Assistant", layout="wide")
st.title("XAS Chatbot Assistant")

# === File upload section ===
st.header("Upload your data file")
uploaded_file = st.file_uploader(
    "Choose a data file",
    type=["csv", "txt", "xlsx", "json"]
)

# === Running Modeling.py ===
df = None
if uploaded_file is not None:
    #df = preprocessing(uploaded_file)
    out = run_pipeline(uploaded_file, models_dir=None)
    st.success(f"File `{uploaded_file.name}` uploaded successfully!")
    #out = run_pipeline(df, models_dir=None)



# --- Chat section ---
st.header("Chatbot")
prompt = st.chat_input("Ask me about your dataset...")

if prompt:
    st.chat_message("user").write(prompt)

    if df is None:
        st.chat_message("assistant").write("Please upload a file first.")
    else:
        # Example: simple response + plot of the first two columns (if numeric)
        st.chat_message("assistant").write("Here’s a quick plot of the first two columns:")

        try:
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]

            fig, ax = plt.subplots()
            ax.plot(x, y, label="Uploaded Data")
            ax.set_xlabel(str(df.columns[0]))
            ax.set_ylabel(str(df.columns[1]))
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.chat_message("assistant").write(f"Could not plot data: {e}")