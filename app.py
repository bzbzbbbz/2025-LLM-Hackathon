import streamlit as st

# === BACKEND IMPORTS ===
from Backend.Modeling import run_pipeline
from Backend.RAG_Model import XASManuscriptRAG

# session state init
if "rag" not in st.session_state:
    st.session_state.rag = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None

st.set_page_config(page_title="XAS Chatbot Assistant", layout="wide")
st.title("XAS Chatbot Assistant")

# === File upload section ===
st.header("Upload your data file")
uploaded_file = st.file_uploader(
    "Choose a data file",
    type=["csv", "txt", "xlsx", "json"]
)

# === Running Modeling.py ===
if uploaded_file is not None:
    results = run_pipeline(uploaded_file, models_dir="Random Forest Model")
    st.session_state.model_results = results   # <-- store Modeling.py output
    st.session_state.rag = XASManuscriptRAG()  # <-- init RAG
    st.success(f"File `{uploaded_file.name}` uploaded successfully!")


# --- Chat section ---
st.header("Chatbot")
prompt = st.chat_input("Ask me about your dataset...")

if prompt:
    st.chat_message("user").write(prompt)
    if st.session_state.rag is None:
        st.chat_message("assistant").write("Please upload a file first.")
    else:
        analysis = st.session_state.rag.generate_manuscript_analysis(
            research_question=prompt,
            model_results=st.session_state.model_results,
            raw_results=False
        )
        st.chat_message("assistant").write(analysis)
