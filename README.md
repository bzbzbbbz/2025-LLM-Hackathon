# 🧪 XAScribe

AI-powered research assistant for X-ray Absorption Spectroscopy (XAS) analysis and manuscript generation using Google Gemini 2.5 Flash and RAG technology.

## Features

- **XAS Data Analysis**: Upload and analyze experimental data with machine learning models
- **Literature Processing**: Process research papers for context-aware manuscript generation  
- **AI Manuscript Generation**: Create academic content with proper citations
- **Interactive Chat**: Ask questions about your research data and literature

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Google Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create an API key for Gemini
3. Keep it ready for the app interface

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Access the App
Open your browser to the URL shown in terminal (typically `http://localhost:8501`)

## Usage

1. **Enter API Key** in the sidebar
2. **Upload Data**: 
   - XAS data files (JSON, CSV, XLSX)
   - Research papers (individual files or ZIP)
3. **Process Data** to initialize the system
4. **Generate Content** using the manuscript generator or chat interface

## Example Data

The project includes:
- `example_xas_data/nmc_exp_xas.json` - Sample XAS experimental data
- `example_papers/` - 24 research papers about XAS studies on battery cathodes
- `paper_metadata.csv` - Citation metadata for proper referencing

## File Structure

```
├── app.py                    # Main Streamlit application
├── rag.py                    # RAG system for literature processing
├── Backend/Modeling.py       # XAS data analysis pipeline
├── requirements.txt          # Python dependencies
├── paper_metadata.csv        # Citation metadata
├── example_xas_data/         # Sample experimental data
└── example_papers/           # Sample research papers
```

## Configuration

Set your API key via:
- Streamlit sidebar interface (recommended)
- Environment variable: `export GOOGLE_API_KEY="your-key"`

## Technologies

- **Frontend**: Streamlit
- **AI**: Google Gemini 2.5 Flash
- **Vector DB**: Milvus
- **Embeddings**: BGE-large-en
- **ML**: Scikit-learn, SHAP
- **Materials Science**: Pymatgen

## Troubleshooting

- **API Key Error**: Verify your Gemini API key is valid
- **File Format**: Use supported formats (JSON, CSV, XLSX, TXT, ZIP)
- **Memory Issues**: Process smaller batches for large datasets

---

**XAScribe v2.0** - Advanced AI Research Platform for Materials Science
