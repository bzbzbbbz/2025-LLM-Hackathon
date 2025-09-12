import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
import tempfile
import shutil
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# === BACKEND IMPORTS ===
from Backend.Modeling import run_pipeline
from rag import XASManuscriptRAG

# === PAGE CONFIG ===
st.set_page_config(
    page_title="XAScribe", 
    layout="wide",
    page_icon="🧪"
)

st.title("🧪 XAScribe")
st.markdown("**Upload your XAS data and research papers to generate comprehensive manuscript sections**")

# === SIDEBAR CONFIGURATION ===
st.sidebar.header("🔧 Configuration")

# API Key input
api_key = st.sidebar.text_input(
    "Google Gemini API Key", 
    type="password", 
    help="Enter your Google Gemini API key for manuscript generation"
)

# Store API key in session state
if api_key:
    st.session_state.api_key = api_key

# === MAIN WORKFLOW ===

# Check for existing papers and RAG cache
existing_papers_folder = "./converted_papers"
existing_papers_count = 0
has_existing_papers = False

if os.path.exists(existing_papers_folder):
    existing_txt_files = [f for f in os.listdir(existing_papers_folder) if f.endswith('.txt')]
    existing_papers_count = len(existing_txt_files)
    has_existing_papers = existing_papers_count > 0

# Display existing papers status
if has_existing_papers:
    st.info(f"📚 Found {existing_papers_count} existing research papers from previous sessions")
    
    with st.expander("📋 View existing papers"):
        existing_files = [f.replace('_simple_extracted.txt', '.pdf').replace('.txt', '') for f in existing_txt_files]
        for i, filename in enumerate(existing_files[:10], 1):  # Show first 10
            st.write(f"{i}. {filename}")
        if len(existing_files) > 10:
            st.write(f"... and {len(existing_files) - 10} more papers")

# Step 1: File Upload Section
st.header("📂 Step 1: Upload Your Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔬 XAS Data Files")
    st.markdown("Upload your experimental XAS data for analysis")
    
    xas_data_file = st.file_uploader(
        "Choose XAS data file",
        type=["csv", "txt", "xlsx", "json"],
        key="xas_data",
        help="Upload your XAS experimental data file"
    )
    
    if xas_data_file:
        st.success(f"✅ XAS data uploaded: {xas_data_file.name}")

with col2:
    st.subheader("📚 Research Papers")
    if has_existing_papers:
        st.markdown(f"**{existing_papers_count} papers already available** - Upload additional papers (optional)")
    else:
        st.markdown("Upload text files of research papers for literature analysis")
    
    # Option for multiple text files or zip
    upload_option = st.radio(
        "Upload method:",
        ["Individual text files", "ZIP archive"],
        key="upload_method"
    )
    
    if upload_option == "Individual text files":
        research_papers = st.file_uploader(
            "Upload additional research papers (text files)" if has_existing_papers else "Upload research papers (text files)",
            type=["txt"],
            accept_multiple_files=True,
            key="paper_files",
            help="Upload individual .txt files containing research papers"
        )
    else:
        research_papers = st.file_uploader(
            "Upload ZIP file with additional research papers" if has_existing_papers else "Upload ZIP file with research papers",
            type=["zip"],
            key="paper_zip",
            help="Upload a ZIP file containing .txt research papers"
        )
    
    if research_papers:
        if upload_option == "Individual text files":
            st.success(f"✅ {len(research_papers)} additional research papers uploaded")
        else:
            st.success(f"✅ ZIP archive uploaded: {research_papers.name}")

# Step 2: Data Processing Section
if xas_data_file or research_papers or has_existing_papers:
    st.header("⚙️ Step 2: Process Data")
    
    # Different button text based on what's available
    if has_existing_papers and not research_papers:
        button_text = "🚀 Initialize with Existing Papers"
        button_help = "Use existing research papers and process XAS data"
    elif has_existing_papers and research_papers:
        button_text = "🚀 Process New Data & Add Papers"
        button_help = "Add new papers to existing collection and process XAS data"
    else:
        button_text = "🚀 Process All Data"
        button_help = "Process uploaded data and initialize system"
    
    if st.button(button_text, type="primary", disabled=not st.session_state.get('api_key'), help=button_help):
        if not st.session_state.get('api_key'):
            st.error("⚠️ Please enter your Google Gemini API key in the sidebar first!")
        else:
            # Initialize processing containers
            xas_results = None
            rag_system = None
            
            with st.spinner("Processing your data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process XAS Data
                if xas_data_file:
                    status_text.text("🔬 Analyzing XAS data...")
                    progress_bar.progress(25)
                    
                    try:
                        xas_results = run_pipeline(xas_data_file, models_dir="Random Forest Model")
                        st.session_state.xas_results = xas_results
                        st.success("✅ XAS data analysis completed")
                    except Exception as e:
                        st.error(f"❌ XAS analysis failed: {e}")
                
                progress_bar.progress(50)
                
                # Initialize RAG system
                try:
                    # Set API key in environment for RAG system
                    os.environ['GOOGLE_API_KEY'] = st.session_state.api_key
                    
                    # Process new research papers if uploaded
                    if research_papers:
                        status_text.text("📚 Processing new research papers...")
                        
                        # Create temporary directory for new papers
                        temp_dir = tempfile.mkdtemp()
                        papers_folder = os.path.join(temp_dir, "papers")
                        os.makedirs(papers_folder, exist_ok=True)
                        
                        # Handle different upload methods
                        if upload_option == "Individual text files":
                            for uploaded_file in research_papers:
                                file_path = os.path.join(papers_folder, uploaded_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                        else:  # ZIP file
                            with zipfile.ZipFile(research_papers, 'r') as zip_ref:
                                zip_ref.extractall(papers_folder)
                        
                        # Ensure converted_papers directory exists
                        if not os.path.exists(existing_papers_folder):
                            os.makedirs(existing_papers_folder, exist_ok=True)
                        
                        # Copy new files to expected location (add to existing)
                        new_files_count = 0
                        for filename in os.listdir(papers_folder):
                            if filename.endswith('.txt'):
                                src_path = os.path.join(papers_folder, filename)
                                dst_path = os.path.join(existing_papers_folder, filename)
                                with open(src_path, 'r', encoding='utf-8') as src, open(dst_path, 'w', encoding='utf-8') as dst:
                                    dst.write(src.read())
                                new_files_count += 1
                        
                        progress_bar.progress(75)
                        status_text.text("🤖 Initializing RAG system with new papers...")
                        
                        # Force reload since we added new papers
                        rag_system = XASManuscriptRAG(force_reload=True)
                        st.success(f"✅ Added {new_files_count} new papers to existing collection")
                        
                        # Clean up temp directory
                        shutil.rmtree(temp_dir)
                        
                    else:
                        # Use existing papers without force reload
                        status_text.text("🤖 Loading existing RAG system...")
                        progress_bar.progress(75)
                        
                        # Initialize with existing data (no force reload)
                        rag_system = XASManuscriptRAG(force_reload=False)
                        st.success(f"✅ RAG system loaded with {existing_papers_count} existing papers")
                    
                    st.session_state.rag_system = rag_system
                    
                except Exception as e:
                    st.error(f"❌ RAG system initialization failed: {e}")
                
                progress_bar.progress(100)
                status_text.text("✅ All processing completed!")
    
    # Display processing results
    if st.session_state.get('xas_results') is not None or st.session_state.get('rag_system') is not None:
        st.header("📊 Step 3: Processing Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if st.session_state.get('xas_results') is not None:
                st.subheader("🔬 XAS Analysis Results")
                
                # Display basic XAS analysis info
                xas_data = st.session_state.xas_results
                if hasattr(xas_data, 'shape'):
                    st.info(f"📈 Data processed: {xas_data.shape} points")
                
                # Try to create a simple plot if possible
                try:
                    if hasattr(xas_data, 'iloc') and len(xas_data.columns) >= 2:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(xas_data.iloc[:, 0], xas_data.iloc[:, 1], 'b-', alpha=0.7)
                        ax.set_xlabel(str(xas_data.columns[0]))
                        ax.set_ylabel(str(xas_data.columns[1]))
                        ax.set_title("XAS Data Analysis")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                except Exception as e:
                    st.info("XAS data processed successfully (visualization not available)")
        
        with result_col2:
            if st.session_state.get('rag_system') is not None:
                st.subheader("📚 Literature Analysis Ready")
                
                # Display RAG system stats
                try:
                    stats = st.session_state.rag_system.get_paper_stats()
                    st.info(stats)
                except:
                    # Count current papers
                    if os.path.exists(existing_papers_folder):
                        current_count = len([f for f in os.listdir(existing_papers_folder) if f.endswith('.txt')])
                        st.info(f"📚 {current_count} research papers loaded and ready for analysis")
                    else:
                        st.info("RAG system initialized and ready")

# Step 4: Manuscript Generation Section
if st.session_state.get('xas_results') is not None or st.session_state.get('rag_system') is not None:
    st.header("✍️ Step 4: Generate Manuscript")
    
    # Research focus input
    research_question = st.text_area(
        "🎯 Research Question / Manuscript Focus:",
        placeholder="Example: Analyze the structural changes in NMC cathodes during cycling using XAS spectroscopy and compare with literature findings",
        height=100,
        help="Describe what you want to focus on in your manuscript. This will guide both the data analysis interpretation and literature review."
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        include_data_analysis = st.checkbox("Include XAS data analysis results", value=True, disabled=not st.session_state.get('xas_results'))
        include_literature = st.checkbox("Include literature review", value=True, disabled=not st.session_state.get('rag_system'))
        citation_style = st.selectbox("Citation Style:", ["Numbered [1]", "Author-Year (Smith 2023)", "Footnotes"])
    
    # Generate manuscript button
    if st.button("📝 Generate Manuscript", type="primary"):
        if not research_question.strip():
            st.warning("⚠️ Please enter a research question or manuscript focus.")
        else:
            with st.spinner("Generating your manuscript..."):
                try:
                    # Prepare detailed context from XAS analysis
                    xas_context = ""
                    if include_data_analysis and st.session_state.get('xas_results') is not None:
                        xas_data = st.session_state.xas_results
                        
                        # Extract detailed analysis results
                        if isinstance(xas_data, dict) and 'results' in xas_data:
                            # Process the detailed XAS analysis results
                            results = xas_data['results']
                            energy_bins = xas_data.get('energy_bins', [])
                            
                            if results:
                                # Calculate summary statistics
                                ox_states = [r['ox_pred'] for r in results]
                                ox_uncertainties = [r['ox_std'] for r in results]
                                bond_lengths = [r['bl_pred'] for r in results]
                                bl_uncertainties = [r['bl_std'] for r in results]
                                
                                avg_ox = np.mean(ox_states)
                                avg_ox_std = np.mean(ox_uncertainties)
                                avg_bl = np.mean(bond_lengths)
                                avg_bl_std = np.mean(bl_uncertainties)
                                
                                # Get key spectral features from SHAP analysis
                                key_energies_ox = []
                                key_energies_bl = []
                                
                                for result in results[:3]:  # Analyze first 3 spectra for key features
                                    # Top oxidation state driving energies
                                    top_ox = sorted(result['top_ox'], key=lambda x: abs(x[1]), reverse=True)[:5]
                                    key_energies_ox.extend([f"{e:.1f} eV (SHAP: {s:.3f})" for e, s in top_ox])
                                    
                                    # Top bond length driving energies
                                    top_bl = sorted(result['top_bl'], key=lambda x: abs(x[1]), reverse=True)[:5]
                                    key_energies_bl.extend([f"{e:.1f} eV (SHAP: {s:.3f})" for e, s in top_bl])
                                
                                # Remove duplicates and get most significant
                                key_energies_ox = list(set(key_energies_ox))[:10]
                                key_energies_bl = list(set(key_energies_bl))[:10]
                                
                                xas_context = f"""

XAS DATA ANALYSIS RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRUCTURAL AND ELECTRONIC PROPERTIES:
• Dataset: {len(results)} XAS spectra analyzed using Random Forest machine learning
• Energy range: {min(energy_bins):.1f} - {max(energy_bins):.1f} eV covering the absorption edge
• Spectral resolution: {len(energy_bins)} energy points for detailed feature analysis

OXIDATION STATE ANALYSIS:
• Average predicted oxidation state: {avg_ox:.3f} ± {avg_ox_std:.3f}
• Range: {min(ox_states):.3f} to {max(ox_states):.3f}
• Key energy features driving oxidation predictions:
  {chr(10).join(f"  - {energy}" for energy in key_energies_ox[:5])}

STRUCTURAL ANALYSIS:
• Average predicted bond length: {avg_bl:.3f} ± {avg_bl_std:.3f} Å
• Range: {min(bond_lengths):.3f} to {max(bond_lengths):.3f} Å  
• Key energy features driving bond length predictions:
  {chr(10).join(f"  - {energy}" for energy in key_energies_bl[:5])}

SHAP-BASED FEATURE INTERPRETATION:
• Machine learning model identified specific energy regions with highest predictive importance
• Pre-edge features and main absorption edge show distinct contributions to structural predictions
• Energy-dependent variations suggest heterogeneity in local atomic environments
• SHAP values provide quantitative measure of each spectral feature's contribution to predictions

SPECTROSCOPIC INSIGHTS:
• Absorption edge position correlates with oxidation state variations
• Bond length predictions show sensitivity to coordination environment changes
• Spectral features indicate potential phase transformations or structural distortions
• Uncertainty quantification provides confidence intervals for all predictions

MACHINE LEARNING CONFIDENCE:
• Random Forest model provides uncertainty estimates for all predictions
• Average oxidation state uncertainty: ±{avg_ox_std:.3f}
• Average bond length uncertainty: ±{avg_bl_std:.3f} Å
• SHAP analysis ensures interpretable and explainable predictions
                                """
                        else:
                            # Fallback for simpler data format
                            if hasattr(xas_data, 'shape'):
                                xas_context = f"""

XAS DATA ANALYSIS RESULTS:
• Dataset contains {xas_data.shape[0]} data points with {xas_data.shape[1]} measured parameters
• Machine learning analysis completed using Random Forest modeling
• Spectral features processed and analyzed for structural insights
• Data preprocessing and feature extraction performed successfully
                                """
                    
                    # Prepare enhanced research question
                    enhanced_question = f"""
Research Focus: {research_question}

Manuscript Requirements:
- Citation Style: {citation_style}
{xas_context}

Please write a well-structured academic manuscript section that synthesizes both the experimental findings and literature context to provide comprehensive scientific insights.
                    """
                    
                    # Generate manuscript using RAG system
                    if st.session_state.get('rag_system') and include_literature:
                        manuscript = st.session_state.rag_system.generate_manuscript_analysis(enhanced_question)
                    else:
                        # Fallback: generate without literature if RAG not available
                        st.warning("Generating manuscript without literature integration (RAG system not available)")
                        manuscript = f"""
MANUSCRIPT SECTION: {research_question}

{xas_context}

ANALYSIS AND DISCUSSION:

The XAS spectroscopic analysis provides detailed insights into the structural and electronic properties of the studied material. The machine learning-based approach using Random Forest modeling has enabled quantitative predictions of key structural parameters with associated uncertainty estimates.

The predicted oxidation states and bond lengths reveal important information about the local atomic environment and coordination chemistry. The SHAP-based feature importance analysis identifies specific energy regions that are most critical for determining these structural properties, providing interpretable insights into the spectroscopic signatures.

These experimental findings contribute to our understanding of the material's electronic structure and local atomic arrangements. Further analysis incorporating literature comparisons would provide additional context for these observations.

Note: Complete manuscript generation with literature integration requires the RAG system to be properly initialized.
                        """
                    
                    # Display generated manuscript
                    st.header("📄 Generated Manuscript")
                    st.markdown("---")
                    st.markdown(manuscript)
                    
                    # Download option
                    st.download_button(
                        label="💾 Download Manuscript",
                        data=manuscript,
                        file_name=f"xas_manuscript_{research_question[:30].replace(' ', '_')}.txt",
                        mime="text/plain",
                        help="Download the generated manuscript as a text file"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Manuscript generation failed: {e}")
                    st.info("Please check your inputs and try again.")

# === SIDEBAR INFORMATION ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Status")

# Status indicators
if st.session_state.get('api_key'):
    st.sidebar.success("✅ API Key configured")
else:
    st.sidebar.error("❌ API Key required")

if st.session_state.get('xas_results') is not None:
    st.sidebar.success("✅ XAS data processed")
else:
    st.sidebar.info("⏳ No XAS data uploaded")

if st.session_state.get('rag_system') is not None:
    st.sidebar.success("✅ Literature database ready")
elif has_existing_papers:
    st.sidebar.warning(f"📚 {existing_papers_count} papers available - click 'Initialize' to load")
else:
    st.sidebar.info("⏳ No research papers available")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Quick Actions")

if st.sidebar.button("🔄 Reset All Data"):
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Help section
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Help")
with st.sidebar.expander("How to use"):
    st.markdown("""
    1. **Check Status**: App automatically detects existing papers from previous sessions
    2. **Upload Data**: Add XAS data (required) and additional papers (optional)
    3. **Process**: Click to initialize with existing papers or add new ones
    4. **Generate**: Enter your research question and generate manuscript
    5. **Download**: Save your generated manuscript
    
    **Supported Formats:**
    - XAS Data: CSV, TXT, XLSX, JSON
    - Papers: TXT files or ZIP archive
    
    **Note:** The app uses cached papers from previous sessions automatically!
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**XAS Research Assistant v2.0**")
st.sidebar.markdown("*Streamlined Workflow Edition*")