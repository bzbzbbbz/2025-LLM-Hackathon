import os
import google.generativeai as genai
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import json
from dotenv import load_dotenv
import pickle

# Configuration
COLLECTION_NAME = "xas_papers"
CONVERTED_FOLDER = os.path.join(os.path.dirname(__file__), "converted_papers")
CHUNK_SIZE = 1000  # Larger chunks for better context in manuscript analysis
CHUNK_OVERLAP = 200
CACHE_FILE = os.path.join(os.path.dirname(__file__), "rag_cache.pkl")
load_dotenv()

class XASManuscriptRAG:
    def __init__(self, force_reload=False):
        self.citations = None
        self.model = None
        self.embed_model = None
        self.index = None
        self.texts = []
        self.filenames = []
        self.embeddings = None
        self.temperature = 0

        # Try to load from cache first
        if not force_reload and self.load_cache():
            print("✅ Loaded from cache - skipping paper processing")
            self.setup_gemini()
            self.setup_embeddings()  # Always need embeddings for search

        else:
            print("📚 Loading papers from scratch...")
            metadata_path = os.path.join(os.path.dirname(__file__), "paper_metadata.csv")
            self.citations = pd.read_csv(metadata_path).set_index('filename').to_dict('index')
            self.setup_gemini()
            self.setup_embeddings()
            self.load_papers()
            self.save_cache()

    def setup_gemini(self):
        """Initialize Gemini Pro"""
        try:
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.temperature = 0.2
            print("Gemini Pro configured")
        except Exception as e:
            print(f"Gemini setup failed: {e}")
            raise

    def setup_faiss(self):
        """Setup FAISS index"""
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity with normalized vectors
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        print("FAISS index created with", self.index.ntotal, "vectors")


    def setup_embeddings(self):
        """Setup embedding model"""
        try:
            self.embed_model = SentenceTransformer("BAAI/bge-large-en")
            print("Embedding model loaded")
        except Exception as e:
            print(f"Embedding model failed: {e}")
            raise

    def save_cache(self):
        """Save the loaded data to cache"""
        try:
            cache_data = {
                "citations": self.citations,
                "texts": self.texts,
                "filenames": self.filenames,
                "embeddings": self.embeddings,  # numpy array is fine for pickle
            }
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"Cache saved to {CACHE_FILE}")
            return True
        except Exception as e:
            print(f"Failed to save cache: {e}")
            return False

    def load_cache(self):
        """Load data from cache if available (and rebuild FAISS)"""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "rb") as f:
                    cache_data = pickle.load(f)
                self.citations = cache_data.get("citations")
                self.texts = cache_data.get("texts", [])
                self.filenames = cache_data.get("filenames", [])
                self.embeddings = cache_data.get("embeddings", None)

                # Rebuild FAISS if we have embeddings
                if self.embeddings is not None and len(self.texts) == len(self.filenames) and len(self.texts) > 0:
                    if not isinstance(self.embeddings, np.ndarray):
                        self.embeddings = np.array(self.embeddings, dtype="float32")
                    else:
                        self.embeddings = self.embeddings.astype("float32")
                    self.setup_faiss()
                    print(f"Cache loaded from {CACHE_FILE}")
                    return True
                else:
                    print("Cache present but incomplete; will rebuild from papers.")
                    return False
            return False
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def debug_metadata(self):
        """Debug function to check metadata structure"""
        if self.citations:
            print("📊 Metadata Debug Info:")
            print("=" * 40)
            print(f"Total files in metadata: {len(self.citations)}")
            sample_files = list(self.citations.keys())[:3]  # Show first 3 files
            for filename in sample_files:
                meta = self.citations[filename]
                print(f"\nCSV Filename: {filename}")
                print(f"Available columns: {list(meta.keys())}")
                for key, value in meta.items():
                    print(f"  {key}: {value}")

            # Show some example filenames that might be in the system
            print(f"\nExample CSV filenames:")
            for i, filename in enumerate(list(self.citations.keys())[:5]):
                print(f"  {i+1}. {filename}")
            print("=" * 40)
        else:
            print("No citations metadata loaded")

    def load_papers(self):
        """Load and process all XAS papers"""
        if not os.path.exists(CONVERTED_FOLDER):
            print(f"Folder {CONVERTED_FOLDER} not found!")
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "]
        )

        texts, filenames = [], []
        file_count = 0

        for filename in os.listdir(CONVERTED_FOLDER):
            if filename.endswith(".txt"):
                file_count += 1
                print(f"Processing {filename}...")

                try:
                    with open(os.path.join(CONVERTED_FOLDER, filename), "r", encoding="utf-8") as f:
                        content = f.read()

                    if not content.strip():
                        continue

                    chunks = splitter.split_text(content)

                    for chunk in chunks:
                        if len(chunk.strip()) > 200:  # Minimum chunk size for meaningful analysis
                            texts.append(chunk.strip())
                            filenames.append(filename.replace('_simple_extracted.txt', '.pdf'))

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

        if not texts:
            print("No valid chunks found!")
            return

        print(f"Processing {len(texts)} chunks from {file_count} papers...")

        # Generate embeddings in batches
        embeddings = self.embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        # After you computed 'embeddings'
        self.texts = texts
        self.filenames = filenames
        self.embeddings = embeddings.astype("float32")

        # Setup FAISS index
        self.setup_faiss()

        print(f"Loaded {len(texts)} chunks from {len(set(filenames))} papers")


    def search_papers(self, query, k=15):
        if self.embed_model is None:
            raise ValueError("Embedding model not initialized. Please run setup_embeddings() first.")
        if self.index is None:
            raise ValueError("FAISS index not initialized. Please run load_papers() first.")

        query_embedding = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.texts):
                results.append((self.texts[idx], self.filenames[idx], float(score)))
        return results


    def generate_manuscript_analysis(self, research_question, model_results=None, raw_results=False):
        """Generate comprehensive manuscript-level analysis for XAS research"""
        print(f"Analyzing: {research_question}")

        # Search for relevant content (reduced for focused analysis)
        chunks = self.search_papers(research_question, k=20)

        if not chunks:
            return "No relevant XAS literature found for analysis."

        # Group by paper to ensure diverse coverage
        papers = {}
        for text, filename, score in chunks:
            if filename not in papers:
                papers[filename] = []
            papers[filename].append((text, score))

        # Select best chunks from each paper (max 2 per paper for focused analysis)
        selected_chunks = []
        for filename, paper_chunks in papers.items():
            paper_chunks.sort(key=lambda x: x[1], reverse=True)
            selected_chunks.extend([(text, filename, score) for text, score in paper_chunks[:2]])

        selected_chunks = selected_chunks[:6]  # limit to ~5–6 papers

        # Build context with citations
        context_parts = []
        used_citations = []
        citation_map = {}

        for i, (text, filename, score) in enumerate(selected_chunks, 1):
            meta = self.citations.get(filename, {})
            title = meta.get('title', filename)
            if filename not in citation_map:
                citation_map[filename] = len(used_citations) + 1
                citation_text = meta.get('citation', title)
                used_citations.append(f"[{citation_map[filename]}] {citation_text}")

            citation_num = citation_map[filename]
            context_parts.append(f"From '{title}' [{citation_num}]:\n{text}")

        full_context = "\n\n---\n\n".join(context_parts)

        # 🔑 Two options: nicely formatted or raw JSON
        if model_results:
            if raw_results:
                model_results_text = json.dumps(model_results, indent=2)
            else:
                lines = []
                for r in model_results.get("results", []):
                    lines.append(
                        f"- Spectrum {r['index']}: "
                        f"Oxidation = {r['ox_pred']:.2f} ± {r['ox_std']:.2f}, "
                        f"Bond Length = {r['bl_pred']:.2f} ± {r['bl_std']:.2f} Å"
                    )
                model_results_text = "\n".join(lines)
        else:
            model_results_text = "No model predictions were provided."

        # Manuscript-focused prompt
        prompt = f"""You are writing a focused manuscript paragraph analyzing XAS (X-ray Absorption Spectroscopy) research.

    Based on the following literature from {len(selected_chunks)} key research papers, provide a concise but thorough academic analysis addressing: {research_question}

LITERATURE CONTEXT:
{full_context}


Write a single, well-structured manuscript paragraph that synthesizes the key insights. Focus on:

1. **Key Findings**: Main discoveries and insights from the literature and how they compare to the model outputs
2. **Technical Insights**: Specific XAS spectral features and interpretations
3. **Comparative Analysis**: How different studies compare and what consensus exists
4. **Implications**: What these findings mean for the field

IMPORTANT CITATION INSTRUCTIONS:
- Use numbered citations [1], [2], [3], etc. in your text when referencing specific findings
- The citation numbers correspond to the papers listed in the context above
- Integrate citations naturally into your analysis text
- Keep the analysis concise but comprehensive - aim for 1–2 well-developed paragraphs

MANUSCRIPT ANALYSIS:"""

        try:
            print("Generating manuscript analysis...")
            response = self.model.generate_content(prompt)

            if not response.text:
                return "Failed to generate analysis."

            # Format the final output with proper citations
            references_section = "\n\nREFERENCES:\n" + "\n".join(used_citations) if used_citations else ""
            analysis = f"""MANUSCRIPT ANALYSIS: {research_question}

Literature Coverage: {len(papers)} papers, {len(selected_chunks)} text segments analyzed

{response.text}{references_section}

---
Sources Analyzed: {', '.join(papers.keys())}
"""

            return analysis

        except Exception as e:
            return f"Error generating analysis: {str(e)}"

# Main execution
def main(force_reload=False):
    """Initialize and run XAS manuscript analysis system"""
    print("🧪 XAS Manuscript RAG System")
    print("=" * 50)

    try:
        # Initialize system (with caching)
        rag = XASManuscriptRAG(force_reload=force_reload)

        # Generate manuscript analysis
        research_question = "What can XAS reveal about structural changes in NMC cathodes during cycling?"

        analysis = rag.generate_manuscript_analysis(research_question)

        print("\n" + "=" * 60)
        print("MANUSCRIPT ANALYSIS OUTPUT")
        print("=" * 60)
        print(analysis)

        return rag

    except Exception as e:
        print(f"❌ System error: {e}")
        return None

def quick_analysis(question, force_reload=False):
    """Quick analysis function for testing different questions"""
    print(f"🔍 Quick Analysis: {question}")
    print("=" * 50)

    try:
        # Initialize system (with caching)
        rag = XASManuscriptRAG(force_reload=force_reload)

        # Generate analysis
        analysis = rag.generate_manuscript_analysis(question)

        print("\n" + "=" * 60)
        print("ANALYSIS OUTPUT")
        print("=" * 60)
        print(analysis)

        return rag

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return None

# Execute
if __name__ == "__main__":
    rag_system = main()

    print("\n🎉 System ready for XAS manuscript analysis!")
    print("\nUsage options:")
    print("1. rag_system.generate_manuscript_analysis('your research question')")
    print("2. quick_analysis('your research question')  # For quick tests")
    print("3. main(force_reload=True)  # To reload papers from scratch")
    print("\n💡 First run loads papers, subsequent runs use cache for faster startup!")