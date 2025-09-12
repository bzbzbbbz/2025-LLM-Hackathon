import os
import uuid
import time
import pickle
import numpy as np

# Suppress warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import google.generativeai as genai
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import json

# Configuration
CONVERTED_FOLDER = "./converted_papers"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CACHE_FILE = "./rag_cache.pkl"
EMBEDDINGS_FILE = "./embeddings_cache.pkl"

class XASManuscriptRAG:
    def __init__(self, force_reload=False):
        self.citations = None
        self.model = None
        self.embed_model = None
        self.temperature = 0.1
        
        # FAISS vector storage
        self.index = None
        self.texts = []
        self.filenames = []
        self.embeddings = None

        print(f"🔄 Initializing XASManuscriptRAG with force_reload={force_reload}")
        print(f"🔍 Checking cache and database status...")
        print(f"   - Cache file exists: {os.path.exists(CACHE_FILE)}")
        print(f"   - Embeddings file exists: {os.path.exists(EMBEDDINGS_FILE)}")
        print(f"   - Papers folder exists: {os.path.exists(CONVERTED_FOLDER)}")
        if os.path.exists(CONVERTED_FOLDER):
            paper_count = len([f for f in os.listdir(CONVERTED_FOLDER) if f.endswith('.txt')])
            print(f"   - Papers in folder: {paper_count}")
        
        # Try to load from cache first
        cache_loaded = False
        if not force_reload:
            print("📋 Attempting to load from cache...")
            cache_loaded = self.load_cache()
            if cache_loaded:
                print("✅ Cache loaded successfully - proceeding with cached flow")
            else:
                print("❌ Cache loading failed - will load from scratch")
        else:
            print("🔄 Force reload requested - skipping cache")
            
        if not force_reload and cache_loaded:
            print("✅ Loaded from cache - skipping paper processing")
            print("🔌 Setting up Gemini API...")
            self.setup_gemini()
            print("🧠 Loading embedding model...")
            self.setup_embeddings()
            print("📁 Loading vector index...")
            if self.load_vector_index():
                print("✅ Successfully used existing cached data and vectors")
            else:
                print("⚠️ Vector index missing, rebuilding...")
                self.load_papers()
                self.save_cache()
        else:
            if force_reload:
                print("📚 Force reload requested - loading papers from scratch...")
            else:
                print("📚 No cache found - loading papers from scratch...")
            print("📖 Starting fresh paper processing pipeline...")
            self.load_metadata()
            self.setup_gemini()
            self.setup_embeddings()
            print("📄 About to start load_papers() - this will process chunks...")
            self.load_papers()
            self.save_cache()
            print("✅ Fresh processing completed and cache saved")

    def load_metadata(self):
        """Load metadata from CSV file with error handling"""
        metadata_file = "paper_metadata.csv"
        
        try:
            if os.path.exists(metadata_file):
                self.citations = pd.read_csv(metadata_file)
                print(f"✅ Loaded {len(self.citations)} citations from {metadata_file}")
            else:
                print(f"⚠️ Metadata file {metadata_file} not found - will work without citations")
                self.citations = pd.DataFrame()
        except Exception as e:
            print(f"❌ Failed to load metadata: {e}")
            self.citations = pd.DataFrame()

    def setup_gemini(self):
        """Setup Google Gemini API"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.temperature = 0.2
            print("✅ Gemini 2.5 Flash configured")
        except Exception as e:
            print(f"❌ Gemini setup failed: {e}")
            raise

    def setup_embeddings(self):
        """Setup embedding model"""
        try:
            self.embed_model = SentenceTransformer("BAAI/bge-large-en")
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Embedding model failed: {e}")
            raise

    def save_cache(self):
        """Save the loaded data to cache"""
        try:
            cache_data = {
                'citations': self.citations,
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP
            }

            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)

            print(f"✅ Cache saved to {CACHE_FILE}")
            
            # Save vector data separately
            if self.index is not None and len(self.texts) > 0:
                vector_data = {
                    'embeddings': self.embeddings,
                    'texts': self.texts,
                    'filenames': self.filenames
                }
                with open(EMBEDDINGS_FILE, 'wb') as f:
                    pickle.dump(vector_data, f)
                print(f"✅ Vector data saved to {EMBEDDINGS_FILE}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to save cache: {e}")
            return False

    def load_cache(self):
        """Load data from cache if available"""
        print(f"🔍 Checking for cache file: {CACHE_FILE}")
        try:
            if os.path.exists(CACHE_FILE):
                print(f"📁 Cache file found, loading...")
                with open(CACHE_FILE, 'rb') as f:
                    cache_data = pickle.load(f)

                self.citations = cache_data['citations']
                
                print(f"✅ Cache loaded successfully from {CACHE_FILE}")
                if self.citations is not None:
                    citations_type = type(self.citations).__name__
                    if isinstance(self.citations, pd.DataFrame):
                        print(f"   - Citations loaded: {len(self.citations)} papers (DataFrame)")
                    elif isinstance(self.citations, dict):
                        print(f"   - Citations loaded: {len(self.citations)} papers (dict - legacy format)")
                    else:
                        print(f"   - Citations loaded: {citations_type} format")
                else:
                    print(f"   - Citations: None")
                return True
            else:
                print(f"❌ Cache file not found: {CACHE_FILE}")
                return False
        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            return False

    def load_vector_index(self):
        """Load vector index from cache"""
        try:
            if os.path.exists(EMBEDDINGS_FILE):
                print(f"📁 Loading vector data from {EMBEDDINGS_FILE}")
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    vector_data = pickle.load(f)
                
                self.embeddings = vector_data['embeddings']
                self.texts = vector_data['texts']
                self.filenames = vector_data['filenames']
                
                # Rebuild FAISS index
                if self.embeddings is not None and len(self.embeddings) > 0:
                    print(f"🔍 Building FAISS index with {len(self.embeddings)} vectors...")
                    dimension = self.embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                    
                    # Normalize embeddings for cosine similarity
                    normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                    self.index.add(normalized_embeddings.astype('float32'))
                    
                    print(f"✅ FAISS index loaded with {self.index.ntotal} vectors")
                    return True
                else:
                    print("❌ No embeddings found in vector data")
                    return False
            else:
                print(f"❌ Vector file not found: {EMBEDDINGS_FILE}")
                return False
        except Exception as e:
            print(f"❌ Failed to load vector index: {e}")
            return False

    def load_papers(self):
        """Load and process all XAS papers with FAISS"""
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
                        if len(chunk.strip()) > 200:  # Minimum chunk size
                            texts.append(chunk.strip())
                            filenames.append(filename.replace('_simple_extracted.txt', '.pdf'))

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

        if not texts:
            print("No valid chunks found!")
            return

        print(f"Processing {len(texts)} chunks from {file_count} papers...")

        # Generate embeddings
        embeddings = self.embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        # Store data
        self.texts = texts
        self.filenames = filenames
        self.embeddings = embeddings

        # Create FAISS index
        print("🔍 Creating FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add normalized embeddings
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))

        print(f"✅ Loaded {len(texts)} chunks from {len(set(filenames))} papers into FAISS index")

    def search_papers(self, query, k=15):
        """Search for relevant chunks using FAISS"""
        if self.embed_model is None:
            raise ValueError("Embedding model not initialized")

        if self.index is None or len(self.texts) == 0:
            raise ValueError("Vector index not initialized")

        # Generate query embedding
        query_embedding = self.embed_model.encode([query], normalize_embeddings=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search using FAISS
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.texts):  # Ensure valid index
                results.append((
                    self.texts[idx],
                    self.filenames[idx],
                    float(score)
                ))

        return results

    def generate_manuscript_analysis(self, research_question):
        """Generate comprehensive manuscript-level analysis for XAS research"""
        print(f"Analyzing: {research_question}")

        # Search for relevant content
        search_results = self.search_papers(research_question, k=20)
        
        if not search_results:
            return "No relevant papers found for the given research question."

        # Select top chunks with score threshold
        selected_chunks = [(text, filename, score) for text, filename, score in search_results 
                          if score > 0.3][:15]
        
        if not selected_chunks:
            return "No sufficiently relevant content found for the research question."

        # Group by paper and build context
        context_parts = []
        used_citations = []
        citation_map = {}
        citation_counter = 1

        for i, (text, filename, score) in enumerate(selected_chunks, 1):
            # Get citation info from metadata if available
            title = filename
            citation_text = filename
            
            try:
                # Handle both DataFrame and dict cases for self.citations
                if self.citations is not None:
                    if isinstance(self.citations, pd.DataFrame) and not self.citations.empty:
                        citation_row = self.citations[self.citations['filename'] == filename]
                        if not citation_row.empty:
                            title = citation_row.iloc[0].get('title', filename)
                            citation_text = citation_row.iloc[0].get('in_text_citation', filename)
                    elif isinstance(self.citations, dict) and filename in self.citations:
                        # Handle dictionary format (legacy cache compatibility)
                        citation_data = self.citations[filename]
                        if isinstance(citation_data, dict):
                            title = citation_data.get('title', filename)
                            citation_text = citation_data.get('in_text_citation', filename)
            except Exception as e:
                print(f"⚠️ Citation lookup error for {filename}: {e}")
                # Keep default values

            if filename not in citation_map:
                citation_map[filename] = citation_counter
                citation_counter += 1
                used_citations.append(f"[{citation_map[filename]}] {citation_text}")

            citation_num = citation_map[filename]
            context_parts.append(f"From '{title}' [{citation_num}]:\n{text}")

        full_context = "\n\n---\n\n".join(context_parts)

        # Generate manuscript analysis
        prompt = f"""You are writing a focused manuscript paragraph analyzing XAS (X-ray Absorption Spectroscopy) research.

Based on the following literature from {len(selected_chunks)} key research papers, provide a concise but thorough academic analysis addressing: {research_question}

LITERATURE CONTEXT:
{full_context}

Write a single, well-structured manuscript paragraph that synthesizes the key insights. Focus on:

1. **Key Findings**: Main discoveries and insights from the literature
2. **Technical Insights**: Specific XAS spectral features and interpretations
3. **Comparative Analysis**: How different studies compare and what consensus exists
4. **Implications**: What these findings mean for the field

IMPORTANT CITATION INSTRUCTIONS:
- Use numbered citations [1], [2], [3], etc. in your text when referencing specific findings
- The citation numbers correspond to the papers listed in the context above
- Integrate citations naturally into your analysis text
- Focus on synthesizing insights rather than just summarizing individual studies
- Keep the analysis concise but comprehensive - aim for 1-2 well-developed paragraphs
- Use language and writing style as close to the provided context as possible

MANUSCRIPT ANALYSIS:
"""

        try:
            response = self.model.generate_content(prompt)
            analysis = response.text

            # Add citations at the end
            citations_section = "\n\nREFERENCES:\n" + "\n".join(used_citations)
            return analysis + citations_section

        except Exception as e:
            return f"Error generating analysis: {str(e)}"

    def get_paper_stats(self):
        """Get statistics about loaded papers"""
        if hasattr(self, 'texts') and self.texts:
            unique_papers = len(set(self.filenames))
            total_chunks = len(self.texts)
            return f"📚 {unique_papers} papers processed into {total_chunks} searchable chunks"
        else:
            return "📚 No papers currently loaded"

# Main execution functions
def main(force_reload=False):
    """Initialize and run XAS manuscript analysis system"""
    print("🧪 XAS Manuscript RAG System (FAISS Edition)")
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