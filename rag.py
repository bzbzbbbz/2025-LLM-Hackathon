import os
import uuid

# Suppress gRPC warnings and errors
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import google.generativeai as genai
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
import pickle
import json

# Configuration
COLLECTION_NAME = "xas_papers"
CONVERTED_FOLDER = "./converted_papers"
CHUNK_SIZE = 1000  # Larger chunks for better context in manuscript analysis
CHUNK_OVERLAP = 200
CACHE_FILE = "./rag_cache.pkl"

class XASManuscriptRAG:
    def __init__(self, force_reload=False):
        self.citations = None
        self.model = None
        self.embed_model = None
        self.collection = None
        self.temperature = 0.1

        # Try to load from cache first
        if not force_reload and self.load_cache():
            print("✅ Loaded from cache - skipping paper processing")
            self.setup_gemini()
            self.setup_embeddings()  # Always need embeddings for search
            self.setup_milvus_connection()
        else:
            print("📚 Loading papers from scratch...")
            self.load_metadata()
            self.setup_gemini()
            self.setup_milvus()
            self.setup_embeddings()
            self.load_papers()
            self.save_cache()

    def load_metadata(self):
        """Load metadata from CSV file with error handling"""
        try:
            metadata_path = "./paper_metadata.csv"
            if os.path.exists(metadata_path):
                self.citations = pd.read_csv(metadata_path).set_index('filename').to_dict('index')
                print(f"✅ Loaded metadata from {metadata_path}")
            else:
                print(f"⚠️ No metadata file found at {metadata_path}, using filenames as citations")
                self.citations = {}
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            self.citations = {}

    def setup_gemini(self):
        """Initialize Gemini Pro"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
                
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found. Set it in environment variables or through the Streamlit interface.")
                
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.temperature = 0.2
            print("✅ Gemini Pro configured")
        except Exception as e:
            print(f"❌ Gemini setup failed: {e}")
            raise

    def setup_milvus(self):
        """Setup Milvus connection and schema"""
        try:
            # Clean up any existing connections first
            try:
                connections.disconnect("default")
            except:
                pass
            
            db_path = "./milvus_demo.db"
            
            # Remove any existing lock files
            lock_files = [f for f in os.listdir('.') if f.startswith('.milvus_demo') and f.endswith('.db.lock')]
            for lock_file in lock_files:
                try:
                    os.remove(lock_file)
                except:
                    pass
            
            connections.connect(alias="default", uri=db_path)

            # Simple schema optimized for manuscript analysis
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # BGE dimension
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=3000),  # Larger for manuscript chunks
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=200)
            ]

            schema = CollectionSchema(fields, description="XAS paper chunks for manuscript analysis")

            # Drop existing collection if exists
            if COLLECTION_NAME in utility.list_collections():
                utility.drop_collection(COLLECTION_NAME)
    
            self.collection = Collection(COLLECTION_NAME, schema)
            print("✅ Milvus collection created")
        except Exception as e:
            print(f"❌ Milvus setup failed: {e}")
            # Try with a unique database name to avoid conflicts
            unique_db = f"./milvus_demo_{uuid.uuid4().hex[:8]}.db"
            try:
                connections.connect(alias="default", uri=unique_db, force=True)
                # Recreate fields for fallback
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=3000),
                    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=200)
                ]
                schema = CollectionSchema(fields, description="XAS paper chunks for manuscript analysis")
                self.collection = Collection(COLLECTION_NAME, schema)
                print("✅ Milvus collection created with unique database")
            except Exception as e2:
                print(f"❌ Fallback Milvus setup also failed: {e2}")
                raise

    def setup_milvus_connection(self):
        """Setup Milvus connection only (for cached data)"""
        try:
            # Clean up any existing connections first
            try:
                connections.disconnect("default")
            except:
                pass
                
            db_path = "./milvus_demo.db"
            connections.connect(alias="default", uri=db_path)
            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
            print("✅ Milvus connection established")
        except Exception as e:
            print(f"❌ Milvus connection failed: {e}")
            # Try to create new database if connection fails
            print("🔄 Creating new database...")
            self.setup_milvus()

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
                'collection_name': COLLECTION_NAME,
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP
            }

            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)

            print(f"✅ Cache saved to {CACHE_FILE}")
            return True
        except Exception as e:
            print(f"❌ Failed to save cache: {e}")
            return False

    def load_cache(self):
        """Load data from cache if available"""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'rb') as f:
                    cache_data = pickle.load(f)

                self.citations = cache_data['citations']
                print(f"✅ Cache loaded from {CACHE_FILE}")
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
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

        # Insert into Milvus
        self.collection.insert([embeddings.tolist(), texts, filenames])
        self.collection.flush()

        # Create index for fast search
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        self.collection.load()

        print(f"Loaded {len(texts)} chunks from {len(set(filenames))} papers")

    def search_papers(self, query, k=15):
        """Search for relevant chunks across all papers"""
        if self.embed_model is None:
            raise ValueError("Embedding model not initialized. Please run setup_embeddings() first.")

        if self.collection is None:
            raise ValueError("Milvus collection not initialized. Please run setup_milvus() first.")

        query_embedding = self.embed_model.encode([query], normalize_embeddings=True)

        results = self.collection.search(
            data=query_embedding,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=k,
            output_fields=["text", "filename"]
        )

        return [(hit.entity.get("text"), hit.entity.get("filename"), float(hit.score))
                for hit in results[0]]

    def generate_manuscript_analysis(self, research_question):
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
            # Sort by relevance and take top chunk only
            paper_chunks.sort(key=lambda x: x[1], reverse=True)
            selected_chunks.extend([(text, filename, score) for text, score in paper_chunks[:2]])

        # Limit to top 5-6 papers for a focused paragraph
        selected_chunks = selected_chunks[:6]

        # Build comprehensive context with proper citations
        context_parts = []
        used_citations = []
        citation_map = {}  # Map filename to citation number

        for i, (text, filename, score) in enumerate(selected_chunks, 1):
            # Get citation info from metadata
            meta = self.citations.get(filename, {})
            title = meta.get('title', filename)
            in_text = meta.get('in_text_citation', filename)

            # Create citation number
            if filename not in citation_map:
                citation_map[filename] = len(used_citations) + 1
                # Use citation column if available, otherwise fall back to title
                citation_text = meta.get('citation', title)
                used_citations.append(f"[{citation_map[filename]}] {citation_text}")

            citation_num = citation_map[filename]
            context_parts.append(f"From '{title}' [{citation_num}]:\n{text}")

        full_context = "\n\n---\n\n".join(context_parts)


        # Manuscript-focused prompt for XAS analysis
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