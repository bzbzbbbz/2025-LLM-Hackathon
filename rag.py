import os
import uuid
import time

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
        self.rebuilt_during_connection = False  # Track if collection was rebuilt
        self.db_path = None  # Will be set dynamically

        print(f"🔄 Initializing XASManuscriptRAG with force_reload={force_reload}")
        print(f"🔍 Checking cache and database status...")
        print(f"   - Cache file exists: {os.path.exists(CACHE_FILE)}")
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
            self.setup_embeddings()  # Always need embeddings for search
            print("🗄️ Connecting to existing Milvus database...")
            self.setup_milvus_connection()
            
            # If collection was rebuilt during connection, save the cache
            if self.rebuilt_during_connection:
                print("💾 Updating cache after collection rebuild...")
                self.save_cache()
            else:
                print("✅ Successfully used existing cached data and database")
        else:
            if force_reload:
                print("📚 Force reload requested - loading papers from scratch...")
            else:
                print("📚 No cache found - loading papers from scratch...")
            print("📖 Starting fresh paper processing pipeline...")
            self.load_metadata()
            self.setup_gemini()
            self.setup_milvus()
            self.setup_embeddings()
            print("📄 About to start load_papers() - this will process chunks...")
            self.load_papers()
            self.save_cache()  # Save cache after processing papers
            print("✅ Fresh processing completed and cache saved")

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

    def cleanup_milvus_processes(self):
        """Clean up any zombie Milvus processes that might be holding locks"""
        try:
            import subprocess
            
            print("🧹 Starting cleanup of Milvus processes and locks...")
            
            # 1. Kill all milvus processes
            try:
                subprocess.run(['pkill', '-9', 'milvus'], capture_output=True, timeout=5)
                print("   - Killed all Milvus processes")
            except:
                pass
                
            # 2. Find and kill specific processes using database files
            try:
                result = subprocess.run(['lsof', '-t', './milvus_demo*.db'], 
                                      capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    print(f"   - Found {len(pids)} processes using database files")
                    for pid in pids:
                        try:
                            subprocess.run(['kill', '-9', pid.strip()], timeout=3)
                        except:
                            pass
                            
            except Exception:
                pass
                
            # 3. Remove all lock files and temp files
            cleanup_patterns = [
                '.milvus_demo*.db.lock',
                '.milvus_demo*.db-wal', 
                '.milvus_demo*.db-shm',
                '.milvus_demo*.db-journal',
                'milvus_demo_*.db'  # Remove any problematic temp databases
            ]
            
            import glob
            removed_files = 0
            for pattern in cleanup_patterns:
                for lock_file in glob.glob(pattern):
                    try:
                        os.remove(lock_file)
                        print(f"   - Removed: {lock_file}")
                        removed_files += 1
                    except:
                        pass
                        
            if removed_files > 0:
                print(f"   - Removed {removed_files} lock/temp files")
                
            # 4. Wait for cleanup to complete
            time.sleep(1)
            print("✅ Cleanup completed")
                     
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")

    def get_working_database_path(self):
        """Get a working database path, handling locks intelligently"""
        original_db = "./milvus_demo.db"
        
        # If original database exists and is accessible, use it
        if os.path.exists(original_db):
            try:
                # Quick test to see if file is accessible
                with open(original_db, 'rb') as f:
                    f.read(1)
                print(f"💾 Using original database: {original_db}")
                return original_db
            except:
                print(f"⚠️ Original database not accessible")
        
        # If original database has issues, try to rename it and start fresh
        if os.path.exists(original_db):
            try:
                backup_name = f"./milvus_demo_backup_{int(time.time())}.db"
                os.rename(original_db, backup_name)
                print(f"📁 Renamed problematic database to: {backup_name}")
            except:
                print("⚠️ Could not rename problematic database")
        
        print(f"🆕 Will create fresh database: {original_db}")
        return original_db

    def setup_milvus(self):
        """Setup Milvus connection and schema"""
        try:
            # Clean up any zombie processes first
            self.cleanup_milvus_processes()
            
            # Clean up any existing connections first
            try:
                connections.disconnect("default")
            except:
                pass
            
            # Get a working database path
            db_path = self.get_working_database_path()
            self.db_path = db_path
            
            print(f"🔗 Connecting to Milvus database: {db_path}")
            
            # Connect to database
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
            raise

    def setup_milvus_connection(self):
        """Setup Milvus connection only (for cached data)"""
        print("🗄️ === STARTING DATABASE CONNECTION SETUP ===")
        try:
            # Clean up any zombie processes first
            print("🧹 Cleaning up zombie processes...")
            self.cleanup_milvus_processes()
            
            # Clean up any existing connections first
            try:
                connections.disconnect("default")
                print("🔌 Disconnected existing connections")
            except:
                print("🔌 No existing connections to disconnect")
                pass
            
            # Try to connect to the original database
            db_path = "./milvus_demo.db"
            print(f"🔗 Attempting to connect to database: {db_path}")
            
            connection_successful = False
            
            # Check if database exists and is accessible
            if os.path.exists(db_path):
                try:
                    # Test file accessibility
                    with open(db_path, 'rb') as f:
                        f.read(1)
                    
                    # Try to connect
                    connections.connect(alias="default", uri=db_path)
                    self.db_path = db_path
                    connection_successful = True
                    print(f"✅ Connected to database successfully")
                    
                except Exception as conn_error:
                    print(f"❌ Connection failed: {conn_error}")
                    if "opened by another program" in str(conn_error) or "illegal connection" in str(conn_error):
                        print("🔒 Database locked or corrupted, will rebuild")
                    connection_successful = False
            else:
                print("❌ Database file not found")
                connection_successful = False
            
            # If connection failed, rebuild everything
            if not connection_successful:
                print("❌ REBUILD TRIGGER: Could not connect to existing database")
                self.setup_milvus()
                self.load_papers()
                self.rebuilt_during_connection = True
                return
            
            # Check if collection exists
            collections = utility.list_collections()
            print(f"📋 Available collections: {collections}")
            
            if COLLECTION_NAME not in collections:
                print(f"❌ REBUILD TRIGGER: Collection '{COLLECTION_NAME}' not found")
                self.setup_milvus()
                self.load_papers()
                self.rebuilt_during_connection = True
                return
                
            print(f"✅ Collection '{COLLECTION_NAME}' found")
            self.collection = Collection(COLLECTION_NAME)
            
            # Check if collection has data
            try:
                num_entities = self.collection.num_entities
                print(f"📊 Collection has {num_entities} entities")
                
                if num_entities == 0:
                    print("❌ REBUILD TRIGGER: Collection is empty")
                    self.setup_milvus()
                    self.load_papers()
                    self.rebuilt_during_connection = True
                    return
                    
            except Exception as e:
                print(f"❌ REBUILD TRIGGER: Error checking collection entities: {e}")
                self.setup_milvus()
                self.load_papers()
                self.rebuilt_during_connection = True
                return
            
            # Check indexes
            try:
                print("🔍 Checking for indexes...")
                indexes = self.collection.indexes
                has_embedding_index = False
                
                print(f"   Found {len(indexes)} indexes:")
                for index in indexes:
                    print(f"   - Field: {index.field_name}, Type: {index.index_type}")
                    if index.field_name == "embedding":
                        has_embedding_index = True
                        print(f"✅ Found index on embedding field: {index.index_type}")
                
                if not has_embedding_index:
                    print("❌ REBUILD TRIGGER: No index found on embedding field")
                    utility.drop_collection(COLLECTION_NAME)
                    self.setup_milvus()
                    self.load_papers()
                    self.rebuilt_during_connection = True
                    return
                else:
                    print("🚀 Loading collection for search...")
                    self.collection.load()
                    print("✅ Collection loaded successfully")
                    
            except Exception as index_error:
                print(f"❌ REBUILD TRIGGER: Error with indexes: {index_error}")
                try:
                    utility.drop_collection(COLLECTION_NAME)
                except:
                    pass
                self.setup_milvus()
                self.load_papers()
                self.rebuilt_during_connection = True
                return
                
            print("✅ Milvus connection established with existing data - NO REBUILD NEEDED")
            print("🗄️ === DATABASE CONNECTION SETUP COMPLETE ===")
            
        except Exception as e:
            print(f"❌ REBUILD TRIGGER: Milvus connection failed with exception: {e}")
            print("🔄 Rebuilding database...")
            try:
                self.setup_milvus()
                self.load_papers()
                self.rebuilt_during_connection = True
            except Exception as setup_error:
                print(f"❌ Even rebuild failed: {setup_error}")
                raise setup_error

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
                'chunk_overlap': CHUNK_OVERLAP,
                'db_path': self.db_path  # Save the database path used
            }

            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)

            print(f"✅ Cache saved to {CACHE_FILE}")
            print(f"   - Database path: {self.db_path}")
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
                
                # Load database path if available, otherwise use default
                if 'db_path' in cache_data and os.path.exists(cache_data['db_path']):
                    self.db_path = cache_data['db_path']
                    print(f"   - Using cached database: {self.db_path}")
                else:
                    print(f"   - Cached database not found, will use default")
                
                print(f"✅ Cache loaded successfully from {CACHE_FILE}")
                print(f"   - Citations loaded: {len(self.citations) if self.citations else 0} papers")
                print(f"   - Collection name: {cache_data.get('collection_name', 'Unknown')}")
                return True
            else:
                print(f"❌ Cache file not found: {CACHE_FILE}")
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