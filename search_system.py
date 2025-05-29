"""Consolidated search system combining index management and Airbnb setup functionality."""

import os
import sys
import json
import pickle
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pymongo import MongoClient
from bson import Decimal128

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from database import MongoDBConnector, SessionManager
from core_system import JSONRAGSystem
from utils import TextProcessor

logger = logging.getLogger(__name__)

# Import Airbnb configuration from separate file
from airbnb_config import FIELD_CATEGORIES, AIRBNB_CONFIG, AMENITY_MAPPING



# Default values for missing fields
DEFAULT_VALUES = {
    'price': 0,
    'bedrooms': 1,
    'bathrooms': 1.0,
    'accommodates': 2,
    'review_scores_rating': 0,
    'number_of_reviews': 0
}

class SearchSystem:
    """Consolidated search system with index management and Airbnb setup."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the search system."""
        self.config = config or Config()
        self.model = None
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        self.text_processor = TextProcessor()
        
        # TF-IDF components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Caching
        self.processed_docs_file = os.path.join(self.config.DATA_DIR, 'processed_documents.pkl')
        self.embeddings_file = os.path.join(self.config.DATA_DIR, 'embeddings.pkl')
        self.faiss_index_file = os.path.join(self.config.DATA_DIR, 'faiss_index.bin')
        
        logger.info("Search system initialized")
    
    def setup_logging(self):
        """Setup logging for the setup process."""
        # Create logs directory if it doesn't exist
        os.makedirs(self.config.LOGS_DIR, exist_ok=True)
        
        # Setup file logging
        log_filename = os.path.join(self.config.LOGS_DIR, f'setup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info(f"Setup logging initialized. Log file: {log_filename}")
        return logger
    
    def test_mongodb_connection(self):
        """Test MongoDB connection and verify data exists."""
        try:
            logger.info("Testing MongoDB connection...")
            with MongoDBConnector() as db:
                collection = db.collection
                
                # Count documents
                doc_count = collection.count_documents({})
                logger.info(f"Found {doc_count} documents in collection '{self.config.MONGODB_COLLECTION}'")
                
                if doc_count == 0:
                    logger.warning("No documents found in the collection!")
                    logger.info("Please ensure your MongoDB contains Airbnb listing data.")
                    return False
                
                # Sample a document to check structure
                sample_doc = collection.find_one({})
                if sample_doc:
                    keys = list(sample_doc.keys())[:10]
                    logger.info(f"Sample document keys: {keys}...")
                else:
                    logger.warning("Could not retrieve sample document")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False
    
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            logger.info("Model loaded successfully")
    
    def process_airbnb_document(self, document: Dict[str, Any]) -> str:
        """Process a single Airbnb document and extract relevant text."""
        text_parts = []
        
        # Process main fields with weights
        for field, weight in AIRBNB_CONFIG['field_weights'].items():
            if field in document and document[field]:
                value = str(document[field])
                if value and value.lower() not in ['nan', 'null', 'none']:
                    # Repeat high-weight fields for emphasis
                    repeat_count = max(1, int(weight))
                    for _ in range(repeat_count):
                        text_parts.append(self.text_processor.clean_text(value))
        
        # Add specific field processing
        
        # Process amenities list
        if 'amenities' in document and document['amenities']:
            amenities_text = str(document['amenities'])
            # Clean up amenities format
            amenities_text = amenities_text.replace('{', '').replace('}', '').replace('"', '')
            if amenities_text:
                text_parts.append(f"Amenities: {amenities_text}")
        
        # Process location information
        location_parts = []
        for field in FIELD_CATEGORIES['location']:
            if field in document and document[field]:
                location_parts.append(str(document[field]))
        
        if location_parts:
            text_parts.append(f"Location: {' '.join(location_parts)}")
        
        # Process property specifics
        property_info = []
        for field in FIELD_CATEGORIES['accommodation']:
            if field in document and document[field] is not None:
                property_info.append(f"{field}: {document[field]}")
        
        if property_info:
            text_parts.append(f"Property: {' '.join(property_info)}")
        
        # Combine all text
        combined_text = ' '.join(text_parts)
        
        # Final cleaning
        combined_text = self.text_processor.clean_text(combined_text)
        
        return combined_text if combined_text else "No description available"
    
    def load_documents(self, limit: Optional[int] = None, force_reload: bool = False):
        """Load and process documents from MongoDB."""
        # Check if processed documents cache exists
        if os.path.exists(self.processed_docs_file) and not force_reload:
            logger.info("Loading cached processed documents...")
            with open(self.processed_docs_file, 'rb') as f:
                self.documents = pickle.load(f)
            logger.info(f"Loaded {len(self.documents)} cached documents")
            return
        
        logger.info("Loading documents from MongoDB...")
        
        with MongoDBConnector() as db:
            collection = db.collection
            
            # Get documents with limit if specified
            cursor = collection.find({}).limit(limit) if limit else collection.find({})
            
            raw_documents = list(cursor)
            logger.info(f"Retrieved {len(raw_documents)} documents from database")
            
            # Process documents
            self.documents = []
            for i, doc in enumerate(raw_documents):
                try:
                    processed_text = self.process_airbnb_document(doc)
                    
                    processed_doc = {
                        'id': str(doc['_id']),
                        'text': processed_text,
                        'metadata': {
                            'name': doc.get('name', 'Untitled'),
                            'property_type': doc.get('property_type', 'Unknown'),
                            'room_type': doc.get('room_type', 'Unknown'),
                            'price': self._safe_numeric_convert(doc.get('price'), 'price'),
                            'bedrooms': self._safe_numeric_convert(doc.get('bedrooms'), 'bedrooms'),
                            'accommodates': self._safe_numeric_convert(doc.get('accommodates'), 'accommodates'),
                            'rating': self._safe_numeric_convert(doc.get('review_scores_rating'), 'rating'),
                            'neighbourhood': doc.get('neighbourhood_cleansed', ''),
                            'listing_url': doc.get('listing_url', '')
                        }
                    }
                    
                    self.documents.append(processed_doc)
                    
                    if (i + 1) % 1000 == 0:
                        logger.info(f"Processed {i + 1} documents...")
                        
                except Exception as e:
                    logger.error(f"Error processing document {i}: {e}")
                    continue
        
        logger.info(f"Processed {len(self.documents)} documents")
        
        # Cache processed documents
        os.makedirs(os.path.dirname(self.processed_docs_file), exist_ok=True)
        with open(self.processed_docs_file, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Cached processed documents to {self.processed_docs_file}")
    
    def _safe_numeric_convert(self, value: Any, field_type: str) -> Any:
        """Safely convert values to appropriate numeric types."""
        if value is None:
            return DEFAULT_VALUES.get(field_type, 0)
        
        try:
            # Handle MongoDB Decimal128
            if isinstance(value, Decimal128):
                value = float(str(value))
            
            # Handle string conversions
            if isinstance(value, str):
                # Remove currency symbols and commas
                value = value.replace('$', '').replace(',', '').strip()
                
                if not value or value.lower() in ['nan', 'null', 'none', '']:
                    return DEFAULT_VALUES.get(field_type, 0)
            
            # Convert based on field type
            if field_type in ['bedrooms', 'accommodates', 'reviews']:
                return int(float(value))
            else:
                return float(value)
                
        except (ValueError, TypeError):
            return DEFAULT_VALUES.get(field_type, 0)
    
    def generate_embeddings(self, force_rebuild: bool = False):
        """Generate embeddings for all documents."""
        # Check if embeddings cache exists
        if os.path.exists(self.embeddings_file) and not force_rebuild:
            logger.info("Loading cached embeddings...")
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            logger.info(f"Loaded embeddings for {len(self.embeddings)} documents")
            return
        
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        self.load_model()
        
        logger.info(f"Generating embeddings for {len(self.documents)} documents...")
        
        # Extract texts
        texts = [doc['text'] for doc in self.documents]
        
        # Generate embeddings in batches
        batch_size = AIRBNB_CONFIG['embedding_batch_size']
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated embeddings for {min(i + batch_size, len(texts))}/{len(texts)} documents")
        
        self.embeddings = np.array(all_embeddings)
        logger.info(f"Generated embeddings shape: {self.embeddings.shape}")
        
        # Cache embeddings
        os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        logger.info(f"Cached embeddings to {self.embeddings_file}")
    
    def build_faiss_index(self, force_rebuild: bool = False):
        """Build FAISS index for vector similarity search."""
        # Check if index cache exists
        if os.path.exists(self.faiss_index_file) and not force_rebuild:
            logger.info("Loading cached FAISS index...")
            self.faiss_index = faiss.read_index(self.faiss_index_file)
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            return
        
        if self.embeddings is None:
            raise ValueError("No embeddings available. Call generate_embeddings() first.")
        
        logger.info("Building FAISS index...")
        
        # Create index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to index
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
        
        # Cache index
        os.makedirs(os.path.dirname(self.faiss_index_file), exist_ok=True)
        faiss.write_index(self.faiss_index, self.faiss_index_file)
        logger.info(f"Cached FAISS index to {self.faiss_index_file}")
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            self.config.DATA_DIR,
            self.config.LOGS_DIR,
            os.path.join(self.config.DATA_DIR, 'cache'),
            os.path.join(self.config.DATA_DIR, 'exports')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created/verified directory: {directory}")
    
    def build_search_index(self, force_rebuild: bool = False):
        """Build complete search index."""
        logger.info("Starting index creation process...")
        
        if force_rebuild:
            logger.info("Force rebuild requested - clearing caches")
            cache_files = [self.processed_docs_file, self.embeddings_file, self.faiss_index_file]
            for cache_file in cache_files:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    logger.info(f"Removed cache file: {cache_file}")
        
        # Load and process documents
        self.load_documents(force_reload=force_rebuild)
        
        # Generate embeddings
        self.generate_embeddings(force_rebuild=force_rebuild)
        
        # Build FAISS index
        self.build_faiss_index(force_rebuild=force_rebuild)
        
        logger.info("Search index creation completed")
        
        return {
            'documents_count': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0
        }
    
    def initialize_system(self, force_rebuild: bool = False):
        """Initialize the complete search system."""
        logger.info("Initializing Search System...")
        result = self.build_search_index(force_rebuild=force_rebuild)
        logger.info("Search index built successfully")
        return result
    
    def test_rag_system(self):
        """Test the RAG system with sample queries."""
        try:
            logger.info("Testing RAG system...")
            
            # Initialize system
            rag_system = JSONRAGSystem()
            success = rag_system.initialize_system()
            
            if not success:
                logger.error("Failed to initialize RAG system")
                return False
            
            logger.info("RAG system initialized successfully")
            
            # Test a simple query
            test_queries = [
                "Find apartments with wifi",
                "Show me properties under $100",
                "What are some good places to stay?"
            ]
            
            for query in test_queries:
                try:
                    logger.info(f"Testing query: '{query}'")
                    response, session_id, history = rag_system.process_query(query)
                    
                    # Check if we got a response
                    if response and len(response.strip()) > 0:
                        logger.info(f"[OK] Query successful - got {len(response)} character response")
                    else:
                        logger.warning(f"[WARN] Query returned empty response")
                        
                except Exception as e:
                    logger.error(f"[FAIL] Query failed: {e}")
            
            logger.info("RAG system testing completed")
            return True
            
        except Exception as e:
            logger.error(f"RAG system test failed: {e}")
            return False
    
    def run_setup(self, force_rebuild: bool = False, skip_test: bool = False, skip_export: bool = False):
        """Run the complete setup process."""
        self.setup_logging()
        logger.info("Starting Airbnb RAG System setup...")
        logger.info(f"Arguments: force_rebuild={force_rebuild}, skip_test={skip_test}, skip_export={skip_export}")
        
        steps = [
            ("Creating directories", self.create_directories),
            ("Testing MongoDB connection", self.test_mongodb_connection),
            ("Building search index", lambda: self.build_search_index(force_rebuild=force_rebuild)),
        ]
        
        if not skip_test:
            steps.append(("Testing RAG system", self.test_rag_system))
        
        failed_steps = []
        
        for step_name, step_func in steps:
            logger.info(f"\nExecuting step: {step_name}")
            
            try:
                if step_name == "Building search index":
                    logger.info("Initializing Index Manager...")
                    logger.info("Building search index (this may take a while)...")
                    
                    result = step_func()
                    
                    logger.info(f"Successfully built index with {result['documents_count']} documents")
                    logger.info(f"Embedding dimension: {result['embedding_dimension']}")
                    logger.info(f"FAISS index size: {result['faiss_index_size']}")
                    
                else:
                    result = step_func()
                
                if result is False:
                    failed_steps.append(step_name)
                    logger.error(f"[FAIL] {step_name} failed")
                else:
                    logger.info(f"[OK] {step_name} completed successfully")
                    
            except Exception as e:
                failed_steps.append(step_name)
                logger.error(f"[FAIL] {step_name} failed with error: {e}")
        
        # Print final summary
        self._print_setup_summary(failed_steps)
        
        if failed_steps:
            logger.error(f"\nSetup completed with {len(failed_steps)} failed step(s): {', '.join(failed_steps)}")
            return False
        else:
            logger.info("\nSetup completed successfully!")
            return True
    
    def _print_setup_summary(self, failed_steps: List[str]):
        """Print setup completion summary."""
        logger.info("\n" + "=" * 60)
        logger.info("AIRBNB RAG SYSTEM SETUP COMPLETE")
        logger.info("=" * 60)
        
        # Configuration info
        logger.info("Configuration:")
        logger.info(f"  MongoDB URI: {self.config.MONGODB_URI}")
        logger.info(f"  Database: {self.config.MONGODB_DATABASE}")
        logger.info(f"  Collection: {self.config.MONGODB_COLLECTION}")
        logger.info(f"  Data Directory: {self.config.DATA_DIR}")
        logger.info(f"  Embedding Model: {self.config.EMBEDDING_MODEL}")
        
        # Files created
        logger.info("\nFiles Created:")
        files_to_check = [
            self.faiss_index_file,
            self.processed_docs_file,
            self.embeddings_file
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"  [OK] {file_path} ({size_mb:.1f} MB)")
            else:
                logger.info(f"  [MISSING] {file_path}")
        
        # Next steps
        if not failed_steps:
            logger.info("\nNext Steps:")
            logger.info("  1. Run 'python main.py' to start the web interface")
            logger.info("  2. Open your browser to http://localhost:7861")
            logger.info("  3. Try some sample queries like:")
            logger.info("     - 'Find 2-bedroom apartments with WiFi'")
            logger.info("     - 'Show me places under $100'")
            logger.info("     - 'What are some highly rated properties?'")
        
        logger.info("\n" + "=" * 60)

def main():
    """Main entry point for setup script."""
    parser = argparse.ArgumentParser(description='Setup Airbnb RAG System')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild of all indexes and caches')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip RAG system testing')
    parser.add_argument('--skip-export', action='store_true',
                       help='Skip documentation export')
    
    args = parser.parse_args()
    
    search_system = SearchSystem()
    success = search_system.run_setup(
        force_rebuild=args.force_rebuild,
        skip_test=args.skip_test,
        skip_export=args.skip_export
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
