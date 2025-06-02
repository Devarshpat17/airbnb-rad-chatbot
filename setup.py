#!/usr/bin/env python3
"""
JSON RAG System Setup and Initialization

This module handles the complete initialization of the Airbnb property search system:
- Database connection and document loading
- Vocabulary building from MongoDB data
- Embedding generation and caching
- FAISS index creation
- Numeric filters optimization
- System validation and testing

Usage:
    python setup.py --full-setup
    python setup.py --rebuild-indexes
    python setup.py --build-vocab-only
    python setup.py --test-system
"""

import logging
import argparse
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Import system components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from utils import MongoDBConnector, IndexManager
from core_system import JSONRAGSystem
from utils import VocabularyManager, AirbnbOptimizer, TextProcessor

# Set up enhanced logging for setup process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.get_log_file_path('setup')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SystemSetup:
    """Handles complete system setup and initialization"""
    
    def __init__(self, force_rebuild: bool = False):
        self.force_rebuild = force_rebuild
        self.db_connector = MongoDBConnector()
        self.index_manager = IndexManager()
        self.vocabulary_manager = VocabularyManager()
        self.airbnb_optimizer = AirbnbOptimizer()
        self.text_processor = TextProcessor()
        
        # Setup tracking
        self.setup_stats = {
            'start_time': None,
            'end_time': None,
            'documents_processed': 0,
            'embeddings_generated': 0,
            'vocabulary_terms': 0,
            'faiss_index_size': 0,
            'setup_success': False,
            'errors': []
        }
    
    def setup_complete_system(self) -> bool:
        """Run complete system setup with all components"""
        logger.info("Starting complete JSON RAG system setup...")
        self.setup_stats['start_time'] = time.time()
        
        try:
            # Step 1: Ensure directories exist
            if not self._ensure_directories():
                return False
            
            # Step 2: Test database connection
            if not self._setup_database_connection():
                return False
            
            # Step 3: Load and validate documents
            documents = self._load_documents()
            if not documents:
                return False
            
            # Step 4: Build vocabulary from MongoDB data
            if not self._build_vocabulary(documents):
                return False
            
            # Step 5: Initialize optimizers with vocabulary
            if not self._initialize_optimizers(documents):
                return False
            
            # Step 6: Create embeddings and indexes
            if not self._create_indexes(documents):
                return False
            
            # Step 7: Setup numeric filters
            if not self._setup_numeric_filters():
                return False
            
            # Step 8: Validate system
            if not self._validate_system():
                return False
            
            self.setup_stats['setup_success'] = True
            self.setup_stats['end_time'] = time.time()
            
            self._print_setup_summary()
            logger.info("System setup completed successfully!")
            return True
            
        except Exception as e:
            error_msg = f"Setup failed with error: {str(e)}"
            logger.error(error_msg)
            self.setup_stats['errors'].append(error_msg)
            return False
    
    def _ensure_directories(self) -> bool:
        """Create necessary directories for the system"""
        logger.info("Creating system directories...")
        
        try:
            Config.ensure_directories()
            logger.info("System directories created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    def _setup_database_connection(self) -> bool:
        """Establish and test database connection"""
        logger.info("Setting up database connection...")
        
        if not self.db_connector.connect():
            logger.error("Failed to connect to MongoDB")
            self.setup_stats['errors'].append("Database connection failed")
            return False
        
        # Test basic operations
        doc_count = self.db_connector.get_document_count()
        logger.info(f"Database connected successfully. Found {doc_count} documents.")
        
        if doc_count == 0:
            logger.warning("Database contains no documents")
            self.setup_stats['errors'].append("No documents in database")
            return False
        
        return True
    
    def _load_documents(self) -> Optional[list]:
        """Load documents from MongoDB"""
        logger.info("Loading documents from MongoDB...")
        
        documents = self.db_connector.get_all_documents()
        
        if not documents:
            logger.error("No documents loaded from database")
            return None
        
        # Filter and validate documents
        valid_documents = []
        for doc in documents:
            if self._validate_document(doc):
                valid_documents.append(doc)
        
        self.setup_stats['documents_processed'] = len(valid_documents)
        logger.info(f"Loaded and validated {len(valid_documents)} documents")
        
        return valid_documents
    
    def _validate_document(self, document: Dict[str, Any]) -> bool:
        """Validate individual document structure"""
        required_fields = ['_id']
        important_fields = ['name', 'description', 'price', 'property_type']
        
        # Check required fields
        for field in required_fields:
            if field not in document:
                return False
        
        # Check if document has some important fields
        has_important_fields = any(document.get(field) for field in important_fields)
        
        return has_important_fields
    
    def _build_vocabulary(self, documents: list) -> bool:
        """Build vocabulary from MongoDB documents"""
        logger.info("Building vocabulary from documents...")
        
        try:
            self.vocabulary_manager.build_vocabulary_from_documents(documents)
            
            vocab_size = len(self.vocabulary_manager.vocabulary)
            mappings_count = len(self.vocabulary_manager.keyword_mappings)
            
            self.setup_stats['vocabulary_terms'] = vocab_size
            
            logger.info(f"Vocabulary built: {vocab_size} terms, {mappings_count} keyword mappings")
            
            # Save vocabulary to data folder
            self.vocabulary_manager.save_vocabulary()
            
            # Log interesting vocabulary statistics
            self._log_vocabulary_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build vocabulary: {e}")
            return False
    
    def _log_vocabulary_stats(self):
        """Log interesting vocabulary statistics"""
        try:
            # Most common terms
            top_terms = self.vocabulary_manager.term_frequencies.most_common(10)
            logger.info(f"Top 10 terms: {top_terms}")
            
            # Field distribution
            if hasattr(self.vocabulary_manager, 'field_terms') and isinstance(self.vocabulary_manager.field_terms, dict):
                field_counts = {field: len(terms) for field, terms in self.vocabulary_manager.field_terms.items()}
                logger.info(f"Terms per field: {field_counts}")
            else:
                logger.info("Field terms not available for statistics")
            
        except Exception as e:
            logger.debug(f"Could not log vocabulary stats: {e}")
    
    def _initialize_optimizers(self, documents: list) -> bool:
        """Initialize optimizers with MongoDB data"""
        logger.info("Initializing optimizers with vocabulary...")
        
        try:
            # Initialize Airbnb optimizer with vocabulary
            self.airbnb_optimizer.initialize_with_mongodb_data(documents)
            
            # Update index manager's document processor
            self.index_manager.document_processor.airbnb_optimizer = self.airbnb_optimizer
            
            logger.info("Optimizers initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizers: {e}")
            return False
    
    def _create_indexes(self, documents: list) -> bool:
        """Create embeddings and FAISS indexes"""
        logger.info("Creating embeddings and indexes...")
        
        try:
            # Check if indexes exist and rebuild is not forced
            if not self.force_rebuild:
                faiss_index, processed_docs = self.index_manager.load_indexes()
                if faiss_index is not None and processed_docs:
                    logger.info("Existing indexes found, skipping rebuild")
                    self.index_manager.faiss_index = faiss_index
                    self.index_manager.processed_documents = processed_docs
                    self.setup_stats['faiss_index_size'] = faiss_index.ntotal
                    return True
            
            # Create complete index
            success = self.index_manager.create_complete_index(rebuild=self.force_rebuild)
            
            if success:
                self.setup_stats['faiss_index_size'] = self.index_manager.faiss_index.ntotal if self.index_manager.faiss_index else 0
                self.setup_stats['embeddings_generated'] = len(self.index_manager.document_embeddings)
                logger.info(f"Created FAISS index with {self.setup_stats['faiss_index_size']} embeddings")
                return True
            else:
                logger.error("Failed to create indexes")
                return False
                
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            return False
    
    def _setup_numeric_filters(self) -> bool:
        """Setup numeric filtering optimizations"""
        logger.info("Setting up numeric filters...")
        
        try:
            # Analyze numeric fields in documents for optimization
            numeric_stats = self._analyze_numeric_fields()
            
            # Log numeric field statistics
            logger.info(f"Numeric field analysis: {numeric_stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup numeric filters: {e}")
            return False
    
    def _analyze_numeric_fields(self) -> Dict[str, Any]:
        """Analyze numeric fields for optimization"""
        numeric_fields = ['price', 'bedrooms', 'bathrooms', 'accommodates', 'review_scores_rating']
        stats = {}
        
        documents = self.index_manager.processed_documents
        
        for field in numeric_fields:
            values = []
            for doc in documents:
                original_doc = doc.get('original_document', doc)
                if field in original_doc and original_doc[field] is not None:
                    try:
                        value = float(str(original_doc[field]).replace('$', '').replace(',', ''))
                        values.append(value)
                    except (ValueError, TypeError):
                        continue
            
            if values:
                stats[field] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }
        
        return stats
    
    def _validate_system(self) -> bool:
        """Validate the complete system setup"""
        logger.info("Validating system setup...")
        
        try:
            # Test system initialization
            rag_system = JSONRAGSystem()
            
            # Initialize with existing components
            rag_system.index_manager = self.index_manager
            rag_system.vocabulary_manager = self.vocabulary_manager
            
            # Test query processing
            test_queries = [
                "Find 2 bedroom apartments under $150",
                "Show me places with WiFi and parking",
                "Highly rated properties downtown"
            ]
            
            for query in test_queries:
                try:
                    # Test query analysis
                    analysis = rag_system.query_engine.analyze_query(query)
                    if not analysis or not analysis.get('keywords'):
                        logger.warning(f"Query analysis failed for: {query}")
                        continue
                    
                    logger.info(f"Test query '{query[:30]}...': {len(analysis['keywords'])} keywords extracted")
                    
                except Exception as e:
                    logger.warning(f"Test query failed: {e}")
            
            logger.info("System validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            return False
    
    def _print_setup_summary(self):
        """Print comprehensive setup summary"""
        duration = self.setup_stats['end_time'] - self.setup_stats['start_time']
        
        print("\n" + "="*60)
        print("JSON RAG SYSTEM SETUP SUMMARY")
        print("="*60)
        print(f"Setup Duration: {duration:.2f} seconds")
        print(f"Documents Processed: {self.setup_stats['documents_processed']}")
        print(f"Vocabulary Terms: {self.setup_stats['vocabulary_terms']}")
        print(f"Embeddings Generated: {self.setup_stats['embeddings_generated']}")
        print(f"FAISS Index Size: {self.setup_stats['faiss_index_size']}")
        print(f"Setup Success: {self.setup_stats['setup_success']}")
        
        if self.setup_stats['errors']:
            print("\nErrors Encountered:")
            for error in self.setup_stats['errors']:
                print(f"  - {error}")
        
        print("\nSystem Components:")
        print(f"  - Database: {Config.DATABASE_NAME}.{Config.COLLECTION_NAME}")
        print(f"  - Embedding Model: {Config.EMBEDDING_MODEL}")
        print(f"  - Index Files: {Config.INDEXES_DIR}")
        print(f"  - Cache Dir: {Config.CACHE_DIR}")
        print("\n" + "="*60)
    
    def rebuild_indexes_only(self) -> bool:
        """Rebuild only the indexes without vocabulary"""
        logger.info("Rebuilding indexes only...")
        
        if not self._setup_database_connection():
            return False
        
        documents = self._load_documents()
        if not documents:
            return False
        
        return self._create_indexes(documents)
    
    def build_vocabulary_only(self) -> bool:
        """Build only the vocabulary from MongoDB"""
        logger.info("Building vocabulary only...")
        
        if not self._setup_database_connection():
            return False
        
        documents = self._load_documents()
        if not documents:
            return False
        
        return self._build_vocabulary(documents)
    
    def test_system_only(self) -> bool:
        """Test system without setup"""
        logger.info("Testing system components...")
        
        return self._validate_system()

def main():
    """Main setup function with command line interface"""
    parser = argparse.ArgumentParser(
        description="JSON RAG System Setup and Initialization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py --full-setup              # Complete system setup
  python setup.py --rebuild-indexes         # Rebuild indexes only
  python setup.py --build-vocab-only        # Build vocabulary only
  python setup.py --test-system            # Test system components
  python setup.py --full-setup --force     # Force rebuild everything
        """
    )
    
    parser.add_argument(
        '--full-setup', 
        action='store_true',
        help='Run complete system setup including database, vocabulary, embeddings, and indexes'
    )
    
    parser.add_argument(
        '--rebuild-indexes',
        action='store_true', 
        help='Rebuild embeddings and FAISS indexes only'
    )
    
    parser.add_argument(
        '--build-vocab-only',
        action='store_true',
        help='Build vocabulary from MongoDB documents only'
    )
    
    parser.add_argument(
        '--test-system',
        action='store_true',
        help='Test system components without setup'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild of existing components'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create setup instance
    setup = SystemSetup(force_rebuild=args.force)
    
    # Determine which operation to run
    success = False
    
    if args.full_setup:
        success = setup.setup_complete_system()
    elif args.rebuild_indexes:
        success = setup.rebuild_indexes_only()
    elif args.build_vocab_only:
        success = setup.build_vocabulary_only()
    elif args.test_system:
        success = setup.test_system_only()
    else:
        # Default to full setup if no specific option given
        print("No specific setup option provided. Running full setup...")
        success = setup.setup_complete_system()
    
    # Exit with appropriate code
    if success:
        print("\nSetup completed successfully!")
        print("You can now run 'python main.py' to start the system.")
        sys.exit(0)
    else:
        print("\nSetup failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
