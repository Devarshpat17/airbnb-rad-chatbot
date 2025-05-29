"""
Index Management for JSON RAG System
Handles document processing, embedding generation, and search index creation.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from config import Config
from database import MongoDBConnector
from utils import TextProcessor
from airbnb_config import AIRBNB_CONFIG, FIELD_CATEGORIES, get_field_weight
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndexManager:
    """
    Manages the creation and maintenance of search indexes for the JSON RAG system.
    Handles document processing, embedding generation, and FAISS index creation.
    """
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.model = None
        self.faiss_index = None
        self.processed_docs = []
        
    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        return self.model
    
    def load_documents_from_mongodb(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load documents from MongoDB collection.
        
        Args:
            limit: Maximum number of documents to load
            
        Returns:
            List of document dictionaries
        """
        logger.info("Loading documents from MongoDB...")
        documents = []
        
        try:
            with MongoDBConnector() as db:
                collection = db.get_collection()
                
                # Create query with limit if specified
                cursor = collection.find({})
                if limit:
                    cursor = cursor.limit(limit)
                
                for doc in cursor:
                    # Convert ObjectId to string
                    if '_id' in doc:
                        doc['_id'] = str(doc['_id'])
                    documents.append(doc)
                    
                logger.info(f"Loaded {len(documents)} documents from MongoDB")
                
        except Exception as e:
            logger.error(f"Error loading documents from MongoDB: {e}")
            raise
            
        return documents
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw MongoDB documents into searchable format.
        
        Args:
            documents: List of raw document dictionaries
            
        Returns:
            List of processed documents with searchable text
        """
        logger.info(f"Processing {len(documents)} documents...")
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                processed_doc = self._process_single_document(doc)
                processed_docs.append(processed_doc)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1} documents")
                    
            except Exception as e:
                logger.warning(f"Error processing document {doc.get('_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_docs)} documents")
        return processed_docs
    
    def _process_single_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single document into searchable format.
        
        Args:
            doc: Raw document dictionary
            
        Returns:
            Processed document with searchable text and extracted fields
        """
        # Create searchable text by combining weighted fields
        searchable_parts = []
        extracted_fields = {}
        field_completion_count = 0
        total_fields = 0
        
        # Process each field according to its weight and category
        for field_name, field_value in doc.items():
            if field_value is None or field_value == '':
                continue
                
            total_fields += 1
            field_completion_count += 1
            
            # Get field weight from configuration
            weight = get_field_weight(field_name)
            
            # Convert field value to string
            field_str = self._convert_field_to_string(field_value)
            
            if field_str and weight > 0:
                # Clean and process the text
                cleaned_text = self.text_processor.clean_text(field_str)
                
                # Add to searchable text with weight consideration
                if weight >= 0.5:  # High importance fields
                    searchable_parts.append(cleaned_text)
                    if weight >= 0.8:  # Very high importance - add twice
                        searchable_parts.append(cleaned_text)
                
                # Store important fields for quick access
                if field_name in ['name', 'price', 'property_type', 'room_type', 
                                'accommodates', 'bedrooms', 'bathrooms', 'amenities',
                                'neighbourhood_cleansed', 'review_scores_rating']:
                    extracted_fields[field_name] = field_value
        
        # Calculate field completion rate
        field_completion_rate = field_completion_count / max(total_fields, 1) if total_fields > 0 else 0
        
        # Create final searchable text
        searchable_text = ' '.join(searchable_parts)
        
        return {
            'document_id': str(doc.get('_id', '')),
            'searchable_text': searchable_text,
            'extracted_fields': extracted_fields,
            'field_completion_rate': field_completion_rate,
            'original_doc': doc
        }
    
    def _convert_field_to_string(self, field_value: Any) -> str:
        """
        Convert any field value to a string representation.
        
        Args:
            field_value: The field value to convert
            
        Returns:
            String representation of the field value
        """
        if field_value is None:
            return ''
        
        if isinstance(field_value, (str, int, float)):
            return str(field_value)
        
        if isinstance(field_value, list):
            # Handle arrays (like amenities)
            return ' '.join([str(item) for item in field_value if item is not None])
        
        if isinstance(field_value, dict):
            # Handle nested objects
            return ' '.join([str(v) for v in field_value.values() if v is not None])
        
        # For any other type, try to convert to string
        try:
            return str(field_value)
        except:
            return ''
    
    def create_embeddings(self, processed_docs: List[Dict[str, Any]], 
                         use_cache: bool = True) -> np.ndarray:
        """
        Create embeddings for processed documents.
        
        Args:
            processed_docs: List of processed documents
            use_cache: Whether to use cached embeddings if available
            
        Returns:
            Numpy array of embeddings
        """
        cache_file = Config.EMBEDDINGS_CACHE_PATH
        
        # Try to load cached embeddings
        if use_cache and os.path.exists(cache_file):
            try:
                logger.info("Loading cached embeddings...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Verify cache matches current documents
                if (len(cached_data.get('embeddings', [])) == len(processed_docs) and
                    cached_data.get('model') == Config.EMBEDDING_MODEL):
                    
                    logger.info("Using cached embeddings")
                    return np.array(cached_data['embeddings'])
                else:
                    logger.info("Cache mismatch, generating new embeddings")
            except Exception as e:
                logger.warning(f"Error loading cached embeddings: {e}")
        
        # Generate new embeddings
        logger.info(f"Generating embeddings for {len(processed_docs)} documents...")
        model = self._load_model()
        
        # Extract texts for embedding
        texts = [doc['searchable_text'] for doc in processed_docs]
        
        # Process in batches to handle memory efficiently
        batch_size = AIRBNB_CONFIG.get('embedding_batch_size', 32)
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
            all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        embeddings_array = np.array(all_embeddings)
        
        # Cache the embeddings
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            cache_data = {
                'embeddings': embeddings_array.tolist(),
                'model': Config.EMBEDDING_MODEL,
                'doc_count': len(processed_docs)
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cached embeddings to {cache_file}")
        except Exception as e:
            logger.warning(f"Error caching embeddings: {e}")
        
        return embeddings_array
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index for similarity search
        """
        logger.info(f"Creating FAISS index with {len(embeddings)} embeddings...")
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create FAISS index
        dimension = normalized_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add embeddings to index
        index.add(normalized_embeddings.astype(np.float32))
        
        logger.info(f"Created FAISS index with dimension {dimension}")
        return index
    
    def save_indexes(self, faiss_index: faiss.Index, processed_docs: List[Dict[str, Any]]):
        """
        Save FAISS index and processed documents to disk.
        
        Args:
            faiss_index: FAISS index to save
            processed_docs: Processed documents to save
        """
        logger.info("Saving indexes to disk...")
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(Config.FAISS_INDEX_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(Config.EMBEDDINGS_CACHE_PATH), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(faiss_index, Config.FAISS_INDEX_PATH)
        logger.info(f"Saved FAISS index to {Config.FAISS_INDEX_PATH}")
        
        # Save processed documents (using embeddings cache path as base for processed docs)
        processed_docs_path = Config.EMBEDDINGS_CACHE_PATH.replace('embeddings_cache.pkl', 'processed_docs.pkl')
        with open(processed_docs_path, 'wb') as f:
            pickle.dump(processed_docs, f)
        logger.info(f"Saved processed documents to {processed_docs_path}")
    
    def load_indexes(self) -> tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Load FAISS index and processed documents from disk.
        
        Returns:
            Tuple of (FAISS index, processed documents)
        """
        logger.info("Loading indexes from disk...")
        
        # Load FAISS index
        if not os.path.exists(Config.FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {Config.FAISS_INDEX_PATH}")
        
        faiss_index = faiss.read_index(Config.FAISS_INDEX_PATH)
        logger.info(f"Loaded FAISS index from {Config.FAISS_INDEX_PATH}")
        
        # Load processed documents
        processed_docs_path = Config.EMBEDDINGS_CACHE_PATH.replace('embeddings_cache.pkl', 'processed_docs.pkl')
        if not os.path.exists(processed_docs_path):
            raise FileNotFoundError(f"Processed documents not found at {processed_docs_path}")
        
        with open(processed_docs_path, 'rb') as f:
            processed_docs = pickle.load(f)
        logger.info(f"Loaded {len(processed_docs)} processed documents")
        
        return faiss_index, processed_docs
    
    def create_complete_index(self, limit: Optional[int] = None, use_cache: bool = True):
        """
        Main method to create the complete search index.
        
        Args:
            limit: Maximum number of documents to process
            use_cache: Whether to use cached embeddings
        """
        logger.info("Starting complete index creation...")
        
        try:
            # Step 1: Load documents from MongoDB
            documents = self.load_documents_from_mongodb(limit=limit)
            
            if not documents:
                raise ValueError("No documents loaded from MongoDB")
            
            # Step 2: Process documents
            processed_docs = self.process_documents(documents)
            
            if not processed_docs:
                raise ValueError("No documents successfully processed")
            
            # Step 3: Create embeddings
            embeddings = self.create_embeddings(processed_docs, use_cache=use_cache)
            
            # Step 4: Create FAISS index
            faiss_index = self.create_faiss_index(embeddings)
            
            # Step 5: Save indexes
            self.save_indexes(faiss_index, processed_docs)
            
            logger.info("Complete index creation finished successfully!")
            
            # Store for immediate use
            self.faiss_index = faiss_index
            self.processed_docs = processed_docs
            
        except Exception as e:
            logger.error(f"Error creating complete index: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current indexes.
        
        Returns:
            Dictionary with index statistics
        """
        processed_docs_path = Config.EMBEDDINGS_CACHE_PATH.replace('embeddings_cache.pkl', 'processed_docs.pkl')
        stats = {
            'faiss_index_exists': os.path.exists(Config.FAISS_INDEX_PATH),
            'processed_docs_exists': os.path.exists(processed_docs_path),
            'embeddings_cache_exists': os.path.exists(Config.EMBEDDINGS_CACHE_PATH)
        }
        
        # Get FAISS index stats
        if stats['faiss_index_exists']:
            try:
                index = faiss.read_index(Config.FAISS_INDEX_PATH)
                stats['faiss_total_vectors'] = index.ntotal
                stats['faiss_dimension'] = index.d
            except Exception as e:
                stats['faiss_error'] = str(e)
        
        # Get processed docs stats
        if stats['processed_docs_exists']:
            try:
                with open(processed_docs_path, 'rb') as f:
                    docs = pickle.load(f)
                stats['processed_docs_count'] = len(docs)
                if docs:
                    stats['avg_completion_rate'] = sum(d.get('field_completion_rate', 0) for d in docs) / len(docs)
            except Exception as e:
                stats['processed_docs_error'] = str(e)
        
        return stats
    
    def rebuild_index(self, limit: Optional[int] = None):
        """
        Rebuild the index from scratch (ignoring cache).
        
        Args:
            limit: Maximum number of documents to process
        """
        logger.info("Rebuilding index from scratch...")
        
        # Remove cached files
        processed_docs_path = Config.EMBEDDINGS_CACHE_PATH.replace('embeddings_cache.pkl', 'processed_docs.pkl')
        for path in [Config.FAISS_INDEX_PATH, processed_docs_path, Config.EMBEDDINGS_CACHE_PATH]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed {path}")
        
        # Create new index
        self.create_complete_index(limit=limit, use_cache=False)
