#!/usr/bin/env python3
"""
JSON RAG System - Utilities Module
Consolidated utilities including database, indexing, and processing components
"""

import logging
import time
import pickle
import numpy as np
import json
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from contextlib import contextmanager
from collections import Counter, defaultdict

# Third-party imports with fallbacks
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
except ImportError:
    MongoClient = None
    ConnectionFailure = Exception
    ServerSelectionTimeoutError = Exception

try:
    import faiss
except ImportError:
    faiss = None
    
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import spacy
except ImportError:
    spacy = None

from config.config import Config

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    user_query: str
    system_response: str
    timestamp: datetime
    entities_extracted: List[str] = field(default_factory=list)
    intent_classified: str = ''
    search_query_used: str = ''
    documents_retrieved: List[str] = field(default_factory=list)
    response_time: float = 0.0
    numeric_constraints: Dict[str, Any] = field(default_factory=dict)
    search_strategy: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionContext:
    """Maintains context for a user session"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    current_topic: str = ''
    accumulated_entities: List[str] = field(default_factory=list)
    search_constraints: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, timeout_seconds: int = Config.SESSION_TIMEOUT) -> bool:
        """Check if session has expired"""
        return (datetime.now() - self.last_activity).total_seconds() > timeout_seconds
    
    def add_conversation_turn(self, turn: ConversationTurn):
        """Add a new conversation turn to history"""
        self.conversation_history.append(turn)
        self.last_activity = datetime.now()
        
        # Limit conversation history size
        if len(self.conversation_history) > Config.MAX_CONVERSATION_HISTORY:
            self.conversation_history = self.conversation_history[-Config.MAX_CONVERSATION_HISTORY:]
    
    def get_recent_context(self, turns: int = 3) -> List[ConversationTurn]:
        """Get recent conversation turns for context"""
        return self.conversation_history[-turns:] if self.conversation_history else []
    
    def update_constraints(self, new_constraints: Dict[str, Any]):
        """Update search constraints with new information"""
        self.search_constraints.update(new_constraints)
    
    def clear_constraints(self):
        """Clear all search constraints"""
        self.search_constraints.clear()

class IntentClassifier:
    """Simple intent classification for user queries"""
    
    INTENT_PATTERNS = {
        'GREETING': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
        'SEARCH': ['find', 'search', 'look for', 'show me', 'get me', 'list', 'want'],
        'FOLLOW_UP': ['what about', 'how about', 'also', 'and', 'additionally', 'plus'],
        'PRICE': ['price', 'cost', 'expensive', 'cheap', 'budget', 'affordable', 'under', 'over'],
        'LOCATION': ['where', 'location', 'area', 'neighborhood', 'near', 'close to'],
        'AMENITY': ['amenity', 'feature', 'has', 'with', 'include', 'offer'],
        'COMPARISON': ['compare', 'versus', 'vs', 'difference', 'better', 'best'],
        'INFO': ['tell me', 'info', 'information', 'about', 'details', 'describe']
    }
    
    @classmethod
    def classify_intent(cls, query: str) -> str:
        """Classify the intent of a user query"""
        query_lower = query.lower()
        
        # Count matches for each intent
        intent_scores = {}
        for intent, patterns in cls.INTENT_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return the intent with highest score, default to SEARCH
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        return 'SEARCH'

class SessionManager:
    """Manages user sessions and conversation context"""
    
    def __init__(self):
        self.sessions: Dict[str, SessionContext] = {}
        self.intent_classifier = IntentClassifier()
    
    def create_session(self, session_id: str) -> SessionContext:
        """Create a new user session"""
        session = SessionContext(
            session_id=session_id,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        self.sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> SessionContext:
        """Get existing session or create new one"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if not session.is_expired():
                return session
            else:
                # Session expired, remove it
                del self.sessions[session_id]
                logger.info(f"Session expired and removed: {session_id}")
        
        # Create new session
        return self.create_session(session_id)
    
    def add_conversation_turn(self, session_id: str, user_query: str, 
                            system_response: str, **metadata):
        """Add a conversation turn to session"""
        session = self.get_session(session_id)
        
        # Classify intent
        intent = self.intent_classifier.classify_intent(user_query)
        
        turn = ConversationTurn(
            user_query=user_query,
            system_response=system_response,
            timestamp=datetime.now(),
            intent_classified=intent,
            **metadata
        )
        
        session.add_conversation_turn(turn)
    
    def get_conversation_context(self, session_id: str, turns: int = 3) -> str:
        """Get formatted conversation context for query enhancement"""
        session = self.get_session(session_id)
        recent_turns = session.get_recent_context(turns)
        
        if not recent_turns:
            return ""
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_query}")
            if turn.entities_extracted:
                entities_str = ", ".join(turn.entities_extracted)
                context_parts.append(f"Entities: {entities_str}")
        
        return " | ".join(context_parts)
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired_sessions = [
            sid for sid, session in self.sessions.items() 
            if session.is_expired()
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions"""
        return {
            'active_sessions': len(self.sessions),
            'total_conversations': sum(len(s.conversation_history) for s in self.sessions.values()),
            'average_turns_per_session': sum(len(s.conversation_history) for s in self.sessions.values()) / max(len(self.sessions), 1)
        }

# ============================================================================
# DATABASE CONNECTIVITY
# ============================================================================

class MongoDBConnector:
    """Handles MongoDB connection and operations"""
    
    def __init__(self):
        self.client = None
        self.database = None
        self.collection = None
        self.connected = False
    
    def connect(self) -> bool:
        """Establish connection to MongoDB"""
        if MongoClient is None:
            logger.error("PyMongo not available. Install with: pip install pymongo")
            return False
        
        try:
            self.client = MongoClient(
                Config.MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test the connection
            self.client.admin.command('ismaster')
            
            self.database = self.client[Config.DATABASE_NAME]
            self.collection = self.database[Config.COLLECTION_NAME]
            self.connected = True
            
            logger.info(f"Successfully connected to MongoDB: {Config.DATABASE_NAME}.{Config.COLLECTION_NAME}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from MongoDB")
    
    @contextmanager
    def get_connection(self):
        """Context manager for safe database operations"""
        if not self.connected:
            if not self.connect():
                yield None
                return
        
        try:
            yield self.collection
        except Exception as e:
            logger.error(f"Database operation error: {e}")
            yield None
    
    def get_document_count(self) -> int:
        """Get total number of documents in collection"""
        with self.get_connection() as collection:
            if collection is not None:
                try:
                    return collection.count_documents({})
                except Exception as e:
                    logger.error(f"Error counting documents: {e}")
            return 0
    
    def get_sample_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sample documents from collection"""
        with self.get_connection() as collection:
            if collection is not None:
                try:
                    return list(collection.find().limit(limit))
                except Exception as e:
                    logger.error(f"Error fetching sample documents: {e}")
            return []
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from collection"""
        with self.get_connection() as collection:
            if collection is not None:
                try:
                    return list(collection.find())
                except Exception as e:
                    logger.error(f"Error fetching all documents: {e}")
            return []

# ============================================================================
# VOCABULARY MANAGEMENT
# ============================================================================

class VocabularyManager:
    """Manages vocabulary extraction and keyword mappings"""
    
    def __init__(self):
        self.vocabulary = Counter()
        self.keyword_mappings = {}
        self.field_terms = defaultdict(set)
        self.term_frequencies = Counter()
        self.doc_frequencies = Counter()
        self.total_documents = 0
    
    def build_vocabulary_from_documents(self, documents: List[Dict[str, Any]]):
        """Build vocabulary from a list of documents"""
        logger.info("Building vocabulary from documents...")
        
        self.total_documents = len(documents)
        text_processor = TextProcessor()
        
        for doc in documents:
            doc_terms = set()
            
            # Extract text from each field
            for field_name, field_value in doc.items():
                if field_name.startswith('_'):
                    continue
                
                field_text = text_processor.extract_text_from_field(field_value)
                if field_text:
                    terms = text_processor.extract_terms(field_text)
                    
                    # Add to field-specific vocabulary
                    self.field_terms[field_name].update(terms)
                    
                    # Add to global vocabulary
                    for term in terms:
                        self.vocabulary[term] += 1
                        self.term_frequencies[term] += 1
                        doc_terms.add(term)
            
            # Update document frequencies
            for term in doc_terms:
                self.doc_frequencies[term] += 1
        
        # Create keyword mappings
        self._create_keyword_mappings()
        
        logger.info(f"Built vocabulary with {len(self.vocabulary)} unique terms")
        logger.info(f"Created {len(self.keyword_mappings)} keyword mappings")
    
    def _create_keyword_mappings(self):
        """Create keyword to field mappings"""
        for field_name, field_terms in self.field_terms.items():
            for term in field_terms:
                if term not in self.keyword_mappings:
                    self.keyword_mappings[term] = []
                self.keyword_mappings[term].append(field_name)
    
    def save_vocabulary(self):
        """Save vocabulary to files without Unicode issues"""
        try:
            # Clean vocabulary to remove non-ASCII terms
            clean_vocabulary = {}
            for term, count in self.vocabulary.items():
                # Only save ASCII-safe terms
                if all(ord(c) < 128 for c in str(term)):
                    clean_vocabulary[term] = count
            
            # Save vocabulary
            vocab_path = Config.DATA_DIR / 'vocabulary.json'
            with open(vocab_path, 'w', encoding='ascii', errors='replace') as f:
                json.dump(clean_vocabulary, f, ensure_ascii=True, indent=2)
            
            # Clean keyword mappings
            clean_mappings = {}
            for term, mappings in self.keyword_mappings.items():
                if all(ord(c) < 128 for c in str(term)):
                    clean_list = []
                    for mapping in mappings:
                        if all(ord(c) < 128 for c in str(mapping)):
                            clean_list.append(mapping)
                    if clean_list:
                        clean_mappings[term] = clean_list
            
            # Save keyword mappings
            mapping_path = Config.DATA_DIR / 'keyword_mappings.json'
            with open(mapping_path, 'w', encoding='ascii', errors='replace') as f:
                json.dump(clean_mappings, f, ensure_ascii=True, indent=2)
            
            logger.info(f"Saved {len(clean_vocabulary)} clean vocabulary terms to {vocab_path}")
            logger.info(f"Saved {len(clean_mappings)} clean keyword mappings to {mapping_path}")
            
        except Exception as e:
            logger.error(f"Error saving vocabulary: {e}")
    
    def load_vocabulary(self) -> bool:
        """Load vocabulary from files with encoding safety"""
        try:
            # Load vocabulary with fallback encoding
            vocab_path = Config.DATA_DIR / 'vocabulary.json'
            if vocab_path.exists():
                try:
                    with open(vocab_path, 'r', encoding='ascii', errors='replace') as f:
                        vocab_data = json.load(f)
                        self.vocabulary = Counter(vocab_data)
                        self.term_frequencies = Counter(vocab_data)
                except UnicodeDecodeError:
                    # Fallback to UTF-8 if needed
                    with open(vocab_path, 'r', encoding='utf-8', errors='replace') as f:
                        vocab_data = json.load(f)
                        # Filter to ASCII-safe terms only
                        clean_vocabulary = {}
                        for term, count in vocab_data.items():
                            if all(ord(c) < 128 for c in str(term)):
                                clean_vocabulary[term] = count
                        self.vocabulary = Counter(clean_vocabulary)
                        self.term_frequencies = Counter(clean_vocabulary)
            
            # Load keyword mappings with fallback encoding
            mapping_path = Config.DATA_DIR / 'keyword_mappings.json'
            if mapping_path.exists():
                try:
                    with open(mapping_path, 'r', encoding='ascii', errors='replace') as f:
                        self.keyword_mappings = json.load(f)
                except UnicodeDecodeError:
                    # Fallback to UTF-8 if needed
                    with open(mapping_path, 'r', encoding='utf-8', errors='replace') as f:
                        raw_mappings = json.load(f)
                        # Filter to ASCII-safe mappings only
                        clean_mappings = {}
                        for term, mappings in raw_mappings.items():
                            if all(ord(c) < 128 for c in str(term)):
                                clean_list = []
                                for mapping in mappings:
                                    if all(ord(c) < 128 for c in str(mapping)):
                                        clean_list.append(mapping)
                                if clean_list:
                                    clean_mappings[term] = clean_list
                        self.keyword_mappings = clean_mappings
            
            logger.info(f"Vocabulary loaded: {len(self.vocabulary)} terms, {len(self.keyword_mappings)} mappings")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            return False
    
    def get_term_importance(self, term: str) -> float:
        """Calculate TF-IDF like importance score for a term"""
        if term not in self.vocabulary:
            return 0.0
        
        tf = self.term_frequencies[term]
        df = self.doc_frequencies[term]
        
        # Simple TF-IDF calculation
        if df > 0:
            return tf * np.log(self.total_documents / df)
        return 0.0
    
    def get_related_fields(self, term: str) -> List[str]:
        """Get fields related to a specific term"""
        return self.keyword_mappings.get(term, [])
    
    def enhance_query_terms(self, query_keywords: List[str]) -> List[str]:
        """Enhance query terms with related vocabulary"""
        enhanced_keywords = query_keywords.copy()
        
        for keyword in query_keywords:
            # Add related fields as potential search terms
            related_fields = self.get_related_fields(keyword)
            enhanced_keywords.extend(related_fields[:2])  # Limit to prevent expansion explosion
            
            # Add high-frequency terms that commonly appear with this keyword
            if keyword in self.vocabulary:
                # Simple co-occurrence enhancement based on field mappings
                for term in list(self.vocabulary.keys())[:100]:  # Check top 100 terms
                    if (term != keyword and 
                        keyword in term or term in keyword and 
                        len(term) > 2):
                        enhanced_keywords.append(term)
                        break  # Only add one related term to avoid noise
        
        # Remove duplicates and return
        return list(set(enhanced_keywords))
    
    def get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term from the Airbnb configuration"""
        from config.airbnb_config import AIRBNB_CONFIG
        synonyms = AIRBNB_CONFIG.get('synonyms', {})
        return synonyms.get(term.lower(), [])

# ============================================================================
# TEXT PROCESSING
# ============================================================================

class TextProcessor:
    """Handles text processing and NLP operations"""
    
    def __init__(self):
        self.nlp_model = None
        self._load_nlp_model()
    
    def _load_nlp_model(self):
        """Load SpaCy NLP model if available"""
        if spacy:
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("SpaCy model 'en_core_web_sm' not found. Basic text processing will be used.")
                self.nlp_model = None
    
    def extract_text_from_field(self, field_value: Any) -> str:
        """Extract text from various field types with improved encoding handling"""
        if field_value is None:
            return ""
        
        try:
            if isinstance(field_value, str):
                # Clean any problematic characters
                return self.clean_text(field_value)
            elif isinstance(field_value, (int, float)):
                return str(field_value)
            elif isinstance(field_value, list):
                text_parts = []
                for item in field_value:
                    if item is not None:
                        item_str = str(item)
                        cleaned_item = self.clean_text(item_str)
                        if cleaned_item:
                            text_parts.append(cleaned_item)
                return " ".join(text_parts)
            elif isinstance(field_value, dict):
                text_parts = []
                for value in field_value.values():
                    if value is not None:
                        value_str = str(value)
                        cleaned_value = self.clean_text(value_str)
                        if cleaned_value:
                            text_parts.append(cleaned_value)
                return " ".join(text_parts)
            else:
                field_str = str(field_value)
                return self.clean_text(field_str)
        except Exception as e:
            # Fallback for any encoding issues
            try:
                return str(field_value)[:1000]  # Truncate very long fields
            except:
                return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text without Unicode issues"""
        if not text:
            return ""
        
        # Handle encoding issues first
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-ASCII characters to avoid Unicode issues
        text = ''.join(char for char in text if ord(char) < 128)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\-\.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_terms(self, text: str) -> List[str]:
        """Extract terms from text"""
        if not text:
            return []
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        if self.nlp_model:
            # Use SpaCy for better term extraction
            doc = self.nlp_model(cleaned_text)
            terms = []
            
            for token in doc:
                if (not token.is_stop and 
                    not token.is_punct and 
                    not token.is_space and 
                    len(token.text) > 2):
                    terms.append(token.lemma_)
            
            return terms
        else:
            # Fallback to simple word splitting
            words = cleaned_text.split()
            return [word for word in words if len(word) > 2]
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        if not text or not self.nlp_model:
            return []
        
        doc = self.nlp_model(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities

# ============================================================================
# AIRBNB OPTIMIZATION
# ============================================================================

class KeywordExtractor:
    """Extracts and processes keywords from text."""
    
    def __init__(self):
        # Common stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
    def get_query_keywords(self, query: str) -> List[str]:
        """Extract keywords from a query string."""
        if not query:
            return []
            
        # Clean and split the query
        words = query.lower().strip().split()
        
        # Filter out stop words and short words
        keywords = [
            word.strip('.,!?;:"()[]{}') 
            for word in words 
            if len(word) > 2 and word.lower() not in self.stop_words
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword and keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
                
        return unique_keywords
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        entities = {
            'locations': [],
            'numbers': [],
            'amenities': []
        }
        
        # Simple entity extraction (can be enhanced with NLP libraries)
        words = text.lower().split()
        
        # Extract potential locations (capitalized words)
        for word in text.split():
            if word[0].isupper() and len(word) > 2:
                entities['locations'].append(word.lower())
                
        # Extract numbers
        import re
        numbers = re.findall(r'\d+', text)
        entities['numbers'] = numbers
        
        return entities

class AirbnbOptimizer:
    """Airbnb-specific optimizations and field handling"""
    
    # Airbnb-specific field priorities
    FIELD_PRIORITIES = {
        'name': 1.0,
        'description': 0.8,
        'neighborhood_overview': 0.7,
        'space': 0.6,
        'price': 0.9,
        'property_type': 0.8,
        'room_type': 0.7,
        'accommodates': 0.6,
        'bedrooms': 0.6,
        'bathrooms': 0.5,
        'amenities': 0.7,
        'review_scores_rating': 0.8,
        'neighbourhood_cleansed': 0.6,
        'host_name': 0.4,
        'host_about': 0.3
    }
    
    # Numeric fields for special handling
    NUMERIC_FIELDS = {
        'price', 'accommodates', 'bedrooms', 'bathrooms', 'beds',
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'reviews_per_month', 'calculated_host_listings_count'
    }
    
    def __init__(self):
        self.field_stats = {}
        self.vocabulary_manager = None
    
    def initialize_with_mongodb_data(self, documents: List[Dict[str, Any]]):
        """Initialize optimizer with MongoDB data"""
        logger.info("Initializing Airbnb optimizer with MongoDB data...")
        
        self._analyze_field_statistics(documents)
        logger.info(f"Analyzed {len(documents)} documents for field statistics")
    
    def _analyze_field_statistics(self, documents: List[Dict[str, Any]]):
        """Analyze field statistics from documents"""
        field_counts = defaultdict(int)
        field_types = defaultdict(set)
        
        for doc in documents:
            for field, value in doc.items():
                if field.startswith('_'):
                    continue
                
                if value is not None:
                    field_counts[field] += 1
                    field_types[field].add(type(value).__name__)
        
        total_docs = len(documents)
        
        for field, count in field_counts.items():
            self.field_stats[field] = {
                'presence_rate': count / total_docs,
                'types': list(field_types[field]),
                'is_numeric': field in self.NUMERIC_FIELDS
            }
    
    def create_searchable_text(self, document: Dict[str, Any]) -> str:
        """Create searchable text representation of document"""
        text_parts = []
        
        # Process fields by priority
        for field, priority in sorted(self.FIELD_PRIORITIES.items(), key=lambda x: x[1], reverse=True):
            if field in document and document[field] is not None:
                field_text = self._extract_field_text(document[field])
                if field_text:
                    # Weight important fields by repeating them
                    repeat_count = max(1, int(priority * 2))
                    text_parts.extend([field_text] * repeat_count)
        
        # Add other fields with lower priority
        for field, value in document.items():
            if field not in self.FIELD_PRIORITIES and not field.startswith('_'):
                field_text = self._extract_field_text(value)
                if field_text:
                    text_parts.append(field_text)
        
        return " ".join(text_parts)
    
    def _extract_field_text(self, field_value: Any) -> str:
        """Extract text from field value"""
        if field_value is None:
            return ""
        
        if isinstance(field_value, str):
            return field_value
        elif isinstance(field_value, (int, float)):
            return str(field_value)
        elif isinstance(field_value, list):
            return " ".join(str(item) for item in field_value if item is not None)
        else:
            return str(field_value)
    
    def extract_key_fields(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key fields for quick access"""
        key_fields = {}
        
        important_fields = ['name', 'price', 'property_type', 'room_type', 'accommodates', 
                           'bedrooms', 'bathrooms', 'review_scores_rating', 'neighbourhood_cleansed']
        
        for field in important_fields:
            if field in document:
                key_fields[field] = document[field]
        
        return key_fields
    
    def calculate_document_completeness(self, document: Dict[str, Any]) -> float:
        """Calculate how complete a document is"""
        important_fields = list(self.FIELD_PRIORITIES.keys())
        present_fields = sum(1 for field in important_fields if field in document and document[field] is not None)
        
        return present_fields / len(important_fields) if important_fields else 0.0
    
    def get_field_priority(self, field: str) -> float:
        """Get priority score for a field"""
        return self.FIELD_PRIORITIES.get(field, 0.1)
    
    def is_numeric_field(self, field: str) -> bool:
        """Check if field should be treated as numeric"""
        return field in self.NUMERIC_FIELDS
    
    def optimize_query_for_search(self, query: str) -> Dict[str, Any]:
        """Optimize query for Airbnb property search"""
        optimization_results = {
            'optimized_query': query,
            'extracted_keywords': [],
            'numeric_constraints': {},
            'property_filters': {},
            'query_intent': 'search'
        }
        
        query_lower = query.lower()
        
        # Extract basic keywords
        keyword_extractor = KeywordExtractor()
        optimization_results['extracted_keywords'] = keyword_extractor.get_query_keywords(query)
        
        # Extract numeric constraints with improved pattern matching
        numeric_constraints = {}
        
        # Price constraints
        price_patterns = [
            (r'under\s*\$?(\d+)', 'price', 'max'),
            (r'below\s*\$?(\d+)', 'price', 'max'),
            (r'less\s*than\s*\$?(\d+)', 'price', 'max'),
            (r'over\s*\$?(\d+)', 'price', 'min'),
            (r'above\s*\$?(\d+)', 'price', 'min'),
            (r'more\s*than\s*\$?(\d+)', 'price', 'min'),
            (r'\$?(\d+)\s*to\s*\$?(\d+)', 'price', 'range')
        ]
        
        for pattern, field, constraint_type in price_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                try:
                    if constraint_type == 'range' and len(matches[0]) >= 2:
                        numeric_constraints[field] = {
                            'min': float(matches[0][0]),
                            'max': float(matches[0][1])
                        }
                    else:
                        value = float(matches[0])
                        if field not in numeric_constraints:
                            numeric_constraints[field] = {}
                        numeric_constraints[field][constraint_type] = value
                    break
                except (ValueError, IndexError):
                    continue
        
        # Bedroom/bathroom constraints
        room_patterns = [
            (r'(\d+)\s*bedroom', 'bedrooms'),
            (r'(\d+)\s*bed(?!room)', 'bedrooms'),
            (r'(\d+)\s*br\b', 'bedrooms'),
            (r'(\d+)\s*bathroom', 'bathrooms'),
            (r'(\d+)\s*bath(?!room)', 'bathrooms')
        ]
        
        for pattern, field in room_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                try:
                    numeric_constraints[field] = {'exact': int(matches[0])}
                    break
                except (ValueError, IndexError):
                    continue
        
        # Guest capacity
        guest_patterns = [
            (r'(\d+)\s*guest', 'accommodates'),
            (r'(\d+)\s*people', 'accommodates'),
            (r'accommodate\s*(\d+)', 'accommodates')
        ]
        
        for pattern, field in guest_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                try:
                    numeric_constraints[field] = {'exact': int(matches[0])}
                    break
                except (ValueError, IndexError):
                    continue
        
        optimization_results['numeric_constraints'] = numeric_constraints
        
        # Property type filters
        property_filters = {}
        
        # Map common property types
        property_mappings = {
            'apartment': ['apartment', 'flat', 'condo', 'unit'],
            'house': ['house', 'home', 'villa', 'cottage'],
            'room': ['room', 'bedroom', 'private room'],
            'studio': ['studio', 'efficiency']
        }
        
        for prop_type, variants in property_mappings.items():
            if any(variant in query_lower for variant in variants):
                property_filters['property_type'] = prop_type
                break
        
        optimization_results['property_filters'] = property_filters
        
        # Determine query intent
        intent_indicators = {
            'search': ['find', 'search', 'look for', 'show me'],
            'filter': ['with', 'having', 'include'],
            'compare': ['compare', 'versus', 'better'],
            'price': ['price', 'cost', 'budget', 'cheap', 'expensive']
        }
        
        for intent, indicators in intent_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                optimization_results['query_intent'] = intent
                break
        
        return optimization_results

# ============================================================================
# INDEX MANAGEMENT
# ============================================================================

class DocumentProcessor:
    """Processes documents for indexing and search"""
    
    def __init__(self):
        self.airbnb_optimizer = AirbnbOptimizer()
        self.text_processor = TextProcessor()
        
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document for indexing"""
        processed_doc = {
            'document_id': str(document.get('_id', '')),
            'original_document': document,
            'searchable_text': '',
            'extracted_fields': {},
            'field_completion_rate': 0.0,
            'processing_timestamp': time.time()
        }
        
        # Create searchable text representation
        processed_doc['searchable_text'] = self.airbnb_optimizer.create_searchable_text(document)
        
        # Extract key fields for quick access
        processed_doc['extracted_fields'] = self.airbnb_optimizer.extract_key_fields(document)
        
        # Calculate document completeness
        processed_doc['field_completion_rate'] = self.airbnb_optimizer.calculate_document_completeness(document)
        
        return processed_doc
    
    def process_documents_batch(self, documents: List[Dict[str, Any]], 
                              batch_size: int = Config.BATCH_SIZE) -> List[Dict[str, Any]]:
        """Process multiple documents in batches"""
        processed_documents = []
        total_docs = len(documents)
        
        logger.info(f"Processing {total_docs} documents in batches of {batch_size}")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_start_time = time.time()
            
            batch_processed = []
            for doc in batch:
                try:
                    processed_doc = self.process_document(doc)
                    batch_processed.append(processed_doc)
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('_id', 'unknown')}: {e}")
                    continue
            
            processed_documents.extend(batch_processed)
            
            batch_time = time.time() - batch_start_time
            logger.info(f"Processed batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} "
                       f"({len(batch_processed)} docs) in {batch_time:.2f}s")
        
        logger.info(f"Successfully processed {len(processed_documents)} out of {total_docs} documents")
        return processed_documents

class EmbeddingGenerator:
    """Handles embedding generation with caching and error handling"""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        if SentenceTransformer is None:
            logger.error("SentenceTransformer not available")
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text with error handling"""
        if not self.model or not text:
            return None
        
        try:
            # Clean text to avoid encoding issues
            clean_text = ''.join(char for char in text if ord(char) < 128)
            if not clean_text.strip():
                return None
            
            embedding = self.model.encode([clean_text])[0]
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple texts"""
        if not self.model:
            return [None] * len(texts)
        
        try:
            # Clean all texts
            clean_texts = []
            for text in texts:
                clean_text = ''.join(char for char in text if ord(char) < 128)
                clean_texts.append(clean_text if clean_text.strip() else "")
            
            # Filter out empty texts but keep track of indices
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(clean_texts):
                if text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                return [None] * len(texts)
            
            # Generate embeddings
            embeddings = self.model.encode(valid_texts)
            
            # Map back to original indices
            result = [None] * len(texts)
            for i, valid_idx in enumerate(valid_indices):
                result[valid_idx] = embeddings[i]
            
            return result
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)


class IndexManager:
    """Manages FAISS indexes and embeddings"""
    
    def __init__(self):
        self.faiss_index = None
        self.processed_documents = []
        self.document_embeddings = {}
        self.embedding_model = None
        self.embedding_generator = EmbeddingGenerator()
        self.document_processor = DocumentProcessor()
        self.db_connector = MongoDBConnector()
        
    def _load_embedding_model(self) -> bool:
        """Load sentence transformer model"""
        if SentenceTransformer is None:
            logger.error("SentenceTransformer not available. Install with: pip install sentence-transformers")
            return False
        
        try:
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {Config.EMBEDDING_MODEL}")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def create_complete_index(self, rebuild: bool = False) -> bool:
        """Create complete FAISS index from MongoDB"""
        logger.info("Starting complete index creation...")
        
        # Load embedding model
        if not self._load_embedding_model():
            return False
        
        # Check if indexes exist and rebuild is not forced
        if not rebuild:
            loaded_index, loaded_docs = self.load_indexes()
            if loaded_index is not None and loaded_docs:
                logger.info("Existing indexes found, skipping rebuild")
                self.faiss_index = loaded_index
                self.processed_documents = loaded_docs
                return True
        
        # Connect to database
        if not self.db_connector.connect():
            logger.error("Failed to connect to database")
            return False
        
        # Get all documents
        documents = self.db_connector.get_all_documents()
        if not documents:
            logger.error("No documents found in database")
            return False
        
        # Process documents
        self.processed_documents = self.document_processor.process_documents_batch(documents)
        
        # Create embeddings
        if not self._create_embeddings():
            return False
        
        # Build FAISS index
        if not self._build_faiss_index():
            return False
        
        # Save indexes
        self.save_indexes()
        
        logger.info("Complete index creation finished successfully")
        return True
    
    def _create_embeddings(self) -> bool:
        """Create embeddings for processed documents"""
        logger.info("Creating embeddings for documents...")
        
        # Load cached embeddings if available
        cached_embeddings = self._load_cached_embeddings()
        if cached_embeddings:
            self.document_embeddings = cached_embeddings
            logger.info(f"Loaded {len(cached_embeddings)} cached embeddings")
        
        # Create embeddings for new documents
        texts_to_embed = []
        doc_ids_to_embed = []
        
        for doc in self.processed_documents:
            doc_id = doc['document_id']
            if doc_id not in self.document_embeddings:
                texts_to_embed.append(doc['searchable_text'])
                doc_ids_to_embed.append(doc_id)
        
        if texts_to_embed:
            logger.info(f"Creating embeddings for {len(texts_to_embed)} new documents")
            try:
                embeddings = self.embedding_model.encode(texts_to_embed, show_progress_bar=True)
                
                for doc_id, embedding in zip(doc_ids_to_embed, embeddings):
                    self.document_embeddings[doc_id] = embedding
                
                # Save cached embeddings
                self._save_cached_embeddings()
                
            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")
                return False
        
        logger.info(f"Total embeddings available: {len(self.document_embeddings)}")
        return True
    
    def _build_faiss_index(self) -> bool:
        """Build FAISS index from embeddings"""
        if faiss is None:
            logger.error("FAISS not available. Install with: pip install faiss-cpu")
            return False
        
        logger.info("Building FAISS index...")
        
        try:
            # Prepare embeddings matrix
            embedding_matrix = np.array([self.document_embeddings[doc['document_id']] 
                                       for doc in self.processed_documents 
                                       if doc['document_id'] in self.document_embeddings])
            
            if embedding_matrix.size == 0:
                logger.error("No embeddings available for index creation")
                return False
            
            # Create FAISS index
            dimension = embedding_matrix.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embedding_matrix)
            
            # Add embeddings to index
            self.faiss_index.add(embedding_matrix.astype(np.float32))
            
            logger.info(f"Built FAISS index with {self.faiss_index.ntotal} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return False
    
    def _load_cached_embeddings(self) -> Dict[str, np.ndarray]:
        """Load cached embeddings from disk"""
        try:
            if Config.EMBEDDINGS_CACHE_PATH.exists():
                with open(Config.EMBEDDINGS_CACHE_PATH, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cached embeddings: {e}")
        return {}
    
    def _save_cached_embeddings(self):
        """Save embeddings to cache"""
        try:
            with open(Config.EMBEDDINGS_CACHE_PATH, 'wb') as f:
                pickle.dump(self.document_embeddings, f)
            logger.info(f"Saved {len(self.document_embeddings)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving cached embeddings: {e}")
    
    def save_indexes(self):
        """Save FAISS index and processed documents"""
        try:
            # Save FAISS index
            if self.faiss_index:
                faiss.write_index(self.faiss_index, str(Config.FAISS_INDEX_PATH))
                logger.info(f"Saved FAISS index to {Config.FAISS_INDEX_PATH}")
            
            # Save processed documents
            with open(Config.PROCESSED_DOCS_PATH, 'wb') as f:
                pickle.dump(self.processed_documents, f)
            logger.info(f"Saved {len(self.processed_documents)} processed documents")
            
        except Exception as e:
            logger.error(f"Error saving indexes: {e}")
    
    def load_indexes(self) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        """Load FAISS index and processed documents"""
        try:
            faiss_index = None
            processed_docs = []
            
            # Load FAISS index
            if Config.FAISS_INDEX_PATH.exists() and faiss:
                faiss_index = faiss.read_index(str(Config.FAISS_INDEX_PATH))
                logger.info(f"Loaded FAISS index with {faiss_index.ntotal} embeddings")
            
            # Load processed documents
            if Config.PROCESSED_DOCS_PATH.exists():
                with open(Config.PROCESSED_DOCS_PATH, 'rb') as f:
                    processed_docs = pickle.load(f)
                logger.info(f"Loaded {len(processed_docs)} processed documents")
            
            return faiss_index, processed_docs
            
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
            return None, []
    
    def search_similar_documents(self, query_embedding: np.ndarray, 
                               k: int = Config.TOP_K_RESULTS) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents using FAISS index"""
        if self.faiss_index is None or not self.processed_documents:
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            # Return results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.processed_documents):
                    doc = self.processed_documents[idx]
                    results.append((doc, float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
            'processed_documents_count': len(self.processed_documents),
            'embedding_cache_size': len(self.document_embeddings),
            'index_files_exist': {
                'faiss_index': Config.FAISS_INDEX_PATH.exists(),
                'processed_docs': Config.PROCESSED_DOCS_PATH.exists(),
                'embeddings_cache': Config.EMBEDDINGS_CACHE_PATH.exists()
            }
        }
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics for system monitoring"""
        stats = {
            'faiss_index_size': 0,
            'processed_documents_count': len(self.processed_documents),
            'embedding_cache_size': len(self.document_embeddings),
            'embedding_model_loaded': self.embedding_generator.model is not None,
            'database_connected': self.db_connector.connected,
            'index_files_status': {
                'faiss_index_exists': Config.FAISS_INDEX_PATH.exists(),
                'processed_docs_exists': Config.PROCESSED_DOCS_PATH.exists(),
                'embeddings_cache_exists': Config.EMBEDDINGS_CACHE_PATH.exists()
            },
            'memory_usage': {
                'embeddings_mb': len(self.document_embeddings) * 384 * 4 / (1024 * 1024),  # Approx MB
                'documents_count': len(self.processed_documents)
            }
        }
        
        # Add FAISS index size if available
        if self.faiss_index:
            stats['faiss_index_size'] = self.faiss_index.ntotal
        
        return stats
