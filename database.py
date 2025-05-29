"""MongoDB operations and session management for JSON RAG System."""

import logging
import json
from typing import List, Dict, Any, Optional, Generator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import re
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from config import Config

class IntentType(Enum):
    """Types of user intents."""
    SEARCH = "search"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    NEW_TOPIC = "new_topic"
    GREETING = "greeting"
    UNKNOWN = "unknown"
    PRICE = "price"
    REVIEW = "review"

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    timestamp: str
    user_input: str
    intent: IntentType
    entities: List[str]
    search_query: str
    retrieved_docs: List[Dict[str, Any]]
    response: str
    context_used: List[str]

@dataclass
class SessionContext:
    """Holds the session context and state."""
    session_id: str
    created_at: str
    last_activity: str
    conversation_history: List[ConversationTurn]
    current_topic: Optional[str]
    mentioned_entities: List[str]
    search_context: List[str]
    user_preferences: Dict[str, Any]

class MongoDBConnector:
    """Handles MongoDB operations for the JSON RAG system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MongoDB connector."""
        self.config = config or Config.get_mongodb_config()
        self.client: Optional[MongoClient] = None
        self.database: Optional[Database] = None
        self.collection: Optional[Collection] = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(
                self.config["uri"],
                serverSelectionTimeoutMS=20000,
                connectTimeoutMS=20000,
                socketTimeoutMS=30000
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            self.database = self.client[self.config["database"]]
            self.collection = self.database[self.config["collection"]]
            
            self.logger.info(f"Connected to MongoDB: {self.config['database']}.{self.config['collection']}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to MongoDB: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
            self.collection = None
            self.logger.info("Disconnected from MongoDB")
    
    def ensure_indexes(self) -> bool:
        """Create optimized indexes for better search performance."""
        if self.collection is None:
            self.logger.error("No collection available for indexing")
            return False
        
        try:
            # Get existing indexes
            existing_indexes = list(self.collection.list_indexes())
            
            # Drop any existing text index on $**
            for idx in existing_indexes:
                key = dict(idx.get('key', {}))
                if key == {"$**": "text"}:
                    self.logger.info(f"Dropping existing text index: {idx['name']}")
                    self.collection.drop_index(idx['name'])
            
            # Refresh index list
            existing_indexes = list(self.collection.list_indexes())
            index_names = [idx['name'] for idx in existing_indexes]
            
            # Create comprehensive text index
            if 'comprehensive_text_index' not in index_names:
                self.collection.create_index(
                    [("$**", TEXT)],
                    name="comprehensive_text_index",
                    background=True,
                    default_language='english'
                )
                self.logger.info("Created comprehensive text search index")
            
            # Create specific field indexes
            field_indexes = {
                'name_index': 'name',
                'summary_index': 'summary',
                'description_index': 'description',
                'space_index': 'space',
                'location_index': 'address.location',
                'price_index': 'price',
                'property_type_index': 'property_type',
                'room_type_index': 'room_type'
            }
            
            for index_name, field in field_indexes.items():
                if index_name not in index_names:
                    try:
                        self.collection.create_index(
                            [(field, ASCENDING)],
                            name=index_name,
                            background=True,
                            sparse=True
                        )
                        self.logger.info(f"Created index on field: {field}")
                    except Exception as e:
                        self.logger.warning(f"Could not create index on {field}: {e}")
            
            # Create compound indexes
            compound_indexes = [
                {
                    'name': 'location_price_compound',
                    'fields': [("address.location", ASCENDING), ("price", ASCENDING)]
                },
                {
                    'name': 'type_location_compound',
                    'fields': [("property_type", ASCENDING), ("address.location", ASCENDING)]
                },
                {
                    'name': 'room_price_compound',
                    'fields': [("room_type", ASCENDING), ("price", ASCENDING)]
                }
            ]
            
            for compound_idx in compound_indexes:
                if compound_idx['name'] not in index_names:
                    try:
                        self.collection.create_index(
                            compound_idx['fields'],
                            name=compound_idx['name'],
                            background=True,
                            sparse=True
                        )
                        self.logger.info(f"Created compound index: {compound_idx['name']}")
                    except Exception as e:
                        self.logger.warning(f"Could not create compound index {compound_idx['name']}: {e}")
            
            # Log final index count
            final_indexes = list(self.collection.list_indexes())
            self.logger.info(f"Total indexes after optimization: {len(final_indexes)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {e}")
            return False
    
    def get_all_documents(self, batch_size: int = 1000) -> Generator[Dict[str, Any], None, None]:
        """Retrieve all documents from the collection in batches."""
        if self.collection is None:
            self.logger.error("No collection available")
            return
        
        try:
            cursor = self.collection.find({}).batch_size(batch_size)
            for document in cursor:
                yield document
                
        except Exception as e:
            self.logger.error(f"Error fetching documents: {e}")
    
    def get_documents_by_ids(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve specific documents by their IDs."""
        if self.collection is None:
            self.logger.error("No collection available")
            return []
        
        try:
            from bson import ObjectId
            
            # Convert string IDs to ObjectId if needed
            object_ids = []
            for doc_id in document_ids:
                try:
                    if isinstance(doc_id, str) and len(doc_id) == 24:
                        object_ids.append(ObjectId(doc_id))
                    else:
                        object_ids.append(doc_id)
                except Exception:
                    object_ids.append(doc_id)
            
            documents = list(self.collection.find({"_id": {"$in": object_ids}}))
            self.logger.info(f"Retrieved {len(documents)} documents by IDs")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents by IDs: {e}")
            return []
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform text search on the collection."""
        if self.collection is None:
            self.logger.error("No collection available")
            return []
        
        try:
            documents = list(
                self.collection.find(
                    {"$text": {"$search": query}},
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            )
            
            self.logger.info(f"Found {len(documents)} documents for query: {query}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if self.collection is None:
            return {"error": "No collection available"}
        
        try:
            stats = self.database.command("collStats", self.config["collection"])
            return {
                "document_count": stats.get("count", 0),
                "data_size": stats.get("size", 0),
                "average_document_size": stats.get("avgObjSize", 0),
                "indexes": len(list(self.collection.list_indexes()))
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

class SessionManager:
    """Manages user sessions, context, and conversation memory."""
    
    def __init__(self):
        """Initialize the session manager."""
        self.logger = logging.getLogger(__name__)
        self.sessions: Dict[str, SessionContext] = {}
        
        # Intent detection keywords
        self.intent_keywords = {
            IntentType.GREETING: ["hello", "hi", "hey", "good morning", "good afternoon", "greetings"],
            IntentType.FOLLOW_UP: ["more", "tell me more", "continue", "what else", "also", "and", "additionally"],
            IntentType.CLARIFICATION: ["what do you mean", "explain", "clarify", "what is", "how", "why"],
            IntentType.NEW_TOPIC: ["now", "instead", "different", "change topic", "new", "something else"],
            IntentType.PRICE: ["price", "cost", "night", "rate"],
            IntentType.REVIEW: ["review", "rating", "feedback"]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            "numbers": r"\b\d+(?:\.\d+)?\b",
            "dates": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
            "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "urls": r"https?://[^\s]+",
            "currencies": r"\$\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP)"
        }
    
    def create_session(self, session_id: str) -> SessionContext:
        """Create a new session."""
        now = datetime.utcnow().isoformat()
        
        session_context = SessionContext(
            session_id=session_id,
            created_at=now,
            last_activity=now,
            conversation_history=[],
            current_topic=None,
            mentioned_entities=[],
            search_context=[],
            user_preferences={}
        )
        
        self.sessions[session_id] = session_context
        self.logger.info(f"Created new session: {session_id}")
        return session_context
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get existing session or create new one."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Check if session expired
            last_activity = datetime.fromisoformat(session.last_activity)
            if datetime.utcnow() - last_activity > timedelta(seconds=Config.SESSION_TIMEOUT):
                self.logger.info(f"Session {session_id} expired, creating new one")
                del self.sessions[session_id]
                return self.create_session(session_id)
            
            return session
        
        return self.create_session(session_id)
    
    def detect_intent(self, user_input: str, session_context: SessionContext) -> IntentType:
        """Detect user intent based on input and context."""
        user_input_lower = user_input.lower()
        
        # Check for greeting
        if any(greeting in user_input_lower for greeting in self.intent_keywords[IntentType.GREETING]):
            return IntentType.GREETING
        
        # Check for follow-up if there's conversation history
        if session_context.conversation_history:
            if any(keyword in user_input_lower for keyword in self.intent_keywords[IntentType.FOLLOW_UP]):
                return IntentType.FOLLOW_UP
            
            if any(keyword in user_input_lower for keyword in self.intent_keywords[IntentType.CLARIFICATION]):
                return IntentType.CLARIFICATION
        
        # Check for topic change
        if any(keyword in user_input_lower for keyword in self.intent_keywords[IntentType.NEW_TOPIC]):
            return IntentType.NEW_TOPIC
        
        # Check for price inquiry
        if any(keyword in user_input_lower for keyword in self.intent_keywords[IntentType.PRICE]):
            return IntentType.PRICE
        
        # Check for review inquiry
        if any(keyword in user_input_lower for keyword in self.intent_keywords[IntentType.REVIEW]):
            return IntentType.REVIEW
        
        # Default to search intent
        return IntentType.SEARCH
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using regex patterns."""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(f"{entity_type}:{match}")
        
        # Extract potential keywords (capitalized words, specific terms)
        words = text.split()
        for word in words:
            if len(word) > 2 and (word.isupper() or word.istitle()):
                entities.append(f"keyword:{word}")
        
        return list(set(entities))  # Remove duplicates
    
    def enhance_query_with_context(self, user_input: str, session_context: SessionContext) -> str:
        """Enhance user query with session context."""
        enhanced_query = user_input
        
        # Add context from recent conversation
        if session_context.conversation_history:
            recent_turns = session_context.conversation_history[-3:]  # Last 3 turns
            context_terms = []
            
            for turn in recent_turns:
                # Add entities from recent turns
                for entity in turn.entities:
                    if ":" in entity:
                        entity_value = entity.split(":", 1)[1]
                        context_terms.append(entity_value)
                
                # Add important terms from search queries
                if turn.search_query and turn.search_query != user_input:
                    query_words = turn.search_query.split()
                    context_terms.extend([word for word in query_words if len(word) > 3])
            
            # Filter and add unique context terms
            unique_context = list(set(context_terms))[:5]  # Limit to 5 terms
            if unique_context:
                enhanced_query += " " + " ".join(unique_context)
        
        # Add current topic context
        if session_context.current_topic:
            enhanced_query += f" {session_context.current_topic}"
        
        return enhanced_query.strip()
    
    def update_session_context(self, session_context: SessionContext,
                             user_input: str, intent: IntentType,
                             entities: List[str], search_query: str,
                             retrieved_docs: List[Dict[str, Any]],
                             response: str) -> None:
        """Update session context with new conversation turn."""
        now = datetime.utcnow().isoformat()
        
        # Create conversation turn
        turn = ConversationTurn(
            timestamp=now,
            user_input=user_input,
            intent=intent,
            entities=entities,
            search_query=search_query,
            retrieved_docs=retrieved_docs,
            response=response,
            context_used=session_context.search_context.copy()
        )
        
        # Update session
        session_context.conversation_history.append(turn)
        session_context.last_activity = now
        
        # Update entities
        session_context.mentioned_entities.extend(entities)
        session_context.mentioned_entities = list(set(session_context.mentioned_entities))[-20:]  # Keep last 20
        
        # Update search context
        if search_query:
            session_context.search_context.append(search_query)
            session_context.search_context = session_context.search_context[-10:]  # Keep last 10
        
        # Update current topic based on intent
        if intent == IntentType.NEW_TOPIC or not session_context.current_topic:
            # Extract topic from search query
            query_words = search_query.split() if search_query else user_input.split()
            important_words = [word for word in query_words if len(word) > 3]
            if important_words:
                session_context.current_topic = " ".join(important_words[:3])
        
        # Limit conversation history
        if len(session_context.conversation_history) > Config.MAX_SESSION_HISTORY:
            session_context.conversation_history = session_context.conversation_history[-Config.MAX_SESSION_HISTORY:]
        
        self.logger.debug(f"Updated session {session_context.session_id} with new turn")
    
    def get_conversation_summary(self, session_context: SessionContext, last_n_turns: int = 5) -> str:
        """Get a summary of recent conversation."""
        if not session_context.conversation_history:
            return "No previous conversation."
        
        recent_turns = session_context.conversation_history[-last_n_turns:]
        
        summary_parts = []
        summary_parts.append(f"Current topic: {session_context.current_topic or 'General inquiry'}")
        
        if session_context.mentioned_entities:
            entities_str = ", ".join(session_context.mentioned_entities[-5:])
            summary_parts.append(f"Recent entities: {entities_str}")
        
        summary_parts.append(f"Recent queries: {len(recent_turns)} turns")
        
        return "; ".join(summary_parts)
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            last_activity = datetime.fromisoformat(session.last_activity)
            if current_time - last_activity > timedelta(seconds=Config.SESSION_TIMEOUT):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session data as dictionary."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return {
            "session_context": asdict(session),
            "exported_at": datetime.utcnow().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test MongoDB connection
    with MongoDBConnector() as mongo:
        if mongo.collection is not None:
            mongo.ensure_indexes()
            stats = mongo.get_collection_stats()
            print(f"Collection stats: {stats}")
        else:
            print("Failed to connect to MongoDB")
    
    # Test session management
    session_manager = SessionManager()
    session = session_manager.get_session("test_session_123")
    print(f"Created session: {session.session_id}")
    
    # Test intent detection
    test_inputs = [
        "Hello there!",
        "I'm looking for information about Python programming",
        "Tell me more about that",
        "What do you mean by that?",
        "Now I want to know about databases"
    ]
    
    for user_input in test_inputs:
        intent = session_manager.detect_intent(user_input, session)
        entities = session_manager.extract_entities(user_input)
        enhanced_query = session_manager.enhance_query_with_context(user_input, session)
        
        print(f"\nInput: {user_input}")
        print(f"Intent: {intent.value}")
        print(f"Entities: {entities}")
        print(f"Enhanced query: {enhanced_query}")
        
        # Update session
        session_manager.update_session_context(
            session, user_input, intent, entities, enhanced_query, [], f"Response to: {user_input}"
        )
    
    print(f"\nConversation summary: {session_manager.get_conversation_summary(session)}")
