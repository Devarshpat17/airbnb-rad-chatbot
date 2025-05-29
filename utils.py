"""Utility classes for text processing, keyword extraction, and Airbnb optimization."""

import json
import logging
import pickle
import re
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
import decimal
from bson.decimal128 import Decimal128

# NLTK imports with error handling
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
    
    # Download required NLTK data
    for requirement in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(f'tokenizers/{requirement}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{requirement}')
            except LookupError:
                try:
                    nltk.data.find(f'taggers/{requirement}')
                except LookupError:
                    nltk.download(requirement, quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, using basic text processing")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using basic keyword extraction")

from config import Config

class TextProcessor:
    """Process JSON documents into clean, searchable text."""
    
    def __init__(self):
        """Initialize the text processor."""
        self.logger = logging.getLogger(__name__)
        self.ignore_values = self._normalize_ignore_values(Config.IGNORE_VALUES)
        self.processed_count = 0
        self.skipped_count = 0
        self.airbnb_optimizer = AirbnbOptimizer()
    
    def _normalize_ignore_values(self, ignore_set: Set[str]) -> Set[str]:
        """Normalize ignore values to lowercase."""
        normalized = set()
        for value in ignore_set:
            normalized.add(str(value).lower().strip())
        return normalized
    
    def _should_ignore_value(self, value: Any) -> bool:
        """Check if a value should be ignored."""
        if value is None:
            return True
        
        str_value = str(value).lower().strip()
        
        if str_value in self.ignore_values:
            return True
        
        try:
            if float(str_value) == 0:
                return True
        except (ValueError, TypeError):
            pass
        
        if len(str_value) == 0 or str_value.isspace():
            return True
        
        if len(str_value) < Config.MIN_TEXT_LENGTH:
            return True
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', ' ', text)
        text = re.sub(r'[""'']', '"', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
        
        text = text.strip()
        if len(text) > Config.MAX_TEXT_LENGTH:
            text = text[:Config.MAX_TEXT_LENGTH].rsplit(' ', 1)[0] + '...'
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Public interface for text cleaning."""
        if not text or not isinstance(text, str):
            return ""
        return self._clean_text(text)
    
    def _extract_text_from_json(self, data: Any, parent_key: str = "", max_depth: int = 10) -> List[Tuple[str, str]]:
        """Recursively extract text from JSON data structure."""
        if max_depth <= 0:
            return []
        
        text_extracts = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "_id":
                    continue
                
                current_key = f"{parent_key}.{key}" if parent_key else key
                
                if isinstance(value, (dict, list)):
                    text_extracts.extend(
                        self._extract_text_from_json(value, current_key, max_depth - 1)
                    )
                elif not self._should_ignore_value(value):
                    cleaned_text = self._clean_text(str(value))
                    if cleaned_text:
                        text_extracts.append((current_key, cleaned_text))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                
                if isinstance(item, (dict, list)):
                    text_extracts.extend(
                        self._extract_text_from_json(item, current_key, max_depth - 1)
                    )
                elif not self._should_ignore_value(item):
                    cleaned_text = self._clean_text(str(item))
                    if cleaned_text:
                        text_extracts.append((current_key, cleaned_text))
        
        else:
            if not self._should_ignore_value(data):
                cleaned_text = self._clean_text(str(data))
                if cleaned_text:
                    text_extracts.append((parent_key or "value", cleaned_text))
        
        return text_extracts
    
    def process_document(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single JSON document into searchable text."""
        try:
            doc_id = str(document.get("_id", ""))
            
            # Use optimized Airbnb text extraction
            try:
                searchable_text = self.airbnb_optimizer.get_searchable_text(document)
                if not searchable_text or len(searchable_text.strip()) < 10:
                    # Fallback to original method
                    text_extracts = self._extract_text_from_json(document)
                    if not text_extracts:
                        self.skipped_count += 1
                        return None
                    searchable_text = " ".join([text for _, text in text_extracts])
                    extracted_fields = dict(text_extracts)
                else:
                    # Also extract fields for backward compatibility
                    text_extracts = self._extract_text_from_json(document)
                    extracted_fields = dict(text_extracts) if text_extracts else {}
            except Exception as e:
                self.logger.warning(f"Airbnb optimization failed for {doc_id}, using fallback: {e}")
                text_extracts = self._extract_text_from_json(document)
                if not text_extracts:
                    self.skipped_count += 1
                    return None
                searchable_text = " ".join([text for _, text in text_extracts])
                extracted_fields = dict(text_extracts)
            
            # Generate combined text for display
            if 'name' in document:
                combined_text = f"Name: {document['name']}\n"
            else:
                combined_text = f"Document ID: {doc_id}\n"
            
            # Add key information
            key_fields = ['property_type', 'room_type', 'price', 'accommodates', 'bedrooms', 'market']
            for field in key_fields:
                if field in document and document[field]:
                    combined_text += f"{field.replace('_', ' ').title()}: {document[field]}\n"
            
            processed_doc = {
                "document_id": doc_id,
                "source_document": document,
                "extracted_fields": extracted_fields,
                "combined_text": combined_text.strip(),
                "searchable_text": searchable_text,
                "field_count": len(extracted_fields),
                "text_length": len(searchable_text),
                "processed_at": datetime.utcnow().isoformat(),
                "optimization_used": "airbnb_optimized"
            }
            
            self.processed_count += 1
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"Error processing document {document.get('_id', 'unknown')}: {e}")
            self.skipped_count += 1
            return None
    
    def _json_serial(self, obj):
        """JSON serializer for objects not serializable by default."""
        if isinstance(obj, (datetime, )):
            return obj.isoformat()
        return str(obj)
    
    def process_and_save_documents(self, output_file: str = None) -> bool:
        """Process all documents from MongoDB and save to file."""
        output_file = output_file or Config.PROCESSED_TEXT_FILE
        
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            from database import MongoDBConnector
            with MongoDBConnector() as mongo:
                if mongo.collection is None:
                    self.logger.error("Failed to connect to MongoDB")
                    return False
                
                mongo.ensure_indexes()
                stats = mongo.get_collection_stats()
                total_docs = stats.get("document_count", 0)
                
                self.logger.info(f"Processing {total_docs} documents from MongoDB")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for document in mongo.get_all_documents():
                        processed_doc = self.process_document(document)
                        if processed_doc:
                            f.write(json.dumps(processed_doc, ensure_ascii=False, default=self._json_serial) + '\n')
                        
                        if (self.processed_count + self.skipped_count) % 100 == 0:
                            self.logger.info(
                                f"Processed: {self.processed_count}, "
                                f"Skipped: {self.skipped_count}"
                            )
                
                self.logger.info(
                    f"Processing complete. Processed: {self.processed_count}, "
                    f"Skipped: {self.skipped_count}, Output: {output_file}"
                )
                return True
                
        except Exception as e:
            self.logger.error(f"Error during document processing: {e}")
            return False
    
    def load_processed_documents(self, input_file: str = None) -> List[Dict[str, Any]]:
        """Load processed documents from file."""
        input_file = input_file or Config.PROCESSED_TEXT_FILE
        
        try:
            if not Path(input_file).exists():
                self.logger.warning(f"Processed file {input_file} does not exist")
                return []
            
            documents = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    try:
                        if line.strip():
                            doc = json.loads(line)
                            documents.append(doc)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON on line {line_no}: {e}")
            
            self.logger.info(f"Loaded {len(documents)} processed documents from {input_file}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading processed documents: {e}")
            return []
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for search."""
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = self._clean_text(text)
        return text

class KeywordExtractor:
    """Extract and manage keywords for improved search functionality."""
    
    def __init__(self):
        """Initialize the keyword extractor."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
            except LookupError:
                self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
                self.lemmatizer = None
        else:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.lemmatizer = None
        
        # Add domain-specific stop words
        self.stop_words.update({
            'airbnb', 'listing', 'room', 'apartment', 'house', 'place', 'property',
            'guest', 'host', 'stay', 'night', 'day', 'week', 'month', 'year',
            'available', 'book', 'booking', 'reserve', 'reservation'
        })
        
        # Keyword storage
        self.vocabulary = {}
        self.keyword_weights = {}
        self.field_keywords = defaultdict(set)
        self.category_keywords = defaultdict(set)
        
        # TF-IDF vectorizer if available
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95
            )
        else:
            self.tfidf_vectorizer = None
        
        # Paths for saving/loading
        self.vocab_path = Path(Config.DATA_DIR) / "keyword_vocabulary.pkl"
        self.weights_path = Path(Config.DATA_DIR) / "keyword_weights.pkl"
        self.categories_path = Path(Config.DATA_DIR) / "category_keywords.pkl"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        
        # Remove URLs, emails, and special characters
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_keywords_from_text(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text."""
        if not text:
            return []
        
        # Clean text
        clean_text = self._clean_text(text)
        
        if len(clean_text) < min_length:
            return []
        
        # Tokenize
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(clean_text)
            except Exception:
                tokens = clean_text.split()
        else:
            tokens = clean_text.split()
        
        # Filter tokens
        keywords = []
        for token in tokens:
            if (len(token) >= min_length and
                    token not in self.stop_words and
                    token.isalpha()):
                
                # Lemmatize if available
                if self.lemmatizer:
                    try:
                        lemmatized = self.lemmatizer.lemmatize(token)
                        keywords.append(lemmatized)
                    except Exception:
                        keywords.append(token)
                else:
                    keywords.append(token)
        
        return keywords
    
    def _get_important_phrases(self, text: str) -> List[str]:
        """Extract important noun phrases from text."""
        if not text or not NLTK_AVAILABLE:
            return []
        
        clean_text = self._clean_text(text)
        
        try:
            tokens = word_tokenize(clean_text)
            pos_tags = pos_tag(tokens)
            
            phrases = []
            current_phrase = []
            
            for word, pos in pos_tags:
                # Look for noun phrases (adjectives + nouns)
                if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
                    if len(word) >= 3 and word not in self.stop_words:
                        if self.lemmatizer:
                            current_phrase.append(self.lemmatizer.lemmatize(word))
                        else:
                            current_phrase.append(word)
                else:
                    if len(current_phrase) >= 2:  # Multi-word phrases
                        phrases.append(' '.join(current_phrase))
                    current_phrase = []
            
            # Don't forget the last phrase
            if len(current_phrase) >= 2:
                phrases.append(' '.join(current_phrase))
            
            return phrases
            
        except Exception as e:
            self.logger.warning(f"Error extracting phrases: {e}")
            return []
    
    def build_vocabulary_from_db(self) -> bool:
        """Build keyword vocabulary from MongoDB documents."""
        self.logger.info("Building keyword vocabulary from database...")
        
        try:
            from database import MongoDBConnector
            with MongoDBConnector() as mongo:
                if mongo.collection is None:
                    self.logger.error("Could not connect to MongoDB")
                    return False
                
                # Get documents for vocabulary building (limit for performance)
                documents = list(mongo.get_all_documents(batch_size=500))
                
                # For large datasets, sample documents to speed up vocabulary building
                if len(documents) > 1000:
                    import random
                    random.seed(42)  # For reproducible sampling
                    documents = random.sample(documents, 1000)
                    self.logger.info(f"Sampled {len(documents)} documents from large dataset for vocabulary building")
                else:
                    self.logger.info(f"Processing {len(documents)} documents for vocabulary building")
                
                all_texts = []
                field_stats = defaultdict(int)
                
                # Process each document
                for doc in documents:
                    doc_keywords = set()
                    doc_text_parts = []
                    
                    # Extract text from various fields
                    text_fields = [
                        'name', 'summary', 'description', 'space', 'neighborhood_overview',
                        'notes', 'transit', 'access', 'interaction', 'house_rules'
                    ]
                    
                    for field in text_fields:
                        if field in doc and doc[field]:
                            field_text = str(doc[field])
                            doc_text_parts.append(field_text)
                            
                            # Extract keywords for this field
                            field_keywords = self._extract_keywords_from_text(field_text)
                            phrases = self._get_important_phrases(field_text)
                            
                            # Add to field-specific keywords
                            self.field_keywords[field].update(field_keywords + phrases)
                            doc_keywords.update(field_keywords + phrases)
                            field_stats[field] += 1
                    
                    # Combine all text for document-level analysis
                    doc_text = ' '.join(doc_text_parts)
                    if doc_text:
                        all_texts.append(doc_text)
                    
                    # Categorize based on property type, room type, etc.
                    if 'property_type' in doc:
                        prop_type = str(doc['property_type']).lower()
                        self.category_keywords[f"property_{prop_type}"].update(doc_keywords)
                    
                    if 'room_type' in doc:
                        room_type = str(doc['room_type']).lower()
                        self.category_keywords[f"room_{room_type}"].update(doc_keywords)
                
                # Use TF-IDF if available
                if self.tfidf_vectorizer and all_texts and SKLEARN_AVAILABLE:
                    self.logger.info("Computing TF-IDF scores...")
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
                    feature_names = self.tfidf_vectorizer.get_feature_names_out()
                    
                    # Calculate average TF-IDF scores
                    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                    
                    # Create vocabulary with weights
                    for i, term in enumerate(feature_names):
                        self.vocabulary[term] = {
                            'tfidf_score': float(mean_scores[i]),
                            'frequency': int(np.sum(tfidf_matrix.toarray()[:, i] > 0)),
                            'type': 'tfidf_term'
                        }
                        self.keyword_weights[term] = float(mean_scores[i])
                
                # Add field-specific high-frequency terms
                for field, keywords in self.field_keywords.items():
                    field_counter = Counter(keywords)
                    top_field_keywords = field_counter.most_common(100)
                    
                    for keyword, freq in top_field_keywords:
                        if keyword not in self.vocabulary:
                            self.vocabulary[keyword] = {
                                'frequency': freq,
                                'primary_field': field,
                                'type': 'field_specific'
                            }
                            self.keyword_weights[keyword] = freq / len(documents)
                
                self.logger.info(f"Built vocabulary with {len(self.vocabulary)} terms")
                self.logger.info(f"Field statistics: {dict(field_stats)}")
                
                # Save vocabulary
                self.save_vocabulary()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error building vocabulary: {e}")
            return False
    
    def get_query_keywords(self, query: str, strict: bool = False, min_score: float = 0.0) -> Dict[str, float]:
        """Extract keywords from a search query with weights."""
        query_keywords = {}
        
        # Extract basic keywords
        keywords = self._extract_keywords_from_text(query)
        phrases = self._get_important_phrases(query)
        
        all_terms = keywords + phrases
        
        for term in all_terms:
            if term in self.vocabulary:
                # Use pre-computed weight
                weight = self.keyword_weights.get(term, 0.5)
                if not strict or weight >= min_score:
                    query_keywords[term] = weight
            elif not strict:
                # Default weight for unknown terms
                query_keywords[term] = 0.3
        
        # Boost exact matches
        query_lower = query.lower()
        for vocab_term in self.vocabulary:
            if vocab_term in query_lower:
                weight = self.keyword_weights.get(vocab_term, 0.5) * 1.5
                if not strict or weight >= min_score:
                    query_keywords[vocab_term] = weight
        
        return query_keywords
    
    def save_vocabulary(self) -> bool:
        """Save vocabulary and related data to disk."""
        try:
            # Save main vocabulary
            with open(self.vocab_path, 'wb') as f:
                pickle.dump(self.vocabulary, f)
            
            # Save keyword weights
            with open(self.weights_path, 'wb') as f:
                pickle.dump(self.keyword_weights, f)
            
            # Save category keywords
            with open(self.categories_path, 'wb') as f:
                pickle.dump(dict(self.category_keywords), f)
            
            self.logger.info(f"Saved vocabulary to {self.vocab_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving vocabulary: {e}")
            return False
    
    def load_vocabulary(self) -> bool:
        """Load vocabulary from disk."""
        try:
            if self.vocab_path.exists():
                with open(self.vocab_path, 'rb') as f:
                    self.vocabulary = pickle.load(f)
                
                with open(self.weights_path, 'rb') as f:
                    self.keyword_weights = pickle.load(f)
                
                with open(self.categories_path, 'rb') as f:
                    category_data = pickle.load(f)
                    self.category_keywords = defaultdict(set, category_data)
                
                self.logger.info(f"Loaded vocabulary with {len(self.vocabulary)} terms")
                return True
            else:
                self.logger.info("No saved vocabulary found, attempting to build from database...")
                # Try to build vocabulary from database (this may take time on first run)
                try:
                    success = self.build_vocabulary_from_db()
                    if success:
                        self.logger.info("Successfully built and saved vocabulary from database")
                        return True
                    else:
                        self.logger.warning("Failed to build vocabulary from database, continuing without vocabulary")
                        return False
                except Exception as e:
                    self.logger.warning(f"Error building vocabulary from database: {e}, continuing without vocabulary")
                    return False
                
        except Exception as e:
            self.logger.error(f"Error loading vocabulary: {e}")
            return False

class AirbnbOptimizer:
    """Optimization utilities specifically for Airbnb listings."""
    
    def __init__(self):
        """Initialize the Airbnb optimizer."""
        self.logger = logging.getLogger(__name__)
        
        # Essential fields for quick overview (ordered by importance)
        self.key_fields = [
            'name',
            'price',
            'property_type',
            'room_type',
            'accommodates',
            'bedrooms',
            'bathrooms',
            'beds',
            'minimum_nights',
            'maximum_nights',
            'instant_bookable',
            'neighbourhood_cleansed',
            'market'
        ]
        
        # Fields to prioritize for search text
        self.search_priority_fields = [
            'name',
            'summary',
            'description',
            'space',
            'neighborhood_overview',
            'notes',
            'transit',
            'access',
            'interaction',
            'house_rules',
            'amenities',
            'host_about'
        ]
        
        # Fields to exclude from display (sensitive or irrelevant)
        self.exclude_fields = {
            'host_id',
            'host_url',
            'host_thumbnail_url',
            'host_picture_url',
            'listing_url',
            'picture_url',
            'xl_picture_url',
            'medium_url',
            'thumbnail_url',
            'scrape_id',
            'last_scraped',
            'calendar_last_scraped'
        }
    
    def extract_key_info(self, source_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from Airbnb listing for quick display."""
        if not source_doc:
            return {}
        
        key_info = {}
        
        # Extract essential fields first
        for field in self.key_fields:
            if field in source_doc and source_doc[field] is not None:
                value = source_doc[field]
                
                # Format specific fields
                if field == 'price':
                    # Clean price format
                    if isinstance(value, str):
                        value = value.replace('$', '').replace(',', '')
                        try:
                            value = f"${float(value):.2f}"
                        except (ValueError, TypeError):
                            pass
                elif field == 'instant_bookable':
                    # Convert to boolean
                    if isinstance(value, str):
                        value = value.lower() in ['t', 'true', '1', 'yes']
                
                key_info[field] = value
        
        # Extract location information
        location_info = self._extract_location(source_doc)
        if location_info:
            key_info.update(location_info)
        
        # Add description preview
        description_preview = self._get_description_preview(source_doc)
        if description_preview:
            key_info['description_preview'] = description_preview
        
        # Add amenities count
        if 'amenities' in source_doc:
            amenities = source_doc['amenities']
            if isinstance(amenities, list):
                key_info['amenities_count'] = len(amenities)
                # Include top 5 amenities
                key_info['top_amenities'] = amenities[:5]
            elif isinstance(amenities, str):
                # Parse amenities string if needed
                try:
                    amenities_list = json.loads(amenities)
                    if isinstance(amenities_list, list):
                        key_info['amenities_count'] = len(amenities_list)
                        key_info['top_amenities'] = amenities_list[:5]
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Add review summary
        review_fields = ['number_of_reviews', 'review_scores_rating', 'reviews_per_month']
        review_info = {}
        for field in review_fields:
            if field in source_doc and source_doc[field] is not None:
                review_info[field] = source_doc[field]
        
        if review_info:
            key_info['review_summary'] = review_info
        
        return key_info
    
    def _extract_location(self, source_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract location information from various location fields."""
        location_info = {}
        
        # Primary location fields
        location_fields = [
            'neighbourhood_cleansed',
            'neighbourhood_group_cleansed',
            'city',
            'market',
            'smart_location',
            'country_code',
            'country'
        ]
        
        for field in location_fields:
            if field in source_doc and source_doc[field]:
                location_info[field] = source_doc[field]
        
        # Extract coordinates if available
        if 'latitude' in source_doc and 'longitude' in source_doc:
            try:
                lat = float(source_doc['latitude'])
                lng = float(source_doc['longitude'])
                location_info['coordinates'] = {'latitude': lat, 'longitude': lng}
            except (ValueError, TypeError):
                pass
        
        # Extract address information
        if 'address' in source_doc and isinstance(source_doc['address'], dict):
            address = source_doc['address']
            if 'street' in address:
                location_info['street'] = address['street']
            if 'location' in address:
                location_info['address_location'] = address['location']
        
        return location_info
    
    def _get_description_preview(self, source_doc: Dict[str, Any], max_length: int = 200) -> Optional[str]:
        """Get a preview of the property description."""
        description_fields = ['summary', 'description', 'space']
        
        for field in description_fields:
            if field in source_doc and source_doc[field]:
                description = str(source_doc[field]).strip()
                if len(description) > 10:  # Minimum meaningful length
                    if len(description) <= max_length:
                        return description
                    else:
                        # Truncate at word boundary
                        truncated = description[:max_length]
                        last_space = truncated.rfind(' ')
                        if last_space > max_length * 0.7:  # Don't truncate too much
                            return truncated[:last_space] + "..."
                        else:
                            return truncated + "..."
        
        return None
    
    def get_searchable_text(self, source_doc: Dict[str, Any]) -> str:
        """Extract optimized searchable text from Airbnb listing."""
        if not source_doc:
            return ""
        
        text_parts = []
        
        # Process priority fields for search
        for field in self.search_priority_fields:
            if field in source_doc and source_doc[field]:
                field_value = str(source_doc[field]).strip()
                if field_value and len(field_value) > 3:
                    # Clean the text
                    cleaned = re.sub(r'\s+', ' ', field_value)
                    cleaned = re.sub(r'[^\w\s.,!?;:\-\'"()]', ' ', cleaned)
                    cleaned = cleaned.strip()
                    
                    if cleaned:
                        text_parts.append(cleaned)
        
        # Add structured data as text
        structured_fields = {
            'property_type': 'Property type',
            'room_type': 'Room type',
            'accommodates': 'Accommodates',
            'bedrooms': 'Bedrooms',
            'bathrooms': 'Bathrooms',
            'beds': 'Beds',
            'neighbourhood_cleansed': 'Neighborhood',
            'market': 'Market area'
        }
        
        for field, label in structured_fields.items():
            if field in source_doc and source_doc[field]:
                text_parts.append(f"{label}: {source_doc[field]}")
        
        # Amenities (important for search)
        if 'amenities' in source_doc:
            amenities = source_doc['amenities']
            if isinstance(amenities, list):
                amenities_text = ' '.join(amenities[:20])  # Limit to top 20
                text_parts.append(f"Amenities: {amenities_text}")
            elif isinstance(amenities, str):
                try:
                    amenities_list = json.loads(amenities)
                    if isinstance(amenities_list, list):
                        amenities_text = ' '.join(amenities_list[:20])
                        text_parts.append(f"Amenities: {amenities_text}")
                except (json.JSONDecodeError, TypeError):
                    # Use raw string if parsing fails
                    cleaned_amenities = re.sub(r'[\[\]"{}]', '', amenities)
                    text_parts.append(f"Amenities: {cleaned_amenities}")
        
        # Host information
        host_fields = ['host_name', 'host_about']
        for field in host_fields:
            if field in source_doc and source_doc[field]:
                field_value = str(source_doc[field]).strip()
                if field_value and len(field_value) > 3:
                    cleaned = re.sub(r'\s+', ' ', field_value)
                    if field == 'host_name':
                        text_parts.append(f"Host: {cleaned}")
                    else:
                        text_parts.append(f"Host info: {cleaned}")
        
        # Combine all text parts
        full_text = ' '.join(text_parts)
        
        # Final cleaning
        full_text = re.sub(r'\s+', ' ', full_text)
        full_text = full_text.strip()
        
        # Limit total length for performance
        max_length = 5000  # Reasonable limit for search indexing
        if len(full_text) > max_length:
            full_text = full_text[:max_length].rsplit(' ', 1)[0] + "..."
        
        return full_text
    
    def filter_for_display(self, source_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Filter document for optimized display, removing sensitive/irrelevant fields."""
        if not source_doc:
            return {}
        
        filtered_doc = {}
        
        for key, value in source_doc.items():
            # Skip excluded fields
            if key in self.exclude_fields:
                continue
            
            # Skip empty or null values
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            
            # Skip very long text fields that are not essential
            if isinstance(value, str) and len(value) > 1000 and key not in ['description', 'summary', 'space']:
                continue
            
            # Include the field
            filtered_doc[key] = value
        
        # Ensure essential fields are present
        essential_fields = ['_id', 'name', 'property_type', 'room_type', 'price']
        for field in essential_fields:
            if field in source_doc and field not in filtered_doc:
                filtered_doc[field] = source_doc[field]
        
        return filtered_doc
    
    def json_serial(self, obj):
        """JSON serializer for objects not serializable by default."""
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if isinstance(obj, Decimal128):
            return float(obj.to_decimal())
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return str(obj)

# Global instances for easy import
airbnb_optimizer = AirbnbOptimizer()
