"""Improved core RAG system with advanced NLP and better response formatting."""

import json
import logging
import pickle
import re
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from pathlib import Path
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz, process
import faiss
import decimal
import uuid
from bson.decimal128 import Decimal128

# Try to import sklearn for fallback embedding system
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from config import Config
from database import SessionManager, IntentType, SessionContext
from utils import TextProcessor, KeywordExtractor, AirbnbOptimizer
from index_manager import IndexManager
from airbnb_config import AIRBNB_CONFIG

class AdvancedQueryProcessor:
    """Advanced NLP engine for understanding user queries with high accuracy."""
    
    def __init__(self):
        """Initialize the advanced query processor."""
        self.logger = logging.getLogger(__name__)
        
        # Enhanced query patterns for better extraction
        self.price_patterns = [
            r'\$\s*\d+(?:[,.]\d{3})*(?:\.\d{2})?',
            r'\b(?:under|below|less than|cheaper than|maximum|max|up to)\s*\$?\s*\d+',
            r'\b(?:over|above|more than|minimum|min|at least)\s*\$?\s*\d+',
            r'\b(?:between|from)\s*\$?\s*\d+\s*(?:to|and|-|\s)\s*\$?\s*\d+',
            r'\b(?:budget|cheap|affordable|expensive|luxury|premium)\b'
        ]
        
        self.bedroom_patterns = [
            r'(\d+)\s*(?:bed|bedroom|br)(?:room)?s?',
            r'\b(?:studio|one|two|three|four|five|1|2|3|4|5)\s*(?:bed|bedroom)(?:room)?s?',
            r'(\d+)\s*br\b',
            r'\b(studio)\b'
        ]
        
        self.guest_patterns = [
            r'(?:for|accommodate|sleeps?)\s*(\d+)\s*(?:guest|person|people|ppl)?',
            r'(\d+)\s*(?:guest|person|people|ppl)',
            r'party\s*of\s*(\d+)',
            r'group\s*of\s*(\d+)'
        ]
        
        self.location_patterns = [
            r'\bin\s+([A-Za-z][A-Za-z\s,.-]{2,30})(?:\s|$|,)',
            r'\bnear\s+([A-Za-z][A-Za-z\s,.-]{2,30})(?:\s|$|,)',
            r'\baround\s+([A-Za-z][A-Za-z\s,.-]{2,30})(?:\s|$|,)',
            r'\bat\s+([A-Za-z][A-Za-z\s,.-]{2,30})(?:\s|$|,)'
        ]
        
        # Enhanced intent classification
        self.intent_keywords = {
            'search': ['find', 'show', 'looking for', 'search', 'need', 'want', 'get', 'list', 'see'],
            'filter': ['filter', 'narrow down', 'refine', 'specific', 'criteria', 'only', 'just'],
            'compare': ['compare', 'difference', 'versus', 'vs', 'better', 'best', 'which', 'between'],
            'recommend': ['recommend', 'suggest', 'advice', 'opinion', 'think', 'should', 'good'],
            'info': ['tell me', 'what', 'how', 'why', 'where', 'when', 'explain', 'describe', 'about'],
            'count': ['how many', 'number of', 'count', 'total', 'quantity']
        }
        
        # Expanded property type recognition
        self.property_types = {
            'apartment': ['apartment', 'flat', 'unit', 'condo', 'condominium'],
            'house': ['house', 'home', 'villa', 'cottage', 'townhouse'],
            'room': ['room', 'private room', 'bedroom', 'shared room'],
            'studio': ['studio', 'efficiency', 'bachelor'],
            'loft': ['loft', 'penthouse', 'attic'],
            'hotel': ['hotel', 'motel', 'resort', 'lodge'],
            'bnb': ['bed and breakfast', 'bnb', 'b&b', 'guesthouse']
        }
        
        # Comprehensive amenity detection
        self.amenity_patterns = {
            'wifi': r'\b(?:wifi|wi-fi|internet|wireless|broadband)\b',
            'kitchen': r'\b(?:kitchen|kitchenette|cooking|cook|stove|oven|microwave|fridge|refrigerator)\b',
            'parking': r'\b(?:parking|garage|car park|parking space|parking spot)\b',
            'pool': r'\b(?:pool|swimming pool|jacuzzi|hot tub|spa)\b',
            'gym': r'\b(?:gym|fitness|exercise|workout)\b',
            'pet': r'\b(?:pet|dog|cat|animal)\s*(?:friendly|allowed|welcome)?\b',
            'ac': r'\b(?:ac|air\s*conditioning|climate|cooling|heated)\b',
            'tv': r'\b(?:tv|television|cable|netflix|streaming)\b',
            'laundry': r'\b(?:laundry|washer|dryer|washing machine)\b',
            'balcony': r'\b(?:balcony|terrace|patio|deck|outdoor space)\b',
            'garden': r'\b(?:garden|yard|backyard|outdoor area)\b',
            'smoking': r'\b(?:smoking|non-smoking|no smoking)\b',
            'breakfast': r'\b(?:breakfast|morning meal)\b'
        }
        
        # Rating and review patterns
        self.rating_patterns = [
            r'(?:rating|rated|score)\s*(?:of|:)?\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:star|stars)',
            r'(?:above|over|more than)\s*(\d+(?:\.\d+)?)\s*(?:rating|stars?)',
            r'(?:highly|well)\s*(?:rated|reviewed)'
        ]
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse user query with advanced NLP to extract detailed information."""
        if not query or not query.strip():
            return self._empty_parse_result()
        
        query_clean = self._preprocess_query(query)
        
        parsed = {
            'original_query': query,
            'cleaned_query': query_clean,
            'intent': self._classify_intent(query_clean),
            'filters': self._extract_all_filters(query_clean),
            'keywords': self._extract_semantic_keywords(query_clean),
            'entities': self._extract_entities(query_clean),
            'search_terms': self._generate_search_terms(query_clean),
            'confidence': self._calculate_confidence(query_clean)
        }
        
        # Enhance with context understanding
        parsed = self._enhance_with_context(parsed)
        
        self.logger.debug(f"Parsed query: {json.dumps(parsed, indent=2)}")
        return parsed
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better parsing."""
        # Convert to lowercase for processing
        query = query.lower().strip()
        
        # Normalize common variations
        query = re.sub(r'\b(?:w/|with)\b', 'with', query)
        query = re.sub(r'\b(?:w/o|without)\b', 'without', query)
        query = re.sub(r'\b(?:apt|appt)\b', 'apartment', query)
        query = re.sub(r'\b(?:br|bdrm)\b', 'bedroom', query)
        query = re.sub(r'\b(?:ba|bath)\b', 'bathroom', query)
        query = re.sub(r'\$\s+', '$', query)  # Remove spaces after dollar signs
        
        # Handle number words
        word_to_num = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10'
        }
        
        for word, num in word_to_num.items():
            query = re.sub(r'\b' + word + r'\b', num, query)
        
        return query
    
    def _classify_intent(self, query: str) -> str:
        """Classify user intent with improved accuracy."""
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    # Give higher weight to exact matches
                    if keyword == query.strip():
                        score += 3
                    elif query.startswith(keyword) or query.endswith(keyword):
                        score += 2
                    else:
                        score += 1
            
            if score > 0:
                intent_scores[intent] = score
        
        # Default to search if no clear intent
        if not intent_scores:
            return 'search'
        
        return max(intent_scores.items(), key=lambda x: x[1])[0]
    
    def _extract_all_filters(self, query: str) -> Dict[str, Any]:
        """Extract all possible filters from query."""
        filters = {}
        
        # Price filters with improved extraction
        price_filter = self._extract_price_filter(query)
        if price_filter:
            filters['price'] = price_filter
        
        # Bedroom filters
        bedroom_filter = self._extract_bedroom_filter(query)
        if bedroom_filter is not None:
            filters['bedrooms'] = bedroom_filter
        
        # Guest capacity
        guest_filter = self._extract_guest_filter(query)
        if guest_filter:
            filters['accommodates'] = guest_filter
        
        # Location with multiple detection methods
        location_filter = self._extract_location_filter(query)
        if location_filter:
            filters['location'] = location_filter
        
        # Amenities with comprehensive detection
        amenity_filters = self._extract_amenity_filters(query)
        if amenity_filters:
            filters['amenities'] = amenity_filters
        
        # Property type
        property_filter = self._extract_property_type_filter(query)
        if property_filter:
            filters['property_type'] = property_filter
        
        # Rating filters
        rating_filter = self._extract_rating_filter(query)
        if rating_filter:
            filters['rating'] = rating_filter
        
        return filters
    
    def _extract_price_filter(self, query: str) -> Optional[Dict[str, float]]:
        """Extract price constraints with improved accuracy."""
        # Look for explicit price mentions
        for pattern in self.price_patterns:
            matches = list(re.finditer(pattern, query, re.IGNORECASE))
            
            for match in matches:
                price_text = match.group().lower()
                
                # Extract all numbers from the match
                numbers = re.findall(r'\d+(?:[,.]\d{3})*(?:\.\d{2})?', price_text)
                
                if numbers:
                    # Clean and convert numbers
                    cleaned_numbers = []
                    for num_str in numbers:
                        # Remove commas and convert
                        clean_num = num_str.replace(',', '')
                        try:
                            cleaned_numbers.append(float(clean_num))
                        except ValueError:
                            continue
                    
                    if not cleaned_numbers:
                        continue
                    
                    price = cleaned_numbers[0]
                    
                    # Determine filter type based on context
                    if any(word in price_text for word in ['under', 'below', 'less than', 'cheaper', 'max', 'up to']):
                        return {'max': price}
                    elif any(word in price_text for word in ['over', 'above', 'more than', 'min', 'at least']):
                        return {'min': price}
                    elif any(word in query for word in ['between', 'from']) and len(cleaned_numbers) >= 2:
                        return {'min': min(cleaned_numbers[0], cleaned_numbers[1]), 
                               'max': max(cleaned_numbers[0], cleaned_numbers[1])}
                    else:
                        # Single price - treat as target or max depending on context
                        return {'target': price}
        
        # Budget keywords without explicit prices
        if re.search(r'\b(?:cheap|budget|affordable|inexpensive)\b', query):
            return {'max': 100}
        elif re.search(r'\b(?:luxury|premium|expensive|high.?end|upscale)\b', query):
            return {'min': 200}
        
        return None
    
    def _extract_bedroom_filter(self, query: str) -> Optional[int]:
        """Extract bedroom count with improved patterns."""
        for pattern in self.bedroom_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                bedroom_text = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group()
                
                # Handle studio separately
                if 'studio' in bedroom_text.lower():
                    return 0
                
                # Extract numeric value
                if bedroom_text.isdigit():
                    return int(bedroom_text)
                
                # Convert word to number
                word_to_num = {
                    'studio': 0, 'one': 1, 'two': 2, 'three': 3, 
                    'four': 4, 'five': 5, 'six': 6
                }
                
                return word_to_num.get(bedroom_text.lower())
        
        return None
    
    def _extract_guest_filter(self, query: str) -> Optional[int]:
        """Extract guest capacity with better patterns."""
        for pattern in self.guest_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                guest_count = match.group(1)
                if guest_count.isdigit():
                    return int(guest_count)
        
        return None
    
    def _extract_location_filter(self, query: str) -> Optional[str]:
        """Extract location with improved detection."""
        # Try pattern-based extraction first
        for pattern in self.location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Clean up the location
                location = re.sub(r'[,.:;]$', '', location)
                if len(location) > 2 and location.lower() not in ['the', 'and', 'or', 'but']:
                    return location.title()
        
        # Look for capitalized words that might be locations
        words = query.split()
        potential_locations = []
        
        for i, word in enumerate(words):
            if len(word) > 2 and (word[0].isupper() or i > 0):  # Capitalized or not at start
                # Check if preceded by location indicator
                if i > 0 and words[i-1].lower() in ['in', 'near', 'at', 'around', 'close to']:
                    potential_locations.append(word.title())
        
        if potential_locations:
            return ' '.join(potential_locations[:2])  # Limit to 2 words
        
        return None
    
    def _extract_amenity_filters(self, query: str) -> List[str]:
        """Extract amenities with comprehensive detection."""
        found_amenities = []
        
        for amenity, pattern in self.amenity_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                found_amenities.append(amenity)
        
        return found_amenities
    
    def _extract_property_type_filter(self, query: str) -> Optional[str]:
        """Extract property type with expanded recognition."""
        for prop_type, variations in self.property_types.items():
            for variation in variations:
                if re.search(r'\b' + re.escape(variation) + r'\b', query, re.IGNORECASE):
                    return prop_type
        
        return None
    
    def _extract_rating_filter(self, query: str) -> Optional[Dict[str, float]]:
        """Extract rating constraints."""
        for pattern in self.rating_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if match.lastindex and match.lastindex >= 1:
                    rating_value = float(match.group(1))
                    if 'above' in match.group() or 'over' in match.group() or 'more than' in match.group():
                        return {'min': rating_value}
                    else:
                        return {'target': rating_value}
        
        # Check for qualitative rating terms
        if re.search(r'\b(?:highly|well|top|best|excellent)\s*(?:rated|reviewed|rated)\b', query):
            return {'min': 4.5}
        elif re.search(r'\b(?:good|decent|ok|okay)\s*(?:rating|reviews?)\b', query):
            return {'min': 3.5}
        
        return None
    
    def _extract_semantic_keywords(self, query: str) -> List[str]:
        """Extract keywords with semantic understanding."""
        # Enhanced stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 
            'them', 'this', 'that', 'these', 'those', 'can', 'may', 'must'
        }
        
        # Extract words and clean them
        words = re.findall(r'\b\w+\b', query)
        keywords = []
        
        for word in words:
            word_clean = word.lower()
            if (len(word_clean) > 2 and 
                word_clean not in stop_words and 
                not word_clean.isdigit()):
                keywords.append(word_clean)
        
        # Add domain-specific important terms
        domain_terms = {
            'location': ['downtown', 'central', 'suburb', 'city center', 'historic'],
            'quality': ['clean', 'modern', 'updated', 'renovated', 'spacious', 'cozy'],
            'features': ['view', 'quiet', 'bright', 'private', 'shared']
        }
        
        for category, terms in domain_terms.items():
            for term in terms:
                if term in query and term not in keywords:
                    keywords.append(term)
        
        return keywords
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities with improved detection."""
        entities = {
            'numbers': [],
            'money': [],
            'locations': [],
            'dates': [],
            'amenities': []
        }
        
        # Numbers (excluding those that are part of prices)
        entities['numbers'] = re.findall(r'\b(?<!\$)\d+(?!\.\d{2})\b', query)
        
        # Money amounts
        entities['money'] = re.findall(r'\$\d+(?:[,.]\d{3})*(?:\.\d{2})?', query)
        
        # Potential locations (capitalized words)
        words = query.split()
        for i, word in enumerate(words):
            if (len(word) > 2 and 
                word[0].isupper() and 
                not word.lower() in self.intent_keywords.get('search', [])):
                entities['locations'].append(word)
        
        # Amenities from our patterns
        entities['amenities'] = self._extract_amenity_filters(query)
        
        return entities
    
    def _generate_search_terms(self, query: str) -> List[str]:
        """Generate optimized search terms for better matching."""
        search_terms = [query]  # Original query
        
        # Add keyword-based terms
        keywords = self._extract_semantic_keywords(query)
        if keywords:
            # Different combinations of keywords
            if len(keywords) >= 2:
                search_terms.append(' '.join(keywords[:3]))
                search_terms.append(' '.join(keywords[:2]))
            
            # Individual important keywords
            search_terms.extend(keywords[:5])
        
        # Add amenity-focused searches
        amenities = self._extract_amenity_filters(query)
        if amenities:
            search_terms.extend(amenities)
        
        # Add property type if detected
        prop_type = self._extract_property_type_filter(query)
        if prop_type:
            search_terms.append(prop_type)
        
        # Add location-based terms
        location = self._extract_location_filter(query)
        if location:
            search_terms.append(location)
        
        # Remove duplicates and empty terms
        search_terms = list(set([term for term in search_terms if term and len(term) > 1]))
        
        return search_terms
    
    def _calculate_confidence(self, query: str) -> Dict[str, float]:
        """Calculate confidence scores for different aspects."""
        confidence = {
            'intent': 0.5,
            'filters': 0.3,
            'location': 0.3,
            'overall': 0.4
        }
        
        # Intent confidence
        intent_matches = 0
        for keywords in self.intent_keywords.values():
            intent_matches += sum(1 for keyword in keywords if keyword in query)
        
        confidence['intent'] = min(0.95, 0.3 + (intent_matches * 0.15))
        
        # Filter confidence based on detected filters
        filter_indicators = [
            r'\$\d+',  # Price
            r'\d+\s*bed',  # Bedrooms
            r'\d+\s*guest',  # Guests
            r'\bin\s+[A-Z]',  # Location
        ]
        
        filter_score = sum(1 for pattern in filter_indicators if re.search(pattern, query, re.IGNORECASE))
        confidence['filters'] = min(0.9, 0.3 + (filter_score * 0.2))
        
        # Location confidence
        if self._extract_location_filter(query):
            confidence['location'] = 0.8
        
        # Overall confidence
        confidence['overall'] = (
            confidence['intent'] * 0.3 + 
            confidence['filters'] * 0.4 + 
            confidence['location'] * 0.3
        )
        
        return confidence
    
    def _enhance_with_context(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance parsed query with contextual understanding."""
        # Add query complexity score
        complexity_indicators = [
            len(parsed['filters']),
            len(parsed['keywords']),
            1 if parsed['intent'] != 'search' else 0,
            len([e for entities in parsed['entities'].values() for e in entities])
        ]
        
        parsed['complexity_score'] = sum(complexity_indicators) / 10.0
        
        # Add search strategy suggestion
        if parsed['filters']:
            parsed['search_strategy'] = 'filtered_search'
        elif len(parsed['keywords']) > 3:
            parsed['search_strategy'] = 'keyword_rich_search'
        else:
            parsed['search_strategy'] = 'general_search'
        
        return parsed
    
    def _empty_parse_result(self) -> Dict[str, Any]:
        """Return empty parse result for invalid queries."""
        return {
            'original_query': '',
            'cleaned_query': '',
            'intent': 'search',
            'filters': {},
            'keywords': [],
            'entities': {'numbers': [], 'money': [], 'locations': [], 'dates': [], 'amenities': []},
            'search_terms': [],
            'confidence': {'intent': 0.1, 'filters': 0.1, 'location': 0.1, 'overall': 0.1},
            'complexity_score': 0.0,
            'search_strategy': 'general_search'
        }

class ImprovedResponseGenerator:
    """Generate structured responses with summaries and source data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_structured_response(self, 
                                   query: str,
                                   search_results: List[Dict[str, Any]], 
                                   parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured response with summary and source data."""
        if not search_results:
            return self._generate_no_results_response(query)
        
        # Generate summary based on query intent and results
        summary = self._generate_intelligent_summary(query, search_results, parsed_query)
        
        # Format source data with key fields
        sources = self._format_source_data(search_results, parsed_query)
        
        # Generate statistics
        stats = self._generate_result_statistics(search_results, parsed_query)
        
        return {
            'summary': summary,
            'sources': sources,
            'statistics': stats,
            'query_analysis': {
                'intent': parsed_query.get('intent', 'search'),
                'filters_applied': list(parsed_query.get('filters', {}).keys()),
                'results_count': len(search_results)
            }
        }
    
    def _generate_intelligent_summary(self, 
                                    query: str, 
                                    results: List[Dict[str, Any]], 
                                    parsed_query: Dict[str, Any]) -> str:
        """Generate an intelligent summary based on query intent and results."""
        intent = parsed_query.get('intent', 'search')
        filters = parsed_query.get('filters', {})
        result_count = len(results)
        
        # Extract key information from results
        key_info = self._extract_key_result_info(results)
        
        # Generate intent-specific summary
        if intent == 'count':
            return self._generate_count_summary(query, result_count, key_info)
        elif intent == 'compare':
            return self._generate_comparison_summary(query, results, key_info)
        elif intent == 'recommend':
            return self._generate_recommendation_summary(query, results, key_info)
        elif intent == 'info':
            return self._generate_info_summary(query, results, key_info)
        else:
            return self._generate_search_summary(query, results, key_info, filters)
    
    def _extract_key_result_info(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key information from search results."""
        if not results:
            return {}
        
        prices = []
        bedrooms = []
        ratings = []
        property_types = []
        locations = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            if 'price' in metadata and metadata['price'] > 0:
                prices.append(metadata['price'])
            
            if 'bedrooms' in metadata and metadata['bedrooms'] >= 0:
                bedrooms.append(metadata['bedrooms'])
            
            if 'rating' in metadata and metadata['rating'] > 0:
                ratings.append(metadata['rating'])
            
            if 'property_type' in metadata:
                property_types.append(metadata['property_type'])
            
            if 'neighbourhood' in metadata:
                locations.append(metadata['neighbourhood'])
        
        info = {
            'total_results': len(results),
            'price_range': {'min': min(prices), 'max': max(prices), 'avg': sum(prices)/len(prices)} if prices else None,
            'bedroom_range': {'min': min(bedrooms), 'max': max(bedrooms)} if bedrooms else None,
            'avg_rating': sum(ratings) / len(ratings) if ratings else None,
            'common_property_types': self._get_most_common(property_types),
            'common_locations': self._get_most_common(locations)
        }
        
        return info
    
    def _get_most_common(self, items: List[str], top_n: int = 3) -> List[Tuple[str, int]]:
        """Get most common items from a list."""
        if not items:
            return []
        
        counts = {}
        for item in items:
            if item:
                counts[item] = counts.get(item, 0) + 1
        
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def _generate_count_summary(self, query: str, count: int, key_info: Dict[str, Any]) -> str:
        """Generate summary for count queries."""
        if count == 0:
            return f"I found no properties matching your query: '{query}'"
        elif count == 1:
            return f"I found 1 property matching your query: '{query}'"
        else:
            summary = f"I found {count} properties matching your query: '{query}'"
            
            if key_info.get('price_range'):
                price_range = key_info['price_range']
                summary += f" with prices ranging from ${price_range['min']:.0f} to ${price_range['max']:.0f}."
            
            return summary
    
    def _generate_search_summary(self, 
                               query: str, 
                               results: List[Dict[str, Any]], 
                               key_info: Dict[str, Any], 
                               filters: Dict[str, Any]) -> str:
        """Generate summary for search queries."""
        count = len(results)
        
        if count == 0:
            return f"I couldn't find any properties matching your search: '{query}'. Try broadening your criteria."
        
        # Start with basic count
        summary = f"I found {count} propert{'ies' if count != 1 else 'y'} for your search: '{query}'."
        
        # Add key statistics
        details = []
        
        if key_info.get('price_range'):
            price_range = key_info['price_range']
            details.append(f"prices range from ${price_range['min']:.0f} to ${price_range['max']:.0f} (avg: ${price_range['avg']:.0f})")
        
        if key_info.get('bedroom_range'):
            bedroom_range = key_info['bedroom_range']
            if bedroom_range['min'] == bedroom_range['max']:
                details.append(f"all have {bedroom_range['min']} bedroom{'s' if bedroom_range['min'] != 1 else ''}")
            else:
                details.append(f"bedrooms range from {bedroom_range['min']} to {bedroom_range['max']}")
        
        if key_info.get('avg_rating'):
            details.append(f"average rating: {key_info['avg_rating']:.1f}/5")
        
        if key_info.get('common_property_types'):
            top_type = key_info['common_property_types'][0]
            if top_type[1] > count * 0.5:  # More than half
                details.append(f"mostly {top_type[0]}s")
        
        if key_info.get('common_locations') and len(key_info['common_locations']) > 0:
            top_location = key_info['common_locations'][0]
            if top_location[1] > count * 0.3:  # More than 30%
                details.append(f"many in {top_location[0]}")
        
        if details:
            summary += " " + ", ".join(details[:3]) + "."
        
        # Add filter acknowledgment
        if filters:
            filter_desc = []
            if 'price' in filters:
                filter_desc.append("price constraints")
            if 'bedrooms' in filters:
                filter_desc.append(f"{filters['bedrooms']} bedroom{'s' if filters['bedrooms'] != 1 else ''}")
            if 'amenities' in filters:
                filter_desc.append(f"amenities: {', '.join(filters['amenities'])}")
            
            if filter_desc:
                summary += f" These results match your {', '.join(filter_desc)}."
        
        return summary
    
    def _generate_comparison_summary(self, query: str, results: List[Dict[str, Any]], key_info: Dict[str, Any]) -> str:
        """Generate summary for comparison queries."""
        if len(results) < 2:
            return f"I need at least 2 properties to compare. Found {len(results)} result(s)."
        
        # Compare top 2-3 results
        top_results = results[:3]
        summary = f"Comparing the top {len(top_results)} properties from your search:"
        
        for i, result in enumerate(top_results, 1):
            metadata = result.get('metadata', {})
            name = metadata.get('name', f'Property {i}')
            price = metadata.get('price', 0)
            bedrooms = metadata.get('bedrooms', 'Unknown')
            rating = metadata.get('rating', 'Unrated')
            
            summary += f" {i}. {name} - ${price:.0f}/night, {bedrooms} bedroom{'s' if bedrooms != 1 else ''}, rating: {rating:.1f}/5."
        
        return summary
    
    def _generate_recommendation_summary(self, query: str, results: List[Dict[str, Any]], key_info: Dict[str, Any]) -> str:
        """Generate summary for recommendation queries."""
        if not results:
            return "I don't have enough information to make a recommendation based on your criteria."
        
        # Recommend top result
        top_result = results[0]
        metadata = top_result.get('metadata', {})
        name = metadata.get('name', 'this property')
        
        summary = f"I recommend {name} based on your search."
        
        # Add reasoning
        reasons = []
        if metadata.get('rating', 0) >= 4.5:
            reasons.append("high rating")
        if metadata.get('price', float('inf')) <= 150:
            reasons.append("good value")
        
        if reasons:
            summary += f" This property stands out due to its {' and '.join(reasons)}."
        
        return summary
    
    def _generate_info_summary(self, query: str, results: List[Dict[str, Any]], key_info: Dict[str, Any]) -> str:
        """Generate summary for information queries."""
        if not results:
            return "I don't have specific information about that. Try refining your search."
        
        # Provide informative summary about the results
        summary = f"Based on your query about '{query}', here's what I found:"
        
        if key_info.get('total_results'):
            summary += f" There are {key_info['total_results']} relevant properties."
        
        if key_info.get('common_property_types'):
            summary += f" The most common property types are {', '.join([pt[0] for pt in key_info['common_property_types'][:2]])}."
        
        return summary
    
    def _format_source_data(self, results: List[Dict[str, Any]], parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format source data with key fields for each result."""
        formatted_sources = []
        
        for i, result in enumerate(results[:10]):  # Limit to top 10
            metadata = result.get('metadata', {})
            
            source = {
                'rank': i + 1,
                'id': result.get('id', f'result_{i+1}'),
                'name': metadata.get('name', 'Unnamed Property'),
                'property_type': metadata.get('property_type', 'Unknown'),
                'price_per_night': metadata.get('price', 0),
                'bedrooms': metadata.get('bedrooms', 'Unknown'),
                'accommodates': metadata.get('accommodates', 'Unknown'),
                'rating': metadata.get('rating', 'Unrated'),
                'neighbourhood': metadata.get('neighbourhood', 'Unknown'),
                'listing_url': metadata.get('listing_url', ''),
                'relevance_score': result.get('score', 0)
            }
            
            # Add key amenities if available
            amenities_mentioned = parsed_query.get('filters', {}).get('amenities', [])
            if amenities_mentioned:
                source['requested_amenities'] = amenities_mentioned
            
            formatted_sources.append(source)
        
        return formatted_sources
    
    def _generate_result_statistics(self, results: List[Dict[str, Any]], parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistics about the search results."""
        if not results:
            return {'total_results': 0}
        
        stats = {
            'total_results': len(results),
            'query_intent': parsed_query.get('intent', 'search'),
            'filters_applied': len(parsed_query.get('filters', {})),
            'search_confidence': parsed_query.get('confidence', {}).get('overall', 0.5)
        }
        
        # Add price statistics
        prices = [r.get('metadata', {}).get('price', 0) for r in results if r.get('metadata', {}).get('price', 0) > 0]
        if prices:
            stats['price_statistics'] = {
                'min_price': min(prices),
                'max_price': max(prices),
                'average_price': sum(prices) / len(prices),
                'price_range': max(prices) - min(prices)
            }
        
        # Add rating statistics
        ratings = [r.get('metadata', {}).get('rating', 0) for r in results if r.get('metadata', {}).get('rating', 0) > 0]
        if ratings:
            stats['rating_statistics'] = {
                'average_rating': sum(ratings) / len(ratings),
                'min_rating': min(ratings),
                'max_rating': max(ratings)
            }
        
        return stats
    
    def _generate_no_results_response(self, query: str) -> Dict[str, Any]:
        """Generate response when no results are found."""
        return {
            'summary': f"I couldn't find any properties matching your search: '{query}'. You might want to try broadening your criteria or checking for typos.",
            'sources': [],
            'statistics': {'total_results': 0},
            'query_analysis': {
                'intent': 'search',
                'filters_applied': [],
                'results_count': 0
            }
        }

class JSONRAGSystem:
    """Main RAG system with improved NLP and response generation."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the improved RAG system."""
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.session_manager = SessionManager()
        self.query_processor = AdvancedQueryProcessor()
        self.response_generator = ImprovedResponseGenerator()
        
        # Search components
        self.model = None
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        
        # Supporting components
        self.text_processor = TextProcessor()
        self.keyword_extractor = KeywordExtractor()
        self.airbnb_optimizer = AirbnbOptimizer()
        
        self.logger.info("RAG system initialized with advanced NLP capabilities")
    
    def initialize_system(self) -> bool:
        """Initialize the complete RAG system."""
        try:
            self.logger.info("Initializing RAG system components...")
            
            # Load search components
            success = self._load_search_components()
            if not success:
                self.logger.error("Failed to load search components")
                return False
            
            self.logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def _load_search_components(self) -> bool:
        """Load search components (embeddings, FAISS index, documents)."""
        try:
            # Load from index_manager
            index_manager = IndexManager()
            
            # Load existing indexes or create new ones
            try:
                self.faiss_index, self.processed_docs = index_manager.load_indexes()
                logger.info("Successfully loaded existing indexes")
            except FileNotFoundError:
                logger.info("No existing indexes found, creating new ones...")
                index_manager.create_complete_index()
                self.faiss_index, self.processed_docs = index_manager.load_indexes()
            
            # Load the sentence transformer model
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
            
            # Verify that indexes were loaded successfully
            if not self.processed_docs:
                logger.error("No processed documents available after loading/creating indexes")
                return False
            # Documents are already loaded as self.processed_docs
            self.documents = self.processed_docs
            # FAISS index is already loaded as self.faiss_index
            
            self.logger.info(f"Loaded {len(self.documents)} documents and search index")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading search components: {e}")
            return False
    
    def process_query(self, 
                     query: str, 
                     session_id: Optional[str] = None,
                     max_results: int = 10) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Process a user query with advanced NLP and generate structured response."""
        try:
            # Parse query with advanced NLP
            parsed_query = self.query_processor.parse_query(query)
            
            # Create or get session
            if not session_id:
                session_id = str(uuid.uuid4())
                self.session_manager.create_session(session_id)
            
            # Perform search with parsed query
            search_results = self._perform_enhanced_search(parsed_query, max_results)
            
            # Generate structured response
            structured_response = self.response_generator.generate_structured_response(
                query, search_results, parsed_query
            )
            
            # Format final response
            response_text = self._format_final_response(structured_response)
            
            # Update session context
            session_context = self.session_manager.get_session(session_id)
            if session_context:
                # Create conversation turn
                from database import ConversationTurn, IntentType
                conversation_turn = ConversationTurn(
                    timestamp=datetime.now().isoformat(),
                    user_input=query,
                    intent=IntentType.SEARCH,
                    entities=list(parsed_query.get('entities', {}).get('locations', [])),
                    search_query=query,
                    retrieved_docs=[{'id': r.get('id', ''), 'score': r.get('score', 0)} for r in search_results[:5]],
                    response=response_text,
                    context_used=[]
                )
                
                # Add to conversation history
                session_context.conversation_history.append(conversation_turn)
                session_context.last_activity = datetime.now().isoformat()
                
                # Update session
                self.session_manager.update_session_context(
                    session_context, query, response_text, {'results_found': len(search_results)}
                )
            
            # Get conversation history
            history = session_context.conversation_history if session_context else []
            
            return response_text, session_id, history
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            error_response = f"I encountered an error while processing your query: '{query}'. Please try rephrasing your request."
            return error_response, session_id or '', []
    
    def _perform_enhanced_search(self, parsed_query: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Perform enhanced search using parsed query information."""
        if not self.faiss_index or not self.model:
            return []
        
        try:
            # Use search strategy from parsed query
            strategy = parsed_query.get('search_strategy', 'general_search')
            
            if strategy == 'filtered_search':
                return self._filtered_search(parsed_query, max_results)
            elif strategy == 'keyword_rich_search':
                return self._keyword_rich_search(parsed_query, max_results)
            else:
                return self._general_search(parsed_query, max_results)
                
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []
    
    def _filtered_search(self, parsed_query: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Perform filtered search with specific criteria."""
        filters = parsed_query.get('filters', {})
        
        # Start with semantic search
        search_terms = parsed_query.get('search_terms', [parsed_query.get('original_query', '')])
        primary_term = search_terms[0] if search_terms else parsed_query.get('original_query', '')
        
        # Generate query embedding
        query_embedding = self.model.encode([primary_term])
        
        # Search FAISS index
        search_k = max_results * 3  # Get more results for filtering
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)
        
        # Apply filters
        filtered_results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            metadata = doc.get('metadata', {})
            
            # Check filters
            if self._matches_filters(metadata, filters):
                result = {
                    'id': doc.get('id', ''),
                    'text': doc.get('text', ''),
                    'metadata': metadata,
                    'score': float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity
                }
                filtered_results.append(result)
                
                if len(filtered_results) >= max_results:
                    break
        
        return filtered_results
    
    def _keyword_rich_search(self, parsed_query: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Perform keyword-rich search using multiple search terms."""
        search_terms = parsed_query.get('search_terms', [])
        
        # Combine results from multiple search terms
        all_results = {}
        
        for term in search_terms[:3]:  # Use top 3 search terms
            query_embedding = self.model.encode([term])
            distances, indices = self.faiss_index.search(query_embedding.astype('float32'), max_results)
            
            for i, idx in enumerate(indices[0]):
                if idx >= len(self.documents):
                    continue
                    
                doc_id = self.documents[idx].get('id', f'doc_{idx}')
                score = float(1.0 / (1.0 + distances[0][i]))
                
                if doc_id in all_results:
                    all_results[doc_id]['score'] = max(all_results[doc_id]['score'], score)
                else:
                    all_results[doc_id] = {
                        'id': doc_id,
                        'text': self.documents[idx].get('text', ''),
                        'metadata': self.documents[idx].get('metadata', {}),
                        'score': score
                    }
        
        # Sort by score and return top results
        sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
        return sorted_results[:max_results]
    
    def _general_search(self, parsed_query: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Perform general semantic search."""
        query = parsed_query.get('original_query', '')
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), max_results)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            result = {
                'id': doc.get('id', ''),
                'text': doc.get('text', ''),
                'metadata': doc.get('metadata', {}),
                'score': float(1.0 / (1.0 + distances[0][i]))
            }
            results.append(result)
        
        return results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a document matches the specified filters."""
        for filter_key, filter_value in filters.items():
            if filter_key == 'price' and isinstance(filter_value, dict):
                price = metadata.get('price', 0)
                if 'min' in filter_value and price < filter_value['min']:
                    return False
                if 'max' in filter_value and price > filter_value['max']:
                    return False
                if 'target' in filter_value and abs(price - filter_value['target']) > filter_value['target'] * 0.2:
                    return False
            
            elif filter_key == 'bedrooms':
                if metadata.get('bedrooms') != filter_value:
                    return False
            
            elif filter_key == 'accommodates':
                if metadata.get('accommodates', 0) < filter_value:
                    return False
            
            elif filter_key == 'property_type':
                if metadata.get('property_type', '').lower() != filter_value.lower():
                    return False
            
            elif filter_key == 'location':
                location_fields = ['neighbourhood', 'city', 'country']
                location_match = any(
                    filter_value.lower() in str(metadata.get(field, '')).lower()
                    for field in location_fields
                )
                if not location_match:
                    return False
            
            elif filter_key == 'amenities':
                # This would require checking amenities in the full document
                # For now, we'll skip amenity filtering in this simplified version
                pass
            
            elif filter_key == 'rating' and isinstance(filter_value, dict):
                rating = metadata.get('rating', 0)
                if 'min' in filter_value and rating < filter_value['min']:
                    return False
                if 'target' in filter_value and abs(rating - filter_value['target']) > 0.5:
                    return False
        
        return True
    
    def _format_final_response(self, structured_response: Dict[str, Any]) -> str:
        """Format the structured response into a readable text response."""
        summary = structured_response.get('summary', '')
        sources = structured_response.get('sources', [])
        stats = structured_response.get('statistics', {})
        
        # Start with the summary
        response_parts = [summary]
        
        # Add top results if available
        if sources:
            response_parts.append("\nTop Results:")
            
            for i, source in enumerate(sources[:5], 1):  # Show top 5
                name = source.get('name', f'Property {i}')
                price = source.get('price_per_night', 0)
                bedrooms = source.get('bedrooms', 'Unknown')
                rating = source.get('rating', 'Unrated')
                neighbourhood = source.get('neighbourhood', '')
                
                result_line = f"{i}. {name}"
                
                details = []
                if price > 0:
                    details.append(f"${price:.0f}/night")
                if bedrooms != 'Unknown':
                    details.append(f"{bedrooms} bed{'s' if bedrooms != 1 else ''}")
                if rating != 'Unrated' and rating > 0:
                    details.append(f"{rating:.1f}★")
                if neighbourhood:
                    details.append(f"in {neighbourhood}")
                
                if details:
                    result_line += f" - {', '.join(details)}"
                
                response_parts.append(result_line)
        
        return "\n".join(response_parts)

def main():
    """Test the improved RAG system."""
    try:
        system = JSONRAGSystem()
        if not system.initialize_system():
            print("Failed to initialize system")
            return
        
        # Test queries
        test_queries = [
            "Find 2-bedroom apartments under $150 with WiFi",
            "Show me highly rated places in downtown",
            "I need a place for 4 people with parking",
            "What luxury properties do you have?",
            "How many studio apartments are available?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("=" * 50)
            
            response, session_id, history = system.process_query(query)
            print(response)
            print("\n")
            
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
