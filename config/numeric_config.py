#!/usr/bin/env python3
"""
Numeric Configuration for JSON RAG System
Handles numeric keyword mappings and constraints for property search
"""

"""Configuration file with numeric processing settings and mappings."""

class NumericConfig:
    """Centralized configuration for numeric query processing"""

    # Numeric keyword mappings
    NUMERIC_KEYWORDS = {
        # Bedroom mappings
        'bedrooms': {
            'keywords': [
                'bedroom', 'bedrooms', 'bed', 'beds', 'br', 'bdr', 'bdrs',
                'sleeping room', 'sleeping rooms', 'sleep', 'sleeps'
            ],
            'patterns': [
                r'\b(\d+)\s*(?:bed|bedroom|bedrooms|br|bdr)s?\b',
                r'\b(?:bed|bedroom|bedrooms|br|bdr)s?\s*(\d+)\b',
                r'\b(\d+)\s*(?:-|\s)?(?:bed|bedroom|br)\b',
                r'\bstudio\b'  # Special case for studio = 0 bedrooms
            ],
            'field_names': ['bedrooms', 'beds', 'bedroom_count', 'bed_count'],
            'studio_value': 0,
            'default_range': [1, 10]
        },
        
        # Bathroom mappings
        'bathrooms': {
            'keywords': [
                'bathroom', 'bathrooms', 'bath', 'baths', 'ba', 'full bath',
                'half bath', 'powder room', 'washroom', 'restroom'
            ],
            'patterns': [
                r'\b(\d+(?:\.\d+)?)\s*(?:bath|bathroom|bathrooms|ba)s?\b',
                r'\b(?:bath|bathroom|bathrooms|ba)s?\s*(\d+(?:\.\d+)?)\b',
                r'\b(\d+(?:\.\d+)?)\s*(?:-|\s)?(?:bath|bathroom|ba)\b'
            ],
            'field_names': ['bathrooms', 'baths', 'bathroom_count', 'bath_count'],
            'default_range': [1, 8]
        },
        
        # Guest/Accommodation mappings
        'guests': {
            'keywords': [
                'guest', 'guests', 'people', 'person', 'occupant', 'occupants',
                'accommodate', 'accommodates', 'sleeps', 'capacity', 'max guest',
                'maximum guest', 'guest limit', 'occupancy'
            ],
            'patterns': [
                r'\b(?:accommodate|sleeps|guest|guests|people|person)s?\s*(\d+)\b',
                r'\b(\d+)\s*(?:guest|guests|people|person|occupant)s?\b',
                r'\bfor\s*(\d+)\s*(?:guest|guests|people|person)s?\b',
                r'\bup\s*to\s*(\d+)\s*(?:guest|guests|people)s?\b'
            ],
            'field_names': ['guests', 'accommodates', 'guest_capacity', 'max_guests', 'occupancy'],
            'default_range': [1, 16]
        },
        
        # Price mappings
        'price': {
            'keywords': [
                'price', 'cost', 'rate', 'fee', 'charge', 'amount', 'budget',
                'expensive', 'cheap', 'affordable', 'under', 'below', 'above',
                'maximum', 'minimum', 'max', 'min', 'dollar', 'usd'
            ],
            'patterns': [
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\b',
                r'\b(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollar|usd|$)s?\b',
                r'\bunder\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bbelow\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\babove\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bover\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bmax\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bmaximum\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bmin\s*\$?\s*(\d+(?:,\d{3})*)\b',
                r'\bminimum\s*\$?\s*(\d+(?:,\d{3})*)\b'
            ],
            'field_names': ['price', 'cost', 'rate', 'nightly_rate', 'daily_rate'],
            'currency_symbols': ['$', 'USD', 'usd'],
            'default_range': [10, 10000]
        }
    }

    # Range operators for numeric constraints
    RANGE_OPERATORS = {
        'exact': {
            'patterns': [
                r'\bexactly\s*(\d+)\b',
                r'^(\d+)$',
                r'\b(\d+)\s*(?:bed|bedroom|bath|bathroom|guest)s?\s*only\b'
            ],
            'operator': '='
        },
        
        'minimum': {
            'patterns': [
                r'\bat\s*least\s*(\d+)\b',
                r'\bminimum\s*(?:of\s*)?(\d+)\b',
                r'\bmin\s*(\d+)\b',
                r'\b(\d+)\s*(?:or\s*)?(?:more|plus|above)\b',
                r'\b(?:more\s*than|above|over)\s*(\d+)\b'
            ],
            'operator': '>='
        },
        
        'maximum': {
            'patterns': [
                r'\bat\s*most\s*(\d+)\b',
                r'\bmaximum\s*(?:of\s*)?(\d+)\b',
                r'\bmax\s*(\d+)\b',
                r'\b(\d+)\s*(?:or\s*)?(?:less|fewer|below|under)\b',
                r'\b(?:less\s*than|below|under)\s*(\d+)\b',
                r'\bup\s*to\s*(\d+)\b'
            ],
            'operator': '<='
        },
        
        'range': {
            'patterns': [
                r'\bbetween\s*(\d+)\s*(?:and|to|-)\s*(\d+)\b',
                r'\b(\d+)\s*(?:to|-)\s*(\d+)\b',
                r'\b(\d+)\s*through\s*(\d+)\b',
                r'\bfrom\s*(\d+)\s*to\s*(\d+)\b'
            ],
            'operator': 'between'
        }
    }

    # Special number word mappings
    NUMBER_WORDS = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'studio': 0,  # Studio apartments have 0 bedrooms
        'single': 1,
        'double': 2,
        'triple': 3,
        'quad': 4,
        'multiple': 2  # Default for 'multiple'
    }

    # Property type numeric expectations
    PROPERTY_NUMERIC_DEFAULTS = {
        'studio': {'bedrooms': 0, 'bathrooms': 1, 'guests': 2},
        'apartment': {'bedrooms': [1, 3], 'bathrooms': [1, 2], 'guests': [1, 6]},
        'house': {'bedrooms': [2, 6], 'bathrooms': [1, 4], 'guests': [3, 12]},
        'villa': {'bedrooms': [3, 8], 'bathrooms': [2, 6], 'guests': [6, 16]},
        'condo': {'bedrooms': [1, 3], 'bathrooms': [1, 2], 'guests': [2, 6]},
        'loft': {'bedrooms': [1, 2], 'bathrooms': [1, 2], 'guests': [2, 4]},
        'cabin': {'bedrooms': [1, 4], 'bathrooms': [1, 3], 'guests': [2, 8]},
        'townhouse': {'bedrooms': [2, 4], 'bathrooms': [1, 3], 'guests': [4, 8]}
    }

    # Context enhancement patterns
    CONTEXT_PATTERNS = {
        'family': {
            'implications': {'bedrooms': [2, 4], 'guests': [4, 8]},
            'keywords': ['family', 'families', 'kids', 'children', 'child']
        },
        'couple': {
            'implications': {'bedrooms': [1, 2], 'guests': 2},
            'keywords': ['couple', 'couples', 'romantic', 'honeymoon', 'two people']
        },
        'group': {
            'implications': {'bedrooms': [3, 6], 'guests': [6, 12]},
            'keywords': ['group', 'groups', 'friends', 'party', 'large group', 'big group']
        },
        'business': {
            'implications': {'bedrooms': 1, 'guests': [1, 2]},
            'keywords': ['business', 'work', 'corporate', 'meeting', 'conference']
        },
        'luxury': {
            'implications': {'bedrooms': [2, 5], 'bathrooms': [2, 4], 'price': [200, 1000]},
            'keywords': ['luxury', 'luxurious', 'premium', 'high-end', 'upscale']
        },
        'budget': {
            'implications': {'price': [20, 100]},
            'keywords': ['budget', 'cheap', 'affordable', 'economical', 'low-cost']
        }
    }

    # Query intent patterns for numeric queries
    NUMERIC_INTENT_PATTERNS = {
        'specific_search': {
            'patterns': [
                r'\bfind\s.*\d+\s*(?:bed|bedroom|bath|bathroom|guest)s?\b',
                r'\bshow\s.*\d+\s*(?:bed|bedroom|bath|bathroom|guest)s?\b',
                r'\blooking\s+for\s.*\d+\s*(?:bed|bedroom|bath|bathroom|guest)s?\b'
            ],
            'intent': 'search_with_constraints'
        },
        'price_filter': {
            'patterns': [
                r'\b(?:under|below|less\s+than|cheaper\s+than)\s*\$?\d+\b',
                r'\b(?:over|above|more\s+than|expensive\s+than)\s*\$?\d+\b',
                r'\bbetween\s*\$?\d+\s*(?:and|to)\s*\$?\d+\b'
            ],
            'intent': 'price_filter'
        },
        'capacity_search': {
            'patterns': [
                r'\bfor\s*\d+\s*(?:people|guests|person)s?\b',
                r'\baccommodate\s*\d+\s*(?:people|guests)s?\b',
                r'\bsleeps\s*\d+\b'
            ],
            'intent': 'capacity_search'
        }
    }
