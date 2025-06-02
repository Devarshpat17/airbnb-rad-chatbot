"""Configuration file for Airbnb-specific data handling and search optimization."""

# Property type synonyms for query understanding
PROPERTY_TYPE_SYNONYMS = {
    'apartment': [
        'apartment', 'flat', 'unit', 'condo', 'condominium', 'studio',
        'pied-a-terre', 'loft', 'duplex'
    ],
    'house': [
        'house', 'home', 'villa', 'bungalow', 'cottage', 'cabin',
        'chalet', 'mansion', 'residence', 'detached'
    ],
    'room': [
        'room', 'bedroom', 'private room', 'shared room', 'guest room',
        'master bedroom', 'dormitory', 'dorm'
    ],
    'studio': [
        'studio', 'bachelor', 'efficiency', 'micro apartment', 'bedsit',
        'studio flat', 'open-plan'
    ],
    'townhouse': [
        'townhouse', 'townhome', 'row house', 'brownstone', 'mews',
        'terraced house', 'connected home'
    ],
    'guesthouse': [
        'guesthouse', 'guest house', 'guest suite', 'casita', 'annex',
        'granny flat', 'pool house', 'garden house'
    ],
    'unique': [
        'treehouse', 'boat', 'yacht', 'houseboat', 'rv', 'camper',
        'tiny house', 'barn', 'farm stay', 'lighthouse', 'castle'
    ]
}

# Amenity synonyms for flexible matching
AMENITY_SYNONYMS = {
    'wifi': [
        'wifi', 'wi-fi', 'wireless', 'internet', 'broadband', 'connectivity',
        'network', 'web access', 'online'
    ],
    'parking': [
        'parking', 'garage', 'carport', 'driveway', 'parking spot',
        'parking space', 'car park', 'covered parking', 'street parking'
    ],
    'pool': [
        'pool', 'swimming pool', 'swimming', 'swim', 'outdoor pool',
        'indoor pool', 'heated pool', 'lap pool'
    ],
    'kitchen': [
        'kitchen', 'kitchenette', 'cooking', 'full kitchen', 'private kitchen',
        'equipped kitchen', 'modern kitchen', 'chef kitchen'
    ],
    'ac': [
        'ac', 'air conditioning', 'air con', 'cooling', 'climate control',
        'central air', 'air conditioner', 'hvac'
    ],
    'heating': [
        'heating', 'heat', 'central heating', 'heater', 'radiator',
        'furnace', 'heated floors', 'thermostat'
    ],
    'washer': [
        'washer', 'washing machine', 'laundry', 'clothes washer',
        'washer dryer', 'laundry facilities', 'washing'
    ],
    'tv': [
        'tv', 'television', 'smart tv', 'cable tv', 'satellite tv',
        'netflix', 'streaming', 'hdtv', 'flat screen'
    ],
    'workspace': [
        'workspace', 'desk', 'work desk', 'office', 'study area',
        'laptop friendly', 'work station', 'business center'
    ],
    'gym': [
        'gym', 'fitness', 'fitness center', 'exercise room', 'workout',
        'fitness equipment', 'exercise equipment', 'weights'
    ]
}

# Location synonyms for area matching
LOCATION_SYNONYMS = {
    'downtown': [
        'downtown', 'city center', 'central', 'cbd', 'heart of city',
        'city centre', 'urban core', 'midtown'
    ],
    'beach': [
        'beach', 'beachfront', 'oceanfront', 'seaside', 'coastal',
        'waterfront', 'shore', 'beach access'
    ],
    'suburb': [
        'suburb', 'suburban', 'residential', 'residential area',
        'quiet neighborhood', 'suburbs', 'outskirts'
    ],
    'airport': [
        'airport', 'near airport', 'airport area', 'airport vicinity',
        'airport shuttle', 'airport transfer'
    ],
    'shopping': [
        'shopping', 'mall', 'shopping center', 'retail', 'shops',
        'shopping district', 'commercial', 'market'
    ],
    'historic': [
        'historic', 'old town', 'historic district', 'heritage',
        'historic center', 'cultural district', 'historic area'
    ],
    'business': [
        'business district', 'financial district', 'commercial district',
        'business center', 'corporate', 'office district'
    ],
    'entertainment': [
        'entertainment district', 'nightlife', 'theater district',
        'restaurant row', 'dining district', 'entertainment area'
    ]
}

# Field categories for organizing Airbnb property data
FIELD_CATEGORIES = {
    'pricing': {
        'fields': ['price', 'nightly_rate', 'weekly_rate', 'monthly_rate', 'cleaning_fee', 'service_fee', 'total_price'],
        'weight': 0.9,
        'description': 'Pricing information'
    },
    'location': {
        'fields': ['address', 'city', 'state', 'country', 'neighborhood', 'zipcode', 'latitude', 'longitude'],
        'weight': 1.0,
        'description': 'Location details'
    },
    'amenities': {
        'fields': ['amenities', 'house_amenities', 'safety_items', 'accessibility'],
        'weight': 0.8,
        'description': 'Property amenities and features'
    },
    'property': {
        'fields': ['property_type', 'room_type', 'bedrooms', 'bathrooms', 'beds', 'accommodates', 'square_feet'],
        'weight': 1.0,
        'description': 'Property specifications'
    },
    'host_info': {
        'fields': ['host_name', 'host_since', 'host_response_time', 'host_response_rate', 'host_is_superhost'],
        'weight': 0.6,
        'description': 'Host information'
    },
    'reviews': {
        'fields': ['overall_rating', 'accuracy_rating', 'cleanliness_rating', 'checkin_rating', 'communication_rating', 'location_rating', 'value_rating', 'review_count'],
        'weight': 0.8,
        'description': 'Reviews and ratings'
    },
    'general': {
        'fields': ['name', 'description', 'summary', 'space', 'access', 'interaction', 'neighborhood_overview', 'notes', 'transit'],
        'weight': 0.7,
        'description': 'General property information'
    }
}

# Main Airbnb configuration
AIRBNB_CONFIG = {
    # Search field weights
    'field_weights': {
        'name': 1.0,
        'description': 0.7,
        'amenities': 0.8,
        'neighborhood': 0.9,
        'property_type': 0.8,
        'room_type': 0.7,
        'summary': 0.6
    },
    
    # Property type mappings
    'property_mappings': {
        'apartment': ['apartment', 'flat', 'condo', 'condominium', 'unit'],
        'house': ['house', 'home', 'villa', 'cottage', 'cabin', 'bungalow'],
        'room': ['room', 'bedroom', 'private room', 'shared room'],
        'studio': ['studio', 'efficiency', 'bachelor'],
        'loft': ['loft', 'penthouse', 'attic'],
        'townhouse': ['townhouse', 'townhome', 'row house'],
        'other': ['other', 'unique', 'unusual']
    },
    
    # Amenity categories
    'amenity_categories': {
        'kitchen': ['kitchen', 'cooking', 'microwave', 'refrigerator', 'stove', 'oven', 'dishwasher', 'coffee maker'],
        'internet': ['wifi', 'internet', 'wireless', 'broadband'],
        'parking': ['parking', 'garage', 'driveway', 'street parking'],
        'laundry': ['washer', 'dryer', 'laundry', 'washing machine'],
        'entertainment': ['tv', 'television', 'cable', 'netflix', 'streaming'],
        'comfort': ['air conditioning', 'heating', 'fireplace', 'fan'],
        'outdoor': ['pool', 'hot tub', 'balcony', 'patio', 'garden', 'yard'],
        'safety': ['smoke detector', 'carbon monoxide detector', 'first aid kit', 'fire extinguisher']
    },
    
    # Synonym mappings for query expansion (ASCII only)
    'synonyms': {
        'cheap': ['budget', 'affordable', 'economical', 'inexpensive', 'low cost', 'value'],
        'expensive': ['luxury', 'premium', 'high-end', 'upscale', 'costly', 'pricey'],
        'close': ['near', 'nearby', 'adjacent', 'walking distance', 'proximity', 'convenient'],
        'big': ['large', 'spacious', 'huge', 'roomy', 'expansive', 'vast'],
        'small': ['cozy', 'compact', 'tiny', 'intimate', 'snug', 'modest'],
        'clean': ['spotless', 'pristine', 'immaculate', 'tidy', 'neat', 'sanitized'],
        'nice': ['great', 'wonderful', 'excellent', 'amazing', 'fantastic', 'superb'],
        'quiet': ['peaceful', 'tranquil', 'calm', 'serene', 'silent', 'still'],
        'central': ['downtown', 'city center', 'urban', 'metropolitan', 'core', 'heart'],
        'modern': ['contemporary', 'updated', 'renovated', 'new', 'current', 'fresh'],
        'wifi': ['internet', 'wireless', 'broadband', 'connection', 'online'],
        'parking': ['garage', 'spot', 'space', 'lot', 'driveway'],
        'kitchen': ['cooking', 'culinary', 'food prep', 'kitchenette'],
        'pool': ['swimming', 'swim', 'water', 'aquatic'],
        'beach': ['ocean', 'sea', 'waterfront', 'shore', 'coastal'],
        'mountain': ['hill', 'peak', 'elevation', 'scenic', 'nature']
    },
    
    # Price range mappings
    'price_ranges': {
        'budget': {'min': 0, 'max': 50},
        'moderate': {'min': 50, 'max': 150},
        'expensive': {'min': 150, 'max': 300},
        'luxury': {'min': 300, 'max': 1000}
    },
    
    # Accommodation capacity mappings
    'accommodation_ranges': {
        'solo': {'min': 1, 'max': 1},
        'couple': {'min': 2, 'max': 2},
        'small_group': {'min': 3, 'max': 4},
        'large_group': {'min': 5, 'max': 8},
        'party': {'min': 9, 'max': 20}
    }
}
