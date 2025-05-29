"""
Airbnb-specific configuration and field mappings for the JSON RAG system.
Defines field categories, weights, importance rankings, and domain-specific processing rules.
"""

# Field categories for organizing Airbnb property data
FIELD_CATEGORIES = {
    'pricing': [
        'price', 'cleaning_fee', 'extra_people', 'security_deposit',
        'weekly_price', 'monthly_price'
    ],
    'location': [
        'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 
        'city', 'zipcode', 'latitude', 'longitude', 'street',
        'market', 'smart_location', 'country_code', 'country'
    ],
    'amenities': [
        'amenities', 'host_verifications'
    ],
    'host_info': [
        'host_id', 'host_name', 'host_since', 'host_response_time',
        'host_response_rate', 'host_is_superhost', 'host_listings_count',
        'host_total_listings_count', 'host_neighbourhood', 'host_about'
    ],
    'property_details': [
        'property_type', 'room_type', 'bed_type', 'accommodates',
        'bedrooms', 'bathrooms', 'beds', 'minimum_nights', 'maximum_nights'
    ],
    'reviews': [
        'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
        'review_scores_cleanliness', 'review_scores_checkin', 
        'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'reviews_per_month'
    ],
    'availability': [
        'availability_30', 'availability_60', 'availability_90', 
        'availability_365', 'calendar_updated', 'calendar_last_scraped'
    ],
    'policies': [
        'cancellation_policy', 'require_guest_profile_picture',
        'require_guest_phone_verification', 'instant_bookable'
    ]
}

# Main Airbnb configuration
AIRBNB_CONFIG = {
    # Field weights for search relevance
    'field_weights': {
        'name': 1.0,
        'summary': 0.9,
        'description': 0.7,
        'space': 0.6,
        'neighbourhood_overview': 0.6,
        'amenities': 0.8,
        'property_type': 0.7,
        'room_type': 0.6,
        'neighbourhood_cleansed': 0.5,
        'host_about': 0.3
    },
    
    # Field importance for display (higher = more important)
    'field_importance': {
        'name': 10,
        'price': 9,
        'property_type': 8,
        'room_type': 8,
        'accommodates': 7,
        'bedrooms': 7,
        'bathrooms': 7,
        'amenities': 6,
        'neighbourhood_cleansed': 6,
        'review_scores_rating': 5,
        'number_of_reviews': 5,
        'minimum_nights': 4,
        'host_is_superhost': 4
    },
    
    # Query processing rules
    'query_processing': {
        'price_keywords': ['price', 'cost', 'budget', 'expensive', 'cheap', 'affordable', '$'],
        'location_keywords': ['near', 'close', 'area', 'neighborhood', 'district', 'location'],
        'amenity_keywords': ['wifi', 'parking', 'kitchen', 'pool', 'gym', 'pet', 'smoking'],
        'property_type_keywords': ['apartment', 'house', 'condo', 'studio', 'loft', 'villa']
    },
    
    # Price range mappings
    'price_ranges': {
        'budget': (0, 75),
        'moderate': (75, 150),
        'premium': (150, 300),
        'luxury': (300, float('inf'))
    },
    
    # Accommodation type categories
    'accommodation_types': {
        'entire_place': ['Entire home/apt'],
        'private_room': ['Private room'],
        'shared_space': ['Shared room']
    },
    
    # Embedding and search configuration
    'embedding_batch_size': 32,
    'max_search_results': 50,
    'similarity_threshold': 0.3,
    
    # Response generation settings
    'max_summary_length': 500,
    'show_json_in_response': True,
    'include_search_metadata': True
}

# Property type standardization
PROPERTY_TYPE_MAPPING = {
    'apt': 'Apartment',
    'apartment': 'Apartment', 
    'condo': 'Condominium',
    'house': 'House',
    'home': 'House',
    'villa': 'Villa',
    'studio': 'Apartment',
    'loft': 'Loft',
    'townhouse': 'Townhome'
}

# Amenity standardization
AMENITY_MAPPING = {
    'wifi': ['Wifi', 'Wireless Internet', 'Internet'],
    'parking': ['Free parking', 'Paid parking', 'Parking'],
    'kitchen': ['Kitchen', 'Kitchenette'],
    'pool': ['Pool', 'Swimming pool'],
    'gym': ['Gym', 'Fitness center', 'Exercise equipment'],
    'pets': ['Pets allowed', 'Cat(s)', 'Dog(s)'],
    'smoking': ['Smoking allowed']
}

# Field display names for user-friendly output
FIELD_DISPLAY_NAMES = {
    'accommodates': 'Guests',
    'bedrooms': 'Bedrooms', 
    'bathrooms': 'Bathrooms',
    'beds': 'Beds',
    'property_type': 'Property Type',
    'room_type': 'Room Type',
    'neighbourhood_cleansed': 'Neighborhood',
    'minimum_nights': 'Min Nights',
    'review_scores_rating': 'Rating',
    'number_of_reviews': 'Reviews',
    'host_is_superhost': 'Superhost'
}

def get_field_category(field_name):
    """Get the category for a given field name."""
    for category, fields in FIELD_CATEGORIES.items():
        if field_name in fields:
            return category
    return 'other'

def get_field_weight(field_name):
    """Get the search weight for a field."""
    return AIRBNB_CONFIG['field_weights'].get(field_name, 0.1)

def get_field_importance(field_name):
    """Get the display importance for a field."""
    return AIRBNB_CONFIG['field_importance'].get(field_name, 1)

def standardize_property_type(property_type):
    """Standardize property type names."""
    if not property_type:
        return property_type
    
    property_type_lower = property_type.lower()
    return PROPERTY_TYPE_MAPPING.get(property_type_lower, property_type)

def get_amenity_keywords(query_text):
    """Extract amenity-related keywords from query text."""
    query_lower = query_text.lower()
    found_amenities = []
    
    for standard_amenity, variations in AMENITY_MAPPING.items():
        if standard_amenity in query_lower:
            found_amenities.append(standard_amenity)
        else:
            for variation in variations:
                if variation.lower() in query_lower:
                    found_amenities.append(standard_amenity)
                    break
    
    return found_amenities

def get_display_name(field_name):
    """Get user-friendly display name for a field."""
    return FIELD_DISPLAY_NAMES.get(field_name, field_name.replace('_', ' ').title())
