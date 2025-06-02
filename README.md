# JSON RAG System

A sophisticated AI-powered search and retrieval platform designed for Airbnb property data. This system combines state-of-the-art machine learning technologies with traditional search methods to provide intelligent, context-aware property search capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- MongoDB 4.0+ with Airbnb data loaded
- 8GB RAM minimum (16GB recommended)
- 10GB disk space for indexes and cache

### Installation

1. **Clone and Setup Environment**
```bash
git clone <repository-url>
cd json_rag_system
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
export MONGODB_URI="mongodb://localhost:27017"
export PYTHONIOENCODING="ascii"
export PYTHONUTF8="0"
```

3. **Initialize System (Recommended)**
```bash
python setup.py --full-setup
```

4. **Start Application**
```bash
python main.py
```

## 📁 Project Structure

```
json_rag_system/
├── config/                    # Configuration files
│   ├── config.py             # Main configuration
│   ├── airbnb_config.py      # Airbnb-specific settings
│   ├── numeric_config.py     # Numeric processing config
│   ├── logging_config.py     # Logging configuration
│   └── exceptions.py         # Custom exceptions
├── documentation/             # Project documentation
│   ├── COMPLETE_PROJECT_DOCUMENTATION.md     # Master documentation
│   ├── COMPREHENSIVE_TECHNICAL_DOCUMENTATION.docx.txt # Complete Word format guide
│   ├── COMPLETE_TECHNICAL_DOCUMENTATION.docx.txt # Previous documentation
│   └── data_understanding.txt                # Data schema guide
├── cache/                     # Generated cache files
├── data/                      # Vocabulary and configuration data
├── indexes/                   # FAISS indexes and processed documents
├── logs/                      # System logs
├── core_system.py            # Main system components
├── utils.py                  # Utility functions
├── main.py                   # Web interface launcher
├── setup.py                  # System initialization
├── query_processor.py        # Advanced query processing
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Key Features

- **AI-Powered Semantic Search**: Advanced transformer models for query understanding
- **Multi-Modal Search**: Combines semantic, fuzzy, and keyword search
- **Intelligent Query Understanding**: Natural language processing with intent classification
- **Contextual Conversations**: Session-aware follow-up query handling
- **Real-Time Performance**: Sub-2 second search responses
- **Professional Web Interface**: Modern Gradio-based interface

## 📊 Performance Metrics

- **Startup Time**: 10-30 seconds (optimized) vs 2-5 minutes (cold start)
- **Search Response**: <2 seconds for first query
- **Memory Usage**: 2-4 GB typical operation
- **Scalability**: 50,000+ documents efficiently indexed

## 🛠️ Setup Commands

### Complete Setup
```bash
python setup.py --full-setup          # Complete system initialization
```

### Individual Components
```bash
python setup.py --test-db             # Database validation
python setup.py --setup-numeric-config # Numeric patterns
python setup.py --setup-query-processor # Advanced NLP
python setup.py --setup-embeddings    # Embedding cache
```

### Rebuild Operations
```bash
python setup.py --rebuild-vocab       # Vocabulary rebuild
python setup.py --rebuild-indexes     # Index rebuild
python setup.py --rebuild-embeddings  # Cache rebuild
```

## 🔍 Usage Examples

### Basic Queries
- "Find apartments in downtown"
- "2 bedroom places under $100"
- "Places with WiFi and parking"

### Complex Queries
- "Luxury 3-bedroom houses with pool near city center under $300"
- "Highly rated places with good cleanliness scores"

### Follow-up Conversations
```
User: "Find 2 bedroom apartments"
System: [shows results]
User: "What about ones with kitchens?"
System: [adds kitchen requirement to previous search]
```

## 🏗️ Architecture

### Multi-Layer Design
- **Presentation Layer**: Gradio web interface
- **Application Layer**: Main system orchestration
- **Intelligence Layer**: AI-powered search engines
- **Data Layer**: MongoDB + FAISS indexes
- **Infrastructure Layer**: Configuration and logging

### Core Components
- **JSONRAGSystem**: Primary orchestrator
- **QueryUnderstandingEngine**: NLP-powered query analysis
- **SemanticSearchEngine**: AI-powered semantic search
- **NumericSearchEngine**: Constraint-based filtering
- **VocabularyManager**: Domain-specific vocabulary
- **ResponseGenerator**: AI-powered summarization

## 📚 Documentation

Complete technical documentation is available in:
- `documentation/COMPLETE_PROJECT_DOCUMENTATION.md` - Master technical guide
- `documentation/COMPREHENSIVE_TECHNICAL_DOCUMENTATION.docx.txt` - Complete Word-compatible guide with flowcharts and workflows
- `documentation/AI_MODEL_LIMITATIONS_SUMMARY.xlsx.txt` - Excel-compatible AI model limitations and analysis
- `documentation/COMPLETE_TECHNICAL_DOCUMENTATION.docx.txt` - Previous comprehensive documentation
- `documentation/data_understanding.txt` - Data schema reference

## 🔧 Configuration

### Database Settings
```python
MONGODB_URI = "mongodb://localhost:27017"
DATABASE_NAME = "airbnb_database"
COLLECTION_NAME = "properties"
```

### AI Model Settings
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
TOP_K_RESULTS = 5
```

## 🚨 Troubleshooting

### Common Issues
1. **Database Connection Failure**: Check MongoDB service and URI
2. **Memory Issues**: Ensure 8GB+ RAM available
3. **Slow Startup**: Run `python setup.py --full-setup` first
4. **Unicode Errors**: Set `PYTHONIOENCODING=ascii`

### Diagnostic Commands
```bash
python setup.py --test-db              # Test database connection
python setup.py --setup-query-processor # Test advanced features
```

## 📈 System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- MongoDB 4.0+
- 10GB disk space

### Recommended
- Python 3.9+
- 16GB RAM
- SSD storage
- Multi-core CPU

## 🤝 Contributing

1. Review the complete documentation
2. Understand the system architecture
3. Follow the setup procedures
4. Test your changes thoroughly

## 📄 License

This project is part of an academic/research initiative for advanced AI search systems.

---

**For detailed technical information, refer to the comprehensive documentation in the `documentation/` directory.**