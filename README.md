# Advanced Question-Answering System with RAG and SQL Support

This project implements an advanced question-answering system that combines Retrieval-Augmented Generation (RAG) with SQL database querying capabilities. It can handle both unstructured document queries and structured database queries seamlessly.

## Features

- **RAG-based Document Querying**: Process and query various document types (PDF, TXT, DOCX, CSV, etc.)
- **SQL Database Integration**: Direct querying of SQL databases with natural language
- **Modern Web Interface**: Clean and intuitive Gradio-based UI
- **Flexible Configuration**: YAML-based configuration for easy customization
- **Error Handling**: Robust error handling for ambiguous queries and numeric conversions
- **Environment Variable Management**: Uses `.env` for sensitive configurations like API keys

## Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Using_Langchain
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
.
├── configs/
│   └── custom_config.yml    # Configuration settings
├── data/                    
│   ├── unstructured_docs/   # Document storage(store you document inside this directory)
│   └── *.sqlite            # SQLite databases(store you basebase directly inside data directory)
├── src/
│   ├── agent_graph/        # Agent implementation
│   ├── chatbot/           # Chatbot backend
│   ├── utils/             # Utility functions
│   ├── app.py            # Main application
│   └── prepare_vector_db.py # Vector DB preparation
└── requirements.txt       # Python dependencies


```

## Download Sample Database

1. Visit the [Kaggle: E-commerce Dataset by Olist as an SQLite Database](https://www.kaggle.com/datasets/terencicp/e-commerce-dataset-by-olist-as-an-sqlite-database).
2. Save the downloaded file in the `data` directory of your project.

## Configuration

The system is configured through `configs/custom_config.yml`. Key configuration sections include:

- **Vector Database Configuration**: Settings for document processing and embeddings
- **SQL Database Configuration**: Database connection details
- **LLM Configuration**: Language model settings
- **Application Settings**: Web interface configuration

Add paths and description in the custom_config.yml file

## Usage

1. Prepare the vector database (if using document querying):
```bash
python src/prepare_vector_db.py
```

2. Start the application:
```bash
python src/app.py
```

3. Access the web interface at `http://localhost:7860`

## Query Examples

### Document Queries
- "Explain multi head attention?"
- "Summarize the key findings from the research papers."

### SQL Queries
- "Show me the top 5 selling products."
- "What is the total revenue for each category?"

## Features in Detail

### RAG System
- Uses OpenAI embeddings for document processing
- Supports multiple document collections
- Configurable chunk size and overlap
- Persistent vector storage

### SQL Integration
- Natural language to SQL conversion
- Support for complex queries

### Web Interface
- Real-time query processing
- Clear display of available databases and collections
- Easy-to-use chat interface
- Error feedback and suggestions


## Acknowledgments

- Built with LangChain
- Uses OpenAI's GPT models
- Powered by Chroma vector database 