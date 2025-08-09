# Insurance Policy Query API

A FastAPI-based service that uses AI to automatically answer questions about insurance policy documents. The application processes PDF documents from URLs and uses Google's Generative AI with LangGraph workflows to provide accurate, context-based answers.

## Features

- **Document Processing**: Automatically downloads and processes PDF documents from URLs
- **AI-Powered Q&A**: Uses Google Generative AI models (Gemini 2.0) for intelligent question answering
- **Vector Search**: FAISS-based similarity search for relevant content retrieval
- **Parallel Processing**: Multi-threaded question processing with multiple API keys for improved performance
- **RESTful API**: Clean FastAPI interface with automatic documentation
- **Authentication**: Bearer token authentication support
- **Health Monitoring**: Built-in health check endpoints

## How It Works

1. **Document Loading**: Downloads PDF documents from provided URLs
2. **Text Processing**: Extracts and chunks text content using RecursiveCharacterTextSplitter
3. **Vectorization**: Creates FAISS vector store using Google's embedding model
4. **Query Processing**: Uses LangGraph workflow with three nodes:
   - **Query Refiner**: Optimizes questions for better search results
   - **Context Retriever**: Finds relevant document sections using similarity search
   - **Answering LLM**: Generates answers based on retrieved context
5. **Parallel Processing**: Distributes questions across multiple API keys for faster processing

## Project Structure

```
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── request.json              # Sample API request (use this for testing)
├── Proposed_app.txt          # API documentation and usage examples
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI application and routes
│   │   └── models.py         # Pydantic models for requests/responses
│   ├── core/
│   │   ├── service.py        # Main business logic service
│   │   └── workflow.py       # LangGraph AI workflow implementation
│   └── utils/
│       └── document_loader.py # Document processing utilities
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   # Google AI API Keys (JSON array format for multiple keys)
   GOOGLE_API_KEYS=["your_google_api_key_1", "your_google_api_key_2"]
   
   # Optional: API key for endpoint authentication
   API_KEY=your_api_authentication_key
   ```

## Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://localhost:8000` with auto-reload enabled.

### API Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Making Requests

#### Sample Request

You can use the provided `request.json` file as a sample request:

```json
{
    "documents": ["https://example.com/policy.pdf"],
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
}
```

### Response Format

```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "There is a waiting period of thirty-six (36) months...",
        "Yes, the policy covers maternity expenses..."
    ]
}
```

## API Endpoints

### POST `/hackrx/run`

Process policy documents and answer questions.

**Request Body:**
- `documents`: String or array of document URLs (PDF format)
- `questions`: Array of questions to answer

**Response:**
- `answers`: Array of answers corresponding to the input questions

**Authentication:** Bearer token required (if `API_KEY` is set in environment)

### GET `/health`

Health check endpoint to verify service status.

### GET `/`

Root endpoint with API information and links.


## Configuration

### Environment Variables

- `GOOGLE_API_KEYS`: JSON array of Google AI API keys for parallel processing
- `API_KEY`: Optional authentication key for API access
- `GOOGLE_API_KEY`: Fallback single API key (used if GOOGLE_API_KEYS not provided)

### Model Configuration

- **Primary Model**: `gemini-2.0-flash` (for final answer generation)
- **Lite Model**: `gemini-2.0-flash-lite` (for query refinement)
- **Embedding Model**: `models/embedding-001`

## Sample Data

- **`request.json`**: Contains a sample request with an insurance policy URL and typical questions
- **`Proposed_app.txt`**: Provides detailed API documentation, request/response examples, and PowerShell usage commands

## Development

### Running in Development Mode

The application runs with auto-reload enabled by default. Any code changes will automatically restart the server.

### Testing

Use the provided `request.json` file to test the API functionality. The sample includes real insurance policy questions that demonstrate the system's capabilities.

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **LangChain**: Framework for LLM applications
- **LangGraph**: Workflow orchestration for AI applications
- **Google Generative AI**: AI models for embeddings and text generation
- **FAISS**: Vector similarity search
- **PyPDF**: PDF document processing
- **Uvicorn**: ASGI server for FastAPI
