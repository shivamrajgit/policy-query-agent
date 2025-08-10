"""FastAPI application for policy query service."""

import os
import logging
import time
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .models import PolicyQueryRequest, PolicyQueryResponse, ErrorResponse, HealthCheckResponse
from src.core.service import PolicyQueryService

# Configure simple logging for terminal output only
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Insurance Policy Query API",
    description="API for querying insurance policy documents using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global service instance
policy_service = PolicyQueryService()

# API Key validation
def validate_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Validate API key from Authorization header."""
    api_key = os.getenv("API_KEY")
    
    # If no API key is set in environment, skip validation (for development)
    if not api_key:
        return credentials.credentials
    
    if credentials.credentials != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        message="Policy Query API is running"
    )


@app.post(
    "/hackrx/run",
    response_model=PolicyQueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)
async def query_policy_documents(
    request: PolicyQueryRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    Process policy documents and answer questions.
    
    This endpoint accepts a list of document URLs and questions,
    processes the documents using AI, and returns answers to the questions.
    """
    # Generate unique request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()
    
    print(f"\n=== Processing {len(request.documents)} documents, {len(request.questions)} questions ===")
    
    try:
        # Process documents and answer questions
        answers = policy_service.process_documents_and_answer_questions(
            document_urls=request.documents,
            questions=request.questions,
            request_id=request_id
        )
        
        total_time = time.perf_counter() - start_time
        print(f"=== Request completed in {total_time:.2f}s ===\n")
        
        return PolicyQueryResponse(answers=answers)
        
    except ValueError as e:
        total_time = time.perf_counter() - start_time
        print(f"ERROR: Bad request after {total_time:.2f}s: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        total_time = time.perf_counter() - start_time
        print(f"ERROR: Service error after {total_time:.2f}s: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service error: {str(e)}"
        )
    except Exception as e:
        total_time = time.perf_counter() - start_time
        print(f"ERROR: Unexpected error after {total_time:.2f}s: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Insurance Policy Query API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "performance_monitoring": "All requests include detailed backend timing logs in console output"
    }


@app.get("/performance/info")
async def performance_info():
    """Get information about performance monitoring capabilities."""
    return {
        "timing_features": {
            "total_request_time": "Complete request-response cycle timing",
            "document_processing_time": "Time to download, process and vectorize documents", 
            "questions_processing_time": "Time to answer all questions in parallel",
            "individual_question_times": "Per-question timing with model assignment",
            "backend_logging": "Detailed timing logs written to console output"
        },
        "log_format": {
            "request_tracking": "Each request gets unique ID [REQ-xxxxxxxx]",
            "emojis": "Visual indicators: 🚀 start, 📥 docs, 🤔 questions, ✅ success", 
            "timings": "Detailed step-by-step timing in seconds",
            "parallel_processing": "Real-time progress of parallel question processing"
        },
        "log_locations": [
            "Console output (if running in terminal)"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
