"""FastAPI application for policy query service."""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .models import PolicyQueryRequest, PolicyQueryResponse, ErrorResponse, HealthCheckResponse
from src.core.service import PolicyQueryService

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
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),  # Configure properly for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
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
    try:
        # Process documents and answer questions
        answers = policy_service.process_documents_and_answer_questions(
            document_urls=request.documents,
            questions=request.questions
        )
        
        return PolicyQueryResponse(answers=answers)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service error: {str(e)}"
        )
    except Exception as e:
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
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
