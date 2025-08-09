"""Pydantic models for API request and response."""

from typing import List, Union
from pydantic import BaseModel, Field, validator


class PolicyQueryRequest(BaseModel):
    """Request model for policy query API."""
    
    documents: Union[str, List[str]] = Field(..., description="Document URL(s) to process")
    questions: List[str] = Field(..., description="List of questions to answer")
    
    @validator('documents')
    def convert_documents_to_list(cls, v):
        """Convert single document URL to list."""
        if isinstance(v, str):
            return [v]
        return v
    
    @validator('questions')
    def validate_questions(cls, v):
        """Ensure questions list is not empty."""
        if not v:
            raise ValueError("Questions list cannot be empty")
        return v


class PolicyQueryResponse(BaseModel):
    """Response model for policy query API."""
    
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: str = Field(None, description="Detailed error information")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
