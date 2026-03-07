"""
Pydantic models for API requests and responses
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    question: str = Field(..., description="User's question", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for chat history")
    apply_privacy: bool = Field(True, description="Whether to apply privacy protection")
    retriever_type: Optional[str] = Field(None, description="Specific retriever to use (for routing)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is 3M's revenue in 2019?",
                "session_id": "user_123_session_456",
                "apply_privacy": True,
                "retriever_type": "financial"
            }
        }


class PrivacyStats(BaseModel):
    """Privacy statistics"""
    pii_detected: int = Field(0, description="Number of PII entities detected")
    pii_density: float = Field(0.0, description="PII density ratio")
    sentences_removed: int = Field(0, description="Number of sentences removed by Eraser")
    average_risk: float = Field(0.0, description="Average risk score")
    encrypted: bool = Field(False, description="Whether response was encrypted")
    entities: List[str] = Field([], description="Types of entities detected")


class SourceNode(BaseModel):
    """Source document node"""
    text: str = Field(..., description="Node text content")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field({}, description="Node metadata")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Protected answer from LLM")
    original_answer: Optional[str] = Field(None, description="Original answer before privacy protection")
    privacy_applied: bool = Field(False, description="Whether privacy was applied")
    privacy_stats: Optional[PrivacyStats] = Field(None, description="Privacy statistics")
    source_nodes: List[SourceNode] = Field([], description="Source documents used")
    response_time: float = Field(..., description="Response time in seconds")
    routed_to: Optional[str] = Field(None, description="Which retriever was used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is 3M's revenue?",
                "answer": "3M's revenue in 2019 was approximately <MONEY>.",
                "original_answer": "3M's revenue in 2019 was approximately $32.1 billion.",
                "privacy_applied": True,
                "privacy_stats": {
                    "pii_detected": 1,
                    "pii_density": 0.05,
                    "sentences_removed": 0,
                    "average_risk": 0.3,
                    "encrypted": False,
                    "entities": ["MONEY"]
                },
                "response_time": 2.5,
                "routed_to": "financial"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field("healthy", description="Service status")
    version: str = Field(..., description="API version")
    privacy_enabled: bool = Field(..., description="Privacy module status")
    available_retrievers: List[str] = Field([], description="Available retriever types")
    federated_ready: bool = False
    num_clients: int = 0
    flower_grid_status: str = "not_connected"  

class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")