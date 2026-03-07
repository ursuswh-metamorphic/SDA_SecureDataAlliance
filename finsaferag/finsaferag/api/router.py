"""
Routing module to direct queries to appropriate retrievers
This is a MOCK implementation for future expansion
"""
import logging
from typing import Optional, Dict, Any, List
from enum import Enum


logger = logging.getLogger(__name__)


class RetrieverType(str, Enum):
    """Available retriever types"""
    FINANCIAL = "financial"  # Financial documents
    GENERAL = "general"      # General knowledge
    TECHNICAL = "technical"  # Technical documentation
    LEGAL = "legal"          # Legal documents
    DEFAULT = "default"      # Default retriever


class QueryRouter:
    """
    Router to determine which retriever should handle a query.
    
    MOCK IMPLEMENTATION: Currently routes all queries to default retriever.
    Future implementation will use ML/rule-based routing.
    """
    
    def __init__(self):
        self.available_retrievers = [r.value for r in RetrieverType]
        logger.info(f"QueryRouter initialized with retrievers: {self.available_retrievers}")
        
        # Mock: Keywords for different domains (for future implementation)
        self.domain_keywords = {
            RetrieverType.FINANCIAL: [
                "revenue", "profit", "earnings", "stock", "investment", 
                "financial", "fiscal", "balance sheet", "income statement",
                "cash flow", "EBITDA", "ROI", "market cap", "dividend"
            ],
            RetrieverType.TECHNICAL: [
                "algorithm", "code", "programming", "API", "database",
                "architecture", "deployment", "testing", "debug"
            ],
            RetrieverType.LEGAL: [
                "contract", "law", "regulation", "compliance", "policy",
                "agreement", "terms", "legal", "litigation"
            ],
            RetrieverType.GENERAL: []  # Fallback
        }
    
    def route_query(
        self, 
        query: str, 
        user_preference: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route a query to the appropriate retriever.
        
        Args:
            query: User's question
            user_preference: User-specified retriever type
            context: Additional context (session history, user profile, etc.)
            
        Returns:
            Dict with routing decision:
            {
                "retriever_type": str,
                "confidence": float,
                "reasoning": str,
                "fallback": bool
            }
        """
        logger.info(f"Routing query: '{query[:50]}...'")
        
        # If user explicitly specified a retriever, use it
        if user_preference and user_preference in self.available_retrievers:
            logger.info(f"Using user-specified retriever: {user_preference}")
            return {
                "retriever_type": user_preference,
                "confidence": 1.0,
                "reasoning": "User-specified retriever",
                "fallback": False
            }
        
        # MOCK: Simple keyword-based routing (placeholder for ML model)
        detected_type = self._keyword_match(query)
        
        # MOCK: For now, always route to DEFAULT (current implementation)
        # In future, this will route to specific retrievers
        return {
            "retriever_type": RetrieverType.DEFAULT.value,
            "confidence": 0.8,
            "reasoning": f"MOCK: Detected domain '{detected_type}' but routing to DEFAULT (current implementation)",
            "fallback": False,
            "detected_domain": detected_type
        }
    
    def _keyword_match(self, query: str) -> str:
        """
        Simple keyword matching for domain detection.
        MOCK implementation - will be replaced with ML model.
        """
        query_lower = query.lower()
        
        # Score each domain
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            domain_scores[domain] = score
        
        # Get domain with highest score
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            if best_domain[1] > 0:
                return best_domain[0].value
        
        return RetrieverType.DEFAULT.value
    
    def get_available_retrievers(self) -> List[str]:
        """Get list of available retrievers"""
        return self.available_retrievers
    
    def validate_retriever(self, retriever_type: str) -> bool:
        """Check if a retriever type is valid"""
        return retriever_type in self.available_retrievers


# Singleton instance
_router_instance = None


def get_router() -> QueryRouter:
    """Get or create router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = QueryRouter()
    return _router_instance

