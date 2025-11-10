"""
Database Schemas for Real Estate Sales AI Platform

Each Pydantic model maps to a MongoDB collection (lowercased class name).
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any

# Core user (seller) profile
class Seller(BaseModel):
    name: str = Field(..., description="Seller full name")
    email: str = Field(..., description="Unique email for the seller")
    team: Optional[str] = Field(None, description="Team/branch name")
    role: Literal["seller", "manager", "admin"] = Field("seller")
    is_active: bool = Field(True)

# Persona definitions are stored as documents to allow future edits
class Persona(BaseModel):
    key: str = Field(..., description="Unique identifier, e.g., indeciso")
    name: str
    description: str
    traits: List[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"

# A roleplay session (one conversation between a seller and a persona)
class RoleplaySession(BaseModel):
    seller_email: str = Field(..., description="Email of the seller running this session")
    persona_key: str
    status: Literal["active", "finished"] = "active"
    current_score: float = 0.0
    total_messages: int = 0
    # Messages are stored inline for simplicity (each as dict with role, text, ts)
    messages: List[Dict[str, Any]] = Field(default_factory=list)

# Historical snapshots of final scores (useful for analytics)
class SessionScore(BaseModel):
    session_id: str
    seller_email: str
    persona_key: str
    final_score: float
