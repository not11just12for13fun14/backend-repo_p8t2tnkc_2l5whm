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
    password_hash: Optional[str] = Field(None, description="Password hash for auth")

# Persona definitions are stored as documents to allow future edits
class Persona(BaseModel):
    key: str = Field(..., description="Unique identifier, e.g., indeciso")
    name: str
    description: str
    traits: List[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    disc_profile: Optional[Literal["D", "I", "S", "C"]] = None
    triggers: List[str] = Field(default_factory=list, description="Behavioral triggers for this persona")

# A roleplay session (one conversation between a seller and a persona)
class RoleplaySession(BaseModel):
    seller_email: str = Field(..., description="Email of the seller running this session")
    persona_key: str
    status: Literal["active", "finished"] = "active"
    current_score: float = 0.0
    total_messages: int = 0
    scoring_weights: Dict[str, float] = Field(default_factory=lambda: {"rapport": 0.3, "discovery": 0.2, "objection": 0.3, "closing": 0.2})
    premium_unlocked: bool = False
    # Messages are stored inline for simplicity (each as dict with role, text, ts)
    messages: List[Dict[str, Any]] = Field(default_factory=list)

# Historical snapshots of final scores (useful for analytics)
class SessionScore(BaseModel):
    session_id: str
    seller_email: str
    persona_key: str
    final_score: float
    weights: Dict[str, float] = Field(default_factory=dict)
    feedback: Optional[str] = None

# Configuration for dynamic scoring weights (per team or global)
class ScoreConfig(BaseModel):
    scope: Literal["global", "team", "user"] = "global"
    team: Optional[str] = None
    email: Optional[str] = None
    weights: Dict[str, float] = Field(default_factory=lambda: {"rapport": 0.25, "discovery": 0.25, "objection": 0.3, "closing": 0.2})

# Coaching rubric templates
class CoachingRubric(BaseModel):
    name: str
    description: str
    criteria: List[str]
    weights: Dict[str, float]

