"""Pydantic models for Muninn API."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ── Peers ──────────────────────────────────────────────────

class PeerCreate(BaseModel):
    id: str = Field(..., description="Unique peer ID, e.g. 'sombra_rechazo'")
    name: str = Field(..., description="Display name")
    type: str = Field(..., description="sombra|persona|proyecto|tema|sistema")
    description: Optional[str] = None
    representation: Optional[str] = None
    confidence: float = Field(default=0.1, ge=0.0, le=1.0)
    activation_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    embedding_text: Optional[str] = Field(None, description="Text to embed for activation. Defaults to description.")


class PeerUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    representation: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    activation_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None
    embedding_text: Optional[str] = None


class PeerResponse(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str]
    representation: Optional[str]
    confidence: float
    activation_threshold: float
    tags: List[str]
    is_active: int
    activation_count: int
    created_at: Optional[str]
    updated_at: Optional[str]
    last_activated_at: Optional[str]


# ── Memories ───────────────────────────────────────────────

class MemoryCreate(BaseModel):
    content: str = Field(..., description="The memory content")
    type: str = Field(..., description="hecho|episodio|preference|patron|hipotesis")
    source: str = Field(default="manual")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    occurred_at: Optional[str] = None
    session_id: Optional[str] = None
    source_channel: Optional[str] = None
    peer_ids: List[str] = Field(default_factory=list, description="Peers to link this memory to")
    metadata: dict = Field(default_factory=dict)


class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Optional[dict] = None


class MemoryResponse(BaseModel):
    id: int
    content: str
    type: str
    source: str
    confidence: float
    is_active: int
    occurred_at: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    session_id: Optional[str]
    source_channel: Optional[str]
    metadata: dict
    peers: List[dict] = Field(default_factory=list)


# ── Events ─────────────────────────────────────────────────

class EventCreate(BaseModel):
    session_id: Optional[str] = None
    type: str = Field(..., description="user_message|bot_response|tool_call|tool_result|thinking")
    content: str
    channel: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class EventResponse(BaseModel):
    event_id: int
    activations: List[dict]


# ── Connections ────────────────────────────────────────────

class ConnectionCreate(BaseModel):
    from_peer_id: str
    to_peer_id: str
    relation_type: str = Field(..., description="conecta|activa|contradice|evoluciona_de")
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    description: Optional[str] = None


class ConnectionResponse(BaseModel):
    id: int
    from_peer_id: str
    to_peer_id: str
    relation_type: str
    strength: float
    description: Optional[str]
    created_at: Optional[str]


# ── Search ─────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    peer_id: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)
    method: str = Field(default="hybrid", description="semantic|fts|hybrid")


class SearchResult(BaseModel):
    memory_id: int
    content: str
    type: str
    score: float
    peers: List[str] = Field(default_factory=list)


# ── Sessions ───────────────────────────────────────────────

class SessionCreate(BaseModel):
    id: str
    channel: str
    chat_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class SessionResponse(BaseModel):
    id: str
    channel: str
    chat_id: Optional[str]
    started_at: Optional[str]
    ended_at: Optional[str]
    summary: Optional[str]
    event_count: int
    metadata: dict


# ── Generic ────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str
    detail: Optional[dict] = None
