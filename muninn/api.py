"""Muninn API — FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Muninn",
    description="Memory system for AI agents — inspired by depth psychology",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "name": "Muninn",
        "version": "0.1.0",
        "description": "Memory system for AI agents",
    }


@app.get("/stats")
async def stats():
    """General statistics."""
    # TODO: implement with real DB
    return {
        "total_peers": 0,
        "total_memories": 0,
        "total_events": 0,
        "total_activations": 0,
    }
