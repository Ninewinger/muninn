"""Muninn — Memory system for AI agents."""

__version__ = "0.1.0"

from .db import init_db, get_connection
from .embeddings import embed, embed_batch
from .router import route
from .dreaming import dream
