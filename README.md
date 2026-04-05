# 🪶 Muninn

> *Memory system for AI agents — inspired by depth psychology.*

**Local-first · Privacy-first · Zero cloud dependency**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is Muninn?

Muninn is a persistent memory system for AI agents. It gives your agent:

- **Semantic memory** — store and retrieve facts, events, patterns by meaning, not just keywords
- **Autonomous entities (Peers)** — organize knowledge around topics, people, projects that activate when relevant
- **Dreaming/Consolidation** — automatic background process that finds patterns, updates representations, detects contradictions
- **Healthy forgetting** — memory decay with reinforcement, like a real mind

### The idea

In Norse mythology, **Huginn** (thought) and **Muninn** (memory) are Odin's two ravens. They fly across the world each day and return to report what they've seen.

Muninn does the same for your AI agent — it observes everything, remembers what matters, and surfaces the right context at the right time.

### The philosophy

Muninn is designed around a principle from Jungian psychology: **the unconscious is not a storage room, it's an active process.**

Memory isn't just about storing and retrieving data. It's about:
- **What emerges** when the right context appears
- **What connects** to what (and why)
- **What fades** when it's no longer relevant
- **What consolidates** through reflection

This leads to a unique architecture where knowledge is organized into **Peers** — autonomous entities that activate when semantically relevant, maintain their own evolving representation, and connect to each other.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CONTEXT LAYER                         │
│  What the agent sees — representations of active Peers   │
└──────────────────────────┬──────────────────────────────┘
                           │ only active peers
┌──────────────────────────▼──────────────────────────────┐
│                     PEER LAYER                           │
│  Autonomous entities with:                               │
│    - Activation embedding (semantic territory)           │
│    - Living representation (evolving summary)            │
│    - Linked memories (raw data)                          │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   STORAGE LAYER                          │
│  SQLite + sqlite-vec + FTS5                              │
│  Hybrid search: vector similarity + full-text            │
└─────────────────────────────────────────────────────────┘
```

### How Peers work

A Peer is an entity that represents a topic, person, project, or concept. It has:

1. **Activation embedding** — a vector defining its "semantic territory"
2. **Representation** — a pre-built text summary, updated during dreaming
3. **Connections** — typed, weighted links to other peers
4. **Linked memories** — individual facts, events, patterns

When a new event comes in, Muninn's **semantic router** compares it against all peers' embeddings. If similarity exceeds the peer's threshold, it activates — injecting its representation into the agent's context.

```
User: "I felt weird today, not sure why"
        │
        ▼
Semantic Router evaluates all peers:
  Peer "Anxiety":     0.78 ✅ ACTIVATED
  Peer "Relationships": 0.45 (dormant)
  Peer "Gym":         0.10 (dormant)
        │
        ▼
Inject into agent context:
  "[Peer: Anxiety | confidence 0.6]
   User's anxiety manifests especially with AI topics..."
```

### Healthy forgetting

Muninn doesn't keep everything forever. It uses a **decay + reinforcement** system:

- All memories have a `confidence` score that decays over time
- When a memory is accessed or reinforced, confidence goes up
- During dreaming, the LLM evaluates which memories to compress, archive, or reinforce
- Nothing is truly deleted — archived memories can be reactivated if relevant again
- **Versioning with decay** — when a fact is updated, the old version isn't overwritten. It enters the decay system. If a peer finds a pattern in the old value ("weight went up too fast = injury risk"), it gets reinforced. If nobody touches it, it fades to archive.

### Dreaming (Consolidation)

A background process that runs periodically:

1. Reviews new memories since last consolidation
2. Detects patterns across memories of the same peer
3. Updates the peer's representation with new insights
4. Discovers new connections between peers
5. Cleans obsolete memories
6. Adjusts activation thresholds based on real usage

Uses a lightweight LLM (any OpenAI-compatible API) for reasoning.

---

## Features

| Feature | Description |
|---------|-------------|
| 🧠 **Semantic activation** | Peers activate based on meaning, not keywords |
| 🔗 **Typed connections** | Peers link to each other with weighted, described relationships |
| 💭 **Living representations** | Each peer has an evolving summary, not just raw data |
| 🔄 **Memory lifecycle** | ADD / UPDATE / DELETE / NOOP — adaptive, no duplicates |
| 🌙 **Dreaming** | Background consolidation with LLM reasoning |
| 📉 **Healthy forgetting** | Decay + reinforcement + archive |
| 🔍 **Hybrid search** | Vector similarity + full-text (FTS5) |
| 🏠 **100% local** | Single SQLite file, no cloud, no external DB |
| 🌍 **Multilingual** | Supports Spanish, English, and any language via multilingual embeddings |
| 🪶 **Lightweight** | Runs on 8GB RAM, CPU-only, no GPU needed |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [sqlite-vec](https://github.com/asg017/sqlite-vec) extension

### Install

```bash
git clone https://github.com/Ninewinger/muninn.git
cd muninn
pip install -r requirements.txt
```

### Run

```bash
uvicorn muninn.api:app --host 0.0.0.0 --port 8000
```

### Use

```bash
# Create a peer
curl -X POST http://localhost:8000/api/v1/peers \
  -H "Content-Type: application/json" \
  -d '{
    "id": "project_alpha",
    "name": "Project Alpha",
    "type": "proyecto",
    "description": "Main project - game development with emotional mechanics",
    "tags": ["game", "development", "indie"]
  }'

# Store a memory
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Decided to use Determination and Intuition as core emotional variables",
    "type": "hecho",
    "source": "conversation",
    "peer_ids": ["project_alpha"]
  }'

# Send an event (triggers semantic router)
curl -X POST http://localhost:8000/api/v1/events \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_001",
    "type": "user_message",
    "content": "Im thinking about changing the combat system",
    "channel": "cli"
  }'

# Search
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "emotional variables in combat",
    "limit": 5
  }'
```

---

## Database Schema

Muninn uses SQLite with three extensions:
- **sqlite-vec** — vector similarity search
- **FTS5** — full-text search
- **Standard SQLite** — relational data

Core tables:
- `peers` — autonomous entities with embeddings and representations
- `memories` — individual facts, events, patterns (with lifecycle)
- `connections` — typed, weighted relationships between peers
- `events` — input stream (what the agent observes)
- `activations` — audit log of peer activations
- `sessions` — conversation tracking

See [schema.sql](muninn/schema.sql) for full details.

---

## Configuration

Create a `.env` file (or set environment variables):

```env
# Embedding model (local)
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMENSIONS=384

# Or use a local GGUF model via llama.cpp
# EMBEDDING_MODEL_PATH=./models/nomic-embed-text-v1.5.Q8_0.gguf

# LLM for dreaming/consolidation (any OpenAI-compatible API)
LLM_API_URL=https://api.z.ai/api/paas/v4/chat/completions
LLM_MODEL=glm-4.7-flash
LLM_API_KEY=your-key-here

# Server
HOST=0.0.0.0
PORT=8000

# Database
DB_PATH=./muninn.db

# Memory decay
DECAY_RATE=0.95        # multiplier per day
MIN_CONFIDENCE=0.1     # below this → archive
REINFORCE_BOOST=0.1    # boost when accessed
```

---

## Roadmap

### MVP (v0.1) — Current
- [x] Architecture design
- [x] Database schema (SQLite + sqlite-vec + FTS5)
- [x] API endpoints specification
- [ ] FastAPI implementation
- [ ] Semantic router with cosine similarity
- [ ] Basic peer CRUD + activation
- [ ] 5-10 seed peers from existing data
- [ ] Python client library

### v0.2 — Intelligence
- [ ] Memory lifecycle (ADD/UPDATE/DELETE/NOOP)
- [ ] Automatic event stream ingestion
- [ ] Dreaming/consolidation with LLM
- [ ] Automatic peer creation from patterns
- [ ] Connection discovery

### v0.3 — Connections
- [ ] Peer-to-peer activation chains
- [ ] Dynamic representations (LLM-generated during dreaming)
- [ ] Connection graph visualization
- [ ] Sync between multiple machines

### v0.4 — Integration
- [ ] MCP server protocol
- [ ] LangChain/LlamaIndex integration
- [ ] nanobot skill package
- [ ] Obsidian vault sync (peers = note folders)

---

## Why "Muninn"?

In Norse mythology, Huginn (Old Norse for "thought") and Muninn ("memory" or "mind") are a pair of ravens that fly all over the world, Midgard, and bring information to the god Odin.

> *Huginn ok Muninn flúga hverjan dag<br>
> Jörmungrund yfir;<br>
> óumk ek of Hugin, at hann aftr né komiț,<br>
> þó sjástumk meirr um Muninn.*

> *Hugin and Munin fly each day<br>
> over the spacious earth;<br>
> I fear for Hugin, that he come not back,<br>
> yet more anxious am I for Munin.*

— Grímnismál, stanza 20

Huginn is the active agent — thinking, reasoning, responding. **Muninn is the memory system** — observing, remembering, surfacing context. The agent fears losing its memory more than losing its thoughts.

---

## Contributing

Contributions are welcome! This project started as a personal tool and grew into something others might find useful.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Support

If Muninn is useful to you, consider supporting its development:

[![Buy Me a Coffee](https://img.shields.io/badge/Buy_Me_a_Coffee-support-yellow?logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/ninewinger)

Or visit: https://buymeacoffee.com/ninewinger

---

*Built with 🪶 by [Diego Vergara](https://github.com/Ninewinger) and [nanobot](https://github.com/HKUDS/nanobot) 🐈*
