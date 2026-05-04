# 🪶 Muninn

> *Semantic memory system for AI agents — inspired by depth psychology.*

**Local-first · Privacy-first · Multi-backend embeddings**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## What is Muninn?

Muninn is a persistent semantic memory system for AI agents. It gives your agent contextual memory that activates automatically based on meaning, not keywords.

In Norse mythology, **Huginn** (thought) and **Muninn** (memory) are Odin's two ravens. They fly across the world each day and report what they've seen. Muninn does the same for your AI — it observes, remembers, and surfaces the right context at the right time.

### Core concepts

- **Peers** — autonomous knowledge entities (projects, people, topics, concepts). Each peer has multiple *facets* (emotional, physical, social, technical) with their own embeddings.
- **Semantic routing** — incoming messages are compared against all peer facets. Peers above threshold activate and inject their representation into the agent's context.
- **Dreaming** — background consolidation that discovers patterns, updates peer representations, finds connections between peers, and decays irrelevant memories.
- **Hybrid search** — vector similarity (sqlite-vec) + full-text (FTS5) with optional cross-encoder reranking.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  CONTEXT LAYER                            │
│  Injected into agent when peers activate                  │
└──────────────────────────┬──────────────────────────────┘
                           │ semantic route
┌──────────────────────────▼──────────────────────────────┐
│                   ROUTER LAYER                            │
│  Strategy A/B (faceted) · Strategy C (composite)        │
│  Strategy hybrid · Optional reranker                     │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    PEER LAYER                             │
│  Peers with facets, representations, embeddings          │
│  Activation thresholds · Level bonuses · Context bonuses │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  STORAGE LAYER                            │
│  SQLite + sqlite-vec + FTS5                              │
│  Vector search · Full-text · Embedding config            │
└─────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| 🧠 **Semantic activation** | Peers activate based on meaning via facet embeddings |
| 🔗 **Typed connections** | Peers link to each other with weighted, described relationships |
| 💭 **Living representations** | Each peer has an evolving summary (updated during dreaming) |
| 🔄 **Memory lifecycle** | ADD / UPDATE / DELETE / NOOP — adaptive, no duplicates |
| 🌙 **Dreaming** | Background consolidation (embeddings + rules, no LLM needed) |
| 📉 **Healthy forgetting** | Decay + confidence thresholds → archive |
| 🔍 **3 search strategies** | Faceted (fast), Composite (rich), Hybrid (best of both) |
| 🔎 **Reranking** | Optional cross-encoder (local BGE) or OpenRouter Cohere Rerank |
| 🏠 **100% local** | Single SQLite file, no external databases |
| 🌍 **Multilingual** | Spanish, English, and any language via multilingual embeddings |
| 🪶 **Multiple embedding backends** | OpenRouter, Gemini, Qwen3, sentence-transformers |
| 📁 **Obsidian Vault sync** | Index notes directly from an Obsidian vault |
| 🧩 **Hermes plugin** | Built-in plugin for Hermes Agent with prefetch() |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [sqlite-vec](https://github.com/asg017/sqlite-vec) extension loaded in your Python environment

### Install

```bash
git clone https://github.com/Ninewinger/muninn.git
cd muninn
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure

Copy the template and fill in your API keys:

```bash
cp config/env.template .env
# Edit .env with your settings
```

Minimal `.env` for OpenRouter embeddings (recommended):

```env
# Embedding backend (OpenRouter recommended)
EMBEDDING_PROVIDER=openrouter
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# OpenRouter API key
OPENROUTER_API_KEY=sk-or-v1-***

# Database
DB_PATH=./muninn.db
```

### Run

```bash
uvicorn muninn.api:app --host 0.0.0.0 --port 8199
```

---

## API Usage

### Create a Peer

```bash
curl -X POST http://localhost:8199/api/v1/peers \
  -H "Content-Type: application/json" \
  -d '{
    "id": "project_alpha",
    "name": "Project Alpha",
    "type": "proyecto",
    "domain": "Game Development",
    "description": "Main project - game with emotional mechanics",
    "tags": ["game", "development", "indie"]
  }'
```

### Add a Facet to a Peer

```bash
curl -X POST http://localhost:8199/api/v1/peers/project_alpha/facets \
  -H "Content-Type: application/json" \
  -d '{
    "facet_type": "tecnico",
    "text": "Game design patterns, combat mechanics, emotional variables",
    "weight": 1.0
  }'
```

### Store a Memory

```bash
curl -X POST http://localhost:8199/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Decided to use Determination and Intuition as core emotional variables",
    "type": "hecho",
    "source": "conversation",
    "peer_ids": ["project_alpha"]
  }'
```

### Route a message (semantic activation)

```bash
curl -X POST http://localhost:8199/api/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I feel like the combat system needs more emotional weight",
    "top_k": 3,
    "strategy": "hybrid"
  }'
```

Response includes activated peers with their representations, similarity scores, and context bonuses. This is what gets injected into the agent's context.

### Search memories

```bash
curl -X POST http://localhost:8199/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "emotional variables in combat",
    "limit": 5
  }'
```

### Run Dreaming (Consolidation)

```bash
curl -X POST http://localhost:8199/api/v1/dream \
  -H "Content-Type: application/json" \
  -d '{"peer_ids": ["project_alpha"], "strategy": "composite"}'
```

### Event stream

Send an event (auto-routed to relevant peers):

```bash
curl -X POST http://localhost:8199/api/v1/events \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_001",
    "type": "user_message",
    "content": "I think the combat system needs more emotional weight",
    "channel": "cli"
  }'
```

---

## Database Schema

Muninn uses SQLite with `sqlite-vec` (vector search) and FTS5 (full-text search).

Core tables:
- `peers` — autonomous entities with domain, type, activation settings
- `peer_facets` — 3-5 facets per peer, each with its own embedding (emotional, physical, social, technical, contextual)
- `facet_embeddings` — vec0 virtual table for facet similarity search
- `memories` — individual facts, events, patterns linked to peers
- `memory_embeddings` — vec0 virtual table for memory search
- `memory_fts` — FTS5 index for full-text search
- `connections` — typed, weighted relationships between peers
- `events` — input stream (what the agent observes)
- `activations` — audit log of peer activations with scores
- `sessions` — conversation tracking
- `embedding_config` — model name, dimensions, instruction
- `consolidation_log` — dreaming audit trail

Full schema: [muninn/schema_v2.sql](muninn/schema_v2.sql)

---

## Configuration

See [config/env.template](config/env.template) for all options.

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `openrouter` | Backend: `openrouter`, `gemini`, `sentence-transformers`, `qwen` |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Model name for the selected backend |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector dimensions (must match model) |
| `DB_PATH` | `./muninn.db` | Path to SQLite database file |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8199` | Server port |
| `LLM_API_URL` | — | For dreaming with LLM (optional, dreaming works without it) |
| `DECAY_RATE` | `0.95` | Memory confidence multiplier per day |
| `MIN_CONFIDENCE` | `0.1` | Below this → archived |
| `REINFORCE_BOOST` | `0.1` | Confidence boost when accessed |

---

## Embedding Backends

Muninn supports multiple embedding backends configured via `.env`:

| Backend | Provider | Config | Dimensions |
|---------|----------|--------|------------|
| **OpenRouter** | `openrouter` | `EMBEDDING_MODEL=openai/text-embedding-3-small` | 1536 |
| **Gemini** | `gemini` | `EMBEDDING_MODEL=text-embedding-004` | 768 |
| **Qwen3** | `qwen` | `EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B` | 1024 |
| **Sentence-Transformers** | `sentence-transformers` | `EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2` | 384 |

OpenRouter is recommended for production. Gemini is a good free alternative.

---

## Hermes Agent Integration

Muninn has a native plugin for [Hermes Agent](https://github.com/NousResearch/hermes) that:
1. **Prefetches context** — calls Muninn's route endpoint before each turn, injects activated peer representations into the agent's context
2. **Logs sessions** — sends session_end events to Muninn for dreaming
3. **Exposes tools** — `muninn_search`, `muninn_add_memory`, `muninn_list_peers`

### Installation

The plugin lives in `~/.hermes/plugins/muninn/`. Point Hermes to it in config:

```yaml
# config.yaml
plugins:
  - path: ~/.hermes/plugins/muninn
```

### Architecture

```
┌──────────────┐     HTTP      ┌──────────────┐
│  Hermes      │ ──────────►   │  Muninn       │
│  Agent       │    route()    │  FastAPI      │
│              │ ◄──────────   │  Server       │
│  Plugin      │  peer context │  :8199        │
│  muninn/     │               │               │
└──────────────┘               └──────────────┘
```

### How it works

1. **`prefetch()` hook** — before each user message, Hermes calls `GET /api/v1/route?text=<user_message>`. Activated peers with scores > threshold are injected as contextual preamble.

2. **`on_session_end` hook** — when a session ends, events are sent to Muninn for dreaming consolidation.

3. **Custom tools** — three tools registered automatically:
   - `muninn_search(query, top_k=5)` — semantic search across all memories
   - `muninn_add_memory(content, tags, peer_id, memory_type)` — store a memory
   - `muninn_list_peers()` — list all peers and their descriptions

### Example plugin.yaml

```yaml
name: muninn
version: 0.1.0
description: "Muninn — memory system with semantic routing and dreaming"
hooks:
  - on_session_end
  - on_pre_compress
```

> **Note:** The plugin currently uses `on_session_end` and `on_pre_compress` hooks. Prefetch integration (automatic context injection before each turn) is planned but requires Hermes to support the `prefetch` hook natively. As a workaround, the agent can manually call `muninn_search()` and Muninn's route endpoint.

---

## Semantic Router Strategies

| Strategy | Description | Speed | Quality |
|----------|-------------|-------|---------|
| **faceted** (A/B) | Compare query against each facet embedding individually | Fastest | Good |
| **composite** (C) | Build rich composite text per peer, embed once, compare | Medium | Better |
| **hybrid** | Blend faceted + composite scores for best of both | Medium | Best |

The router also supports:
- **Context bonuses** — boosts scores based on time of day, recency, and frequency
- **Reranking** — optional cross-encoder reranking via OpenRouter Cohere Rerank v3.5 or local BGE-reranker-v2-m3
- **Activation thresholds** — per-peer configurable minimum similarity scores

---

## Dreaming (Consolidation)

Dreaming is Muninn's background consolidation process. It runs periodically and:

1. Reviews events since last consolidation
2. Classifies events by relevance to each peer (via facet embeddings)
3. Creates new memories linked to peers
4. Updates peer representations with new insights
5. Discovers connections between peers that co-activate
6. Applies memory decay (confidence → thresholds → archive)
7. Detects patterns across related memories

Dreaming works **without an LLM** — it uses embeddings + rules, making it fast and cheap. An optional LLM can be configured for richer pattern detection.

---

## Project Status

Muninn is actively developed. Current state (v0.2 — "Disco Elysium" architecture):

- ✅ FastAPI server with full CRUD for peers, facets, memories, events, sessions, connections
- ✅ Semantic router with 3 strategies (faceted, composite, hybrid)
- ✅ Multi-backend embeddings (OpenRouter, Gemini, Qwen3, sentence-transformers)
- ✅ Dreaming/consolidation (embeddings + rules)
- ✅ Hermes Agent plugin with tools and session hooks
- ✅ Reranking (OpenRouter API + local cross-encoder)
- ✅ Context bonus system (time, frequency, recency)
- ✅ Obsidian Vault indexer (sync notes to Muninn memories)
- ✅ Feedback loop integration
- ✅ Memory lifecycle (ADD/UPDATE/DELETE/NOOP with decay)

---

## Why "Muninn"?

> *Huginn ok Muninn flúga hverjan dag*
> *Jörmungrund yfir;*
> *óumk ek of Hugin, at hann aftr né komiț,*
> *þó sjástumk meirr um Muninn.*

> *Hugin and Munin fly each day*
> *over the spacious earth;*
> *I fear for Hugin, that he come not back,*
> *yet more anxious am I for Munin.*

— Grímnismál, stanza 20

Huginn (thought) is the active agent — thinking, reasoning, responding. **Muninn is the memory system** — observing, remembering, surfacing context. The agent fears losing its memory more than losing its thoughts.

The architecture is inspired by Jungian psychology: **the unconscious is not a storage room, it's an active process.** Memory isn't about storing and retrieving data — it's about what *emerges* when the right context appears.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Support

If Muninn is useful to you:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support_me-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/ninewinger)

---

*Built with 🪶 by [Diego Vergara](https://github.com/Ninewinger)*