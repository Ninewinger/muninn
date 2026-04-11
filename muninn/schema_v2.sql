-- ============================================================
-- MUNINN — Schema v0.2 — "Disco Elysium" Architecture
-- Motor: SQLite 3 + sqlite-vec + FTS5
-- Cambios v0.1 → v0.2:
--   - peer_facets: cada peer tiene N facetas con embeddings propios
--   - Multi-activación: N peers pueden activarse simultáneamente
--   - Dimensiones flexibles (configurable por modelo)
-- ============================================================

-- Load sqlite-vec extension
-- Windows: .load ./vec0
-- Linux/Mac: .load ./vec0

-- ----------------------------------------------------------
-- 1. PEERS — entidades autónomas (sombras, personas, proyectos, temas)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS peers (
    id              TEXT PRIMARY KEY,           -- ej: "sombra_rechazo"
    name            TEXT NOT NULL,              -- ej: "Sombra de Rechazo"
    type            TEXT NOT NULL,              -- sombra|persona|proyecto|tema|sistema
    domain          TEXT,                       -- ej: "Sombras", "Programacion", "Gym/Rutina"
    description     TEXT,                       -- descripción corta del territorio temático
    representation  TEXT,                       -- resumen vivo (inyectado cuando se activa)
    confidence      REAL DEFAULT 0.1,           -- 0.0-1.0, qué bien conocemos este peer
    activation_threshold REAL DEFAULT 0.25,     -- umbral de similitud para activarse (bajado de 0.65)
    level           REAL DEFAULT 1.0,           -- 0.0-2.0, peso basado en frecuencia/recencia
    max_activations INTEGER DEFAULT 3,          -- cuántas activaciones simultáneas permitir
    tags            TEXT DEFAULT '[]',          -- JSON array: ["relaciones","madre","rechazo"]
    is_active       INTEGER DEFAULT 1,          -- soft delete
    activation_count INTEGER DEFAULT 0,         -- veces activado (debug/tuning)
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now')),
    last_activated_at TEXT,
    metadata        TEXT DEFAULT '{}'           -- JSON: cualquier cosa extra
);

-- ----------------------------------------------------------
-- 1b. PEER_FACETS — facetas compuestas de cada peer
--     Cada peer tiene 3-5 facetas (emocional, fisico, social, contextual, tecnico)
--     Cada faceta tiene su propio embedding
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS peer_facets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    peer_id     TEXT NOT NULL REFERENCES peers(id) ON DELETE CASCADE,
    facet_type  TEXT NOT NULL,                  -- emocional|fisico|social|contextual|tecnico
    text        TEXT NOT NULL,                  -- texto descriptivo de la faceta (para embedding)
    weight      REAL DEFAULT 1.0,              -- peso de esta faceta dentro del peer
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_facets_peer ON peer_facets(peer_id);
CREATE INDEX IF NOT EXISTS idx_facets_type ON peer_facets(facet_type);

-- Embeddings de FACETAS (no de peers enteros)
-- Dimensiones: 1024 para Qwen3-Embedding (MRL truncateable)
-- Nota: cambiar FLOAT[1024] al modelo que se use
CREATE VIRTUAL TABLE IF NOT EXISTS facet_embeddings USING vec0(
    facet_id   INTEGER PRIMARY KEY,
    embedding  FLOAT[1024]
);

-- ----------------------------------------------------------
-- 2. MEMORIES — eventos, hechos, fragmentos de conocimiento
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,                  -- el contenido textual de la memoria
    type        TEXT NOT NULL,                  -- hecho|episodio|preference|patron|hipotesis
    source      TEXT DEFAULT 'conversation',    -- conversation|dreaming|obsidian|manual
    confidence  REAL DEFAULT 0.5,               -- certeza de esta memoria
    is_active   INTEGER DEFAULT 1,              -- soft delete (reemplazada, obsoleta)

    -- Contexto temporal
    occurred_at TEXT,                           -- cuándo pasó el evento real
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now')),

    -- Provenancia
    session_id  TEXT,                           -- qué conversación la generó
    source_channel TEXT,                        -- telegram|cli|dreaming|obsidian

    metadata    TEXT DEFAULT '{}'               -- JSON: datos extra (emociones detectadas, etc.)
);

-- Embeddings de memorias (búsqueda semántica)
CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
    memory_id  INTEGER PRIMARY KEY,
    embedding  FLOAT[1024]
);

-- Búsqueda full-text (FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    type,
    source,
    content='memories',
    content_rowid='id'
);

-- Vinculación many-to-many: memorias ↔ peers
CREATE TABLE IF NOT EXISTS memory_peers (
    memory_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    peer_id     TEXT NOT NULL REFERENCES peers(id) ON DELETE CASCADE,
    relevance   REAL DEFAULT 0.5,               -- qué tan relevante es esta memoria para este peer
    created_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (memory_id, peer_id)
);

-- ----------------------------------------------------------
-- 3. CONNECTIONS — relaciones entre peers
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS connections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    from_peer_id    TEXT NOT NULL REFERENCES peers(id) ON DELETE CASCADE,
    to_peer_id      TEXT NOT NULL REFERENCES peers(id) ON DELETE CASCADE,
    relation_type   TEXT NOT NULL,               -- conecta|activa|contradice|evoluciona_de
    strength        REAL DEFAULT 0.5,            -- 0.0-1.0, fuerza de la conexión
    description     TEXT,                        -- descripción de la relación
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(from_peer_id, to_peer_id, relation_type)
);

-- ----------------------------------------------------------
-- 4. SESSIONS — conversaciones indexadas
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,               -- ej: "2026-04-05_telegram_8281248569"
    channel     TEXT NOT NULL,                  -- telegram|cli
    chat_id     TEXT,                           -- ID del chat/usuario
    started_at  TEXT DEFAULT (datetime('now')),
    ended_at    TEXT,
    summary     TEXT,                           -- resumen de la sesión (generado en dreaming)
    event_count INTEGER DEFAULT 0,
    metadata    TEXT DEFAULT '{}'               -- JSON
);

-- ----------------------------------------------------------
-- 5. EVENTS — log de eventos (input al event stream)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT REFERENCES sessions(id),
    type        TEXT NOT NULL,                  -- user_message|bot_response|tool_call|tool_result|thinking
    content     TEXT NOT NULL,                  -- contenido del evento
    channel     TEXT,                           -- telegram|cli
    created_at  TEXT DEFAULT (datetime('now')),
    metadata    TEXT DEFAULT '{}'               -- JSON
);

-- Embeddings de eventos (para semantic router)
CREATE VIRTUAL TABLE IF NOT EXISTS event_embeddings USING vec0(
    event_id   INTEGER PRIMARY KEY,
    embedding  FLOAT[1024]
);

-- Tabla de activaciones: qué peers y facetas se activaron por qué evento
CREATE TABLE IF NOT EXISTS activations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    INTEGER NOT NULL REFERENCES events(id),
    peer_id     TEXT NOT NULL REFERENCES peers(id),
    facet_id    INTEGER REFERENCES peer_facets(id),  -- nueva: qué faceta disparó la activación
    similarity  REAL NOT NULL,                  -- score de similitud coseno
    bonus_level REAL DEFAULT 0.0,              -- bonus por nivel del peer
    bonus_context REAL DEFAULT 0.0,            -- bonus por contexto (hora, etc.)
    total_score REAL NOT NULL,                  -- score final
    activated_at TEXT DEFAULT (datetime('now'))
);

-- ----------------------------------------------------------
-- 6. EMBEDDING_CONFIG — configuración del modelo de embeddings
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS embedding_config (
    key         TEXT PRIMARY KEY,              -- ej: "model_name", "dimensions", "instruction"
    value       TEXT NOT NULL,                  -- valor
    updated_at  TEXT DEFAULT (datetime('now'))
);

-- Insert default config
INSERT OR IGNORE INTO embedding_config (key, value) VALUES
    ('model_name', 'Qwen/Qwen3-Embedding-8B'),
    ('dimensions', '1024'),
    ('instruction', 'Dado un mensaje de un usuario, identifica que dominio de su vida esta activando'),
    ('max_activations', '3'),
    ('default_threshold', '0.25');

-- ----------------------------------------------------------
-- 7. CONSOLIDATION_LOG — auditoría del dreaming
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS consolidation_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT DEFAULT 'running',     -- running|completed|failed
    memories_processed INTEGER DEFAULT 0,
    peers_updated   INTEGER DEFAULT 0,
    connections_found INTEGER DEFAULT 0,
    memories_added  INTEGER DEFAULT 0,
    memories_deleted INTEGER DEFAULT 0,
    errors          TEXT DEFAULT '[]',          -- JSON array de errores
    notes           TEXT                        -- notas del proceso
);

-- ----------------------------------------------------------
-- ÍNDICES
-- ----------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_occurred ON memories(occurred_at);
CREATE INDEX IF NOT EXISTS idx_memory_peers_peer ON memory_peers(peer_id);
CREATE INDEX IF NOT EXISTS idx_memory_peers_memory ON memory_peers(memory_id);
CREATE INDEX IF NOT EXISTS idx_peers_type ON peers(type);
CREATE INDEX IF NOT EXISTS idx_peers_active ON peers(is_active);
CREATE INDEX IF NOT EXISTS idx_peers_domain ON peers(domain);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_activations_peer ON activations(peer_id);
CREATE INDEX IF NOT EXISTS idx_activations_event ON activations(event_id);
CREATE INDEX IF NOT EXISTS idx_activations_facet ON activations(facet_id);
CREATE INDEX IF NOT EXISTS idx_connections_from ON connections(from_peer_id);
CREATE INDEX IF NOT EXISTS idx_connections_to ON connections(to_peer_id);
