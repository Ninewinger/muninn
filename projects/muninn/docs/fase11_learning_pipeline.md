# Muninn — Fase 11: Pipeline de Escritura (Memoria Episódica)

## Contexto del Proyecto

Muninn es un sistema de memoria semántica para un agente IA (nanobot). Ya tiene funcionando:

- **15 peers** con **72 facetas** en SQLite (`D:\github\muninn\muninn.db`)
- **Routing** funcional: `router_v2.py` — embeddea texto → cosine similarity → reranker → top-k peers
- **Runtime Context** funcionando: nanobot inyecta contexto Muninn en CADA mensaje via import directo de `router_v2.py` en `context.py`
- **Embeddings**: Qwen3-Embedding-8B (1024d, GPU)
- **Reranker**: bge-reranker-v2-m3
- **DB**: SQLite con sqlite-vec, schema en `schema_v2.sql`

## Qué FALTA (este issue)

Muninn hoy es **solo lectura**. Los 15 peers y 72 facetas son estáticos (sembrados en `seed_v2.py`). No hay forma de que el sistema **aprenda nueva información** automáticamente.

Necesitamos un **pipeline de escritura** que:

1. **Detecte** info nueva en conversaciones
2. **Decida** si crear faceta nueva, actualizar existente, o crear peer nuevo
3. **Escriba** en la DB (embeddings incluidos)
4. **Dreaming** consolide periódicamente (refinar, mergear, limpiar)

## Arquitectura Propuesta

```
nanobot detecta info nueva en conversación
  → Llama a POST endpoint o función directa
    → muninn/learning.py (NUEVO)
      1. Analizar qué info es nueva/relevante
      2. Decidir: crear faceta / actualizar faceta / crear peer
      3. Generar embedding (usar embeddings_v2.py existente)
      4. Guardar en SQLite (usar db.py existente)
      5. Retornar confirmación
```

### Archivos a crear/modificar:

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `muninn/learning.py` | **CREAR** | Módulo principal de escritura/aprendizaje |
| `muninn/api.py` | **MODIFICAR** | Agregar endpoints de escritura |
| `muninn/dreaming.py` | **MODIFICAR** | Consolidación mejorada (mergear facetas similares, limpiar huérfanas) |
| Tests | **CREAR** | `tests/test_learning.py` |

### `learning.py` — Funciones principales:

```python
def learn_fact(text: str, db_path: str, source: str = "conversation") -> dict:
    """
    Aprender un hecho nuevo de la conversación.
    
    1. Embeddear el texto
    2. Buscar si ya existe una faceta similar (cosine > 0.85)
       - Si existe: actualizar texto de la faceta (merge)
       - Si no existe: crear faceta nueva en el peer más relevante
    3. Si no hay peer relevante (score < 0.15 para todos): 
       - Crear peer nuevo con 1 faceta
    4. Generar embedding y guardar
    
    Returns: {"action": "created_facet"|"updated_facet"|"created_peer", "details": ...}
    """

def learn_batch(facts: list[str], db_path: str) -> list[dict]:
    """Procesar múltiples facts de una vez (para dreaming)."""

def forget_facet(facet_id: str, db_path: str) -> bool:
    """Eliminar una faceta (para limpieza manual o dreaming)."""

def merge_facets(facet_ids: list[str], db_path: str, merged_text: str) -> dict:
    """Mergear facetas similares en una sola."""
```

### Cómo nanobot llama al pipeline:

Desde el skill `muninn`, nanobot puede ejecutar:

```bash
python skills/muninn/muninn_hook.py --learn "Diego empezó a estudiar Rust esta semana"
```

O directamente via import en `context.py` (como ya funciona el routing).

## ⚠️ PROBLEMAS CONOCIDOS Y RESTRICCIONES

### NO hacer esto:

1. **NO levantar servidor Flask/FastAPI** — Muninn se integra via **import directo** de `router_v2.py`. No hay servidor HTTP corriendo. Todo es `from muninn.xxx import yyy`.

2. **NO usar API HTTP para integración con nanobot** — nanobot importa directamente:
   ```python
   # En nanobot/agent/context.py (YA PARCHEADO)
   from muninn.router_v2 import route_with_context_injection
   result = route_with_context_injection(text=text, db_path=..., top_k=3)
   ```
   
3. **NO cargar modelos de embedding nuevos** — Usar `embeddings_v2.py` que ya tiene Qwen3-8B (1024d) como singleton cacheado. Importar y usar `embed()` o `embed_batch()`.

4. **NO modificar `router_v2.py`** sin preservar la interfaz existente:
   - `route(text, db_path, top_k, context_hour, instruction_override, use_reranker)` → retorna `List[Dict]`
   - `route_with_context_injection(text, db_path, top_k, context_hour)` → retorna `str`
   
5. **NO usar `requests.post("localhost:8000/...")`** — La versión anterior de `context.py` hacía esto. Ya fue reemplazada por import directo.

6. **NO asumir que hay VRAM ilimitada** — GPU tiene 16GB, Qwen3-8B usa ~4.6GB, reranker ~560MB. Solo cargar modelos vía los singletons existentes.

### Hacer esto:

1. **Usar `embeddings_v2.py`** para generar embeddings: `from muninn.embeddings_v2 import embed, get_backend`
2. **Usar `db.py`** para conexiones DB: `from muninn.db import get_connection`
3. **Usar `router_v2.py`** para routing (ya existente): `from muninn.router_v2 import route`
4. **Schema**: Ver `schema_v2.sql` para estructura de tablas (peers, peer_facets, facet_embeddings, embedding_config)
5. **Tests**: Crear tests en `tests/` que usen una DB temporal (no la de producción)
6. **Env vars necesarias**: `HF_HOME=D:\hf_cache`, `HF_HUB_OFFLINE=1`

## Estructura de la DB (schema_v2.sql)

```sql
-- Peers (entidades conceptuales)
CREATE TABLE peers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    domain TEXT NOT NULL,
    representation TEXT,
    activation_threshold REAL DEFAULT 0.25,
    is_active INTEGER DEFAULT 1,
    activation_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Facetas (múltiples aspectos de un peer)
CREATE TABLE peer_facets (
    id TEXT PRIMARY KEY,
    peer_id TEXT NOT NULL REFERENCES peers(id),
    facet_type TEXT NOT NULL,
    text TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings (una por faceta)
CREATE TABLE facet_embeddings (
    facet_id TEXT PRIMARY KEY REFERENCES peer_facets(id),
    embedding BLOB NOT NULL,
    model_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Config
CREATE TABLE embedding_config (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

## Ejemplo de uso esperado

```python
from muninn.learning import learn_fact

# Caso 1: Info nueva que matchea peer existente
result = learn_fact(
    "Diego está aprendiendo Rust, le interesa systems programming",
    db_path="D:\\github\\muninn\\muninn.db"
)
# → {"action": "created_facet", "peer": "programacion", "facet_type": "intereses"}

# Caso 2: Info que actualiza faceta existente (similar)
result = learn_fact(
    "Diego tiene una RTX 5070 Ti con 16GB VRAM",
    db_path="D:\\github\\muninn\\muninn.db"  
)
# → {"action": "updated_facet", "facet_id": "xxx", "reason": "similar to existing"}

# Caso 3: Info completamente nueva
result = learn_fact(
    "Diego adoptó un gato llamado Milo",
    db_path="D:\\github\\muninn\\muninn.db"
)
# → {"action": "created_peer", "peer_name": "mascota_milo", "facet_type": "personal"}
```

## Requisitos de Testing

1. **DB temporal** — crear `:memory:` o archivo temporal, no tocar `muninn.db`
2. **Mocks de embeddings** — no cargar Qwen3-8B real en tests (4GB VRAM). Usar embeddings dummy.
3. **Test cases**:
   - Crear faceta en peer existente
   - Actualizar faceta similar
   - Crear peer nuevo para info no clasificable
   - Merge de facetas duplicadas
   - Limpieza de facetas huérfanas (dreaming)
   - Embedding se genera correctamente

## Entorno

- **PC:** Windows 11, Python 3.11, RTX 5070 Ti (16GB VRAM)
- **Repo:** `D:\github\muninn\`
- **DB producción:** `D:\github\muninn\muninn.db` (NO modificar en tests)
- **Modelos cache:** `D:\hf_cache`
- **Python:** `python` (no python3)
