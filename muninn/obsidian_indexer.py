"""Muninn Obsidian Vault Indexer — Fase 6.

Indexes Obsidian notes as Muninn memories, mapped to peers by folder.
Supports incremental indexing (only new/modified notes).

Peer mapping (folder -> Muninn peer):
  - asistente/        -> hermes_sistema
  - asistente/salud   -> salud_gym (?)
  - asistente/suenos  -> suenos_analisis
  - asistente/valle_alto -> valle_alto
  - asistente/proyecto_juego -> proyecto_juego
  - aprendizaje/      -> peer_learning (?)
  - autoevaluaciones/ -> autoevaluacion
  - comunicacion/     -> relaciones_personales (?)
  - filosofia/        -> sombra_*
  - ia/               -> peer_herramientas (?)
  - libros/           -> peer_learning (?)
  - negocios/         -> finanzas_patrimonio
  - noticiario/       -> hermes_sistema
  - personas/         -> relaciones_personales
  - programacion/     -> programacion
  - proyecto_juego/   -> proyecto_juego
  - proyecto_valle_alto -> valle_alto
  - psicologia/       -> sombra_*
  - reflexiones/      -> sombra_*
  - salud/            -> gym_rutina (?)
  - sombras/          -> sombra_*
  - suplementos/      -> salud_gym (?)
  - ocio/             -> casual_social (?)
  - sincronicidades/  -> sombra_*
  - hardware/         -> peer_herramientas (?)

Folders not indexed:
  - Excalidraw/ (images)
  - _archivados/ (backup)
  - _meta/ (vault meta)
  - plantillas/ (templates)
"""
import os, json, time, hashlib, sqlite3, struct
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from .db import get_connection
from .embeddings_v2 import embed, get_backend
from .feedback_loop import _correction_cache

VAULT_PATH = "/mnt/d/Documentos/Obsidian Vault"

# Map top-level folders to Muninn peers
# Some folders map to specific peers, others are generic
FOLDER_PEER_MAP = {
    "asistente": "hermes_sistema",
    "asistente/suenos": "suenos_analisis",
    "asistente/valle_alto": "valle_alto",
    "asistente/proyecto_juego": "proyecto_juego",
    "asistente/supervisor": "hermes_sistema",
    "asistente/salud": "gym_rutina",
    "aprendizaje": "peer_skills",
    "autoevaluaciones": "autoevaluacion",
    "comunicacion": "relaciones_personales",
    "filosofia": "sombra_fortaleza",
    "hardware": "peer_herramientas",
    "ia": "peer_herramientas",
    "libros": "peer_skills",
    "negocios": "finanzas_patrimonio",
    "noticiario": "hermes_sistema",
    "ocio": "casual_social",
    "personas": "relaciones_personales",
    "programacion": "programacion",
    "programación": "programacion",
    "proyecto_juego": "proyecto_juego",
    "proyecto_valle_alto": "valle_alto",
    "psicologia": "sombra_fortaleza",
    "reflexiones": "sombra_fortaleza",
    "salud": "gym_rutina",
    "sincronicidades": "sombra_angel_atardecer",
    "sombras": "sombra_muerte",
    "suplementos": "gym_rutina",
}

# Tracking file for indexed notes
INDEX_TRACK_PATH = os.path.join(os.path.dirname(__file__), "obsidian_index_state.json")


def _get_index_state() -> dict:
    """Load the index state (file -> hash mapping)."""
    if os.path.exists(INDEX_TRACK_PATH):
        try:
            with open(INDEX_TRACK_PATH) as f:
                return json.load(f)
        except:
            pass
    return {}


def _save_index_state(state: dict):
    """Save the index state."""
    os.makedirs(os.path.dirname(INDEX_TRACK_PATH), exist_ok=True)
    with open(INDEX_TRACK_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _file_hash(path: str) -> str:
    """Compute a quick hash of file content to detect changes."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""


def _get_peer_for_path(rel_path: str) -> str:
    """Determine which Muninn peer a note belongs to by its path."""
    parts = rel_path.replace("\\", "/").split("/")
    
    # Check exact folder matches (longest prefix first)
    for folder in sorted(FOLDER_PEER_MAP.keys(), key=len, reverse=True):
        if rel_path.startswith(folder + "/") or rel_path == folder:
            return FOLDER_PEER_MAP[folder]
    
    # Fallback: use first folder
    if parts[0] in FOLDER_PEER_MAP:
        return FOLDER_PEER_MAP[parts[0]]
    
    return "hermes_sistema"  # default


def _extract_title(content: str, filepath: str) -> str:
    """Extract title from markdown frontmatter or filename."""
    lines = content.split("\n")
    for line in lines:
        if line.startswith("title:"):
            return line.replace("title:", "", 1).strip().strip('"').strip("'")
        if line.startswith("# "):
            return line.replace("# ", "", 1).strip()
        if line == "---":
            break
    return Path(filepath).stem


def _extract_tags(content: str) -> List[str]:
    """Extract #tags from markdown content."""
    tags = []
    for word in content.split():
        if word.startswith("#") and len(word) > 1:
            tag = word.split("[")[0].split(")")[0].strip("#,.;:!?")
            if tag and tag not in tags:
                tags.append(tag)
    return tags[:10]  # limit


def _clean_content(content: str, max_len: int = 4000) -> str:
    """Clean markdown content for embedding: strip code blocks, reduce to meaningful text."""
    # Remove code blocks
    lines = content.split("\n")
    clean = []
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.startswith("---"):
            continue
        clean.append(line)
    
    text = "\n".join(clean).strip()
    return text[:max_len]


def index_note(filepath: str, db_path: str = None, force: bool = False) -> dict:
    """Index a single Obsidian note as a Muninn memory."""
    abs_path = filepath if filepath.startswith("/") else os.path.join(VAULT_PATH, filepath)
    rel_path = os.path.relpath(abs_path, VAULT_PATH)
    
    if not os.path.exists(abs_path):
        return {"status": "skipped", "reason": "not_found"}
    
    if not abs_path.endswith(".md"):
        return {"status": "skipped", "reason": "not_markdown"}
    
    # Skip non-content dirs
    if rel_path.startswith((".", "_archivados", "_meta", "plantillas", "Excalidraw")):
        return {"status": "skipped", "reason": "excluded_dir"}
    
    # Read file
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return {"status": "error", "reason": str(e)}
    
    if not content.strip():
        return {"status": "skipped", "reason": "empty"}
    
    # Check hash for incremental indexing
    file_md5 = _file_hash(abs_path)
    state = _get_index_state()
    
    if not force and rel_path in state and state[rel_path] == file_md5:
        return {"status": "skipped", "reason": "unchanged"}
    
    # Prepare memory content
    title = _extract_title(content, abs_path)
    tags = _extract_tags(content)
    clean_text = _clean_content(content)
    peer_id = _get_peer_for_path(rel_path)
    
    if not clean_text:
        return {"status": "skipped", "reason": "empty_after_clean"}
    
    # Store in Muninn
    conn = get_connection(db_path)
    try:
        # Check if already exists (by checking source metadata)
        existing = conn.execute(
            "SELECT id FROM memories WHERE source = 'obsidian' AND metadata LIKE ?",
            [json.dumps({"filepath": rel_path})[:80] + "%"]
        ).fetchone()
        
        if existing:
            # Update existing memory
            memory_id = existing["id"]
            conn.execute(
                """UPDATE memories SET content = ?, confidence = 0.7, updated_at = datetime('now'),
                   metadata = ? WHERE id = ?""",
                [clean_text, json.dumps({"filepath": rel_path, "title": title, "tags": tags, "size": len(content)}), memory_id]
            )
            action = "updated"
        else:
            # Insert new memory
            conn.execute(
                """INSERT INTO memories (content, type, source, confidence, metadata, session_id, occurred_at)
                   VALUES (?, 'obsidian_note', 'obsidian', 0.7, ?, 'vault_indexer', datetime('now'))""",
                [clean_text, json.dumps({"filepath": rel_path, "title": title, "tags": tags, "size": len(content)})]
            )
            memory_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            action = "created"
        
        # Embed memory
        try:
            vector = embed(clean_text, is_query=False)
            vec_bytes = struct.pack(f"{len(vector)}f", *vector)
            conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", [memory_id])
            conn.execute("INSERT INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)", [memory_id, vec_bytes])
        except Exception as emb_err:
            print(f"  [indexer] Embedding skipped for {rel_path}: {emb_err}", flush=True)
            # Try to free GPU memory after OOM
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
        
        # Link to peer
        conn.execute("DELETE FROM memory_peers WHERE memory_id = ?", [memory_id])
        conn.execute("INSERT INTO memory_peers (memory_id, peer_id, relevance) VALUES (?, ?, ?)",
                     [memory_id, peer_id, 0.8])
        
        # Also FTS index (for keyword search)
        conn.execute("DELETE FROM memory_fts WHERE rowid = ?", [memory_id])
        fts_title = title or Path(rel_path).stem
        conn.execute(
            "INSERT INTO memory_fts (rowid, content, type, source) VALUES (?, ?, 'obsidian_note', 'obsidian')",
            [memory_id, f"{fts_title}\n{clean_text[:3000]}"]
        )
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        return {"status": "error", "reason": str(e), "file": rel_path}
    finally:
        conn.close()
    
    # Update index state
    state[rel_path] = file_md5
    _save_index_state(state)
    
    return {"status": action, "peer": peer_id, "title": title, "tags": tags[:3]}


def index_vault(db_path: str = None, force: bool = False, limit: int = None) -> dict:
    """Index all Obsidian notes.
    
    Args:
        db_path: Muninn database path
        force: Re-index even unchanged files
        limit: Max files to index (for testing)
    
    Returns:
        Summary dict
    """
    print("=" * 60)
    print("  MUNINN — Indexing Obsidian Vault")
    print(f"  Vault: {VAULT_PATH}")
    print("=" * 60)
    
    results = {"created": 0, "updated": 0, "skipped": 0, "errors": 0, "files": []}
    
    # Walk all .md files
    md_files = []
    for root, dirs, files in os.walk(VAULT_PATH):
        # Skip hidden dirs and excluded
        rel_root = os.path.relpath(root, VAULT_PATH)
        if rel_root.startswith((".", "_archivados", "_meta", "plantillas", "Excalidraw")):
            continue
        for f in files:
            if f.endswith(".md"):
                md_files.append(os.path.join(root, f))
    
    total = len(md_files)
    print(f"\n  Found {total} .md files to process")
    
    if limit:
        md_files = md_files[:limit]
        print(f"  Limiting to {limit} files")
    
    for i, fp in enumerate(md_files):
        result = index_note(fp, db_path, force)
        status = result["status"]
        
        if status == "created":
            results["created"] += 1
        elif status == "updated":
            results["updated"] += 1
        elif status == "skipped":
            results["skipped"] += 1
        elif status == "error":
            results["errors"] += 1
            results["files"].append(result.get("file", fp))
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{total} (created={results['created']}, skipped={results['skipped']}, errors={results['errors']})")
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
    
    print(f"\n  DONE: {results['created']} created, {results['updated']} updated, {results['skipped']} skipped, {results['errors']} errors")
    
    return results


def search_vault(query: str, top_k: int = 5, db_path: str = None) -> List[dict]:
    """Search indexed Obsidian notes via Muninn routing."""
    from .router_v2 import route
    
    activations = route(query, db_path=db_path, top_k=top_k)
    
    # Filter to only obsidian memories
    results = []
    conn = get_connection(db_path)
    for act in activations:
        peer_id = act.get("peer_id", "")
        # Get memories linked to this peer
        memories = conn.execute(
            """SELECT m.id, m.content, m.metadata FROM memories m
               JOIN memory_peers mp ON m.id = mp.memory_id
               WHERE mp.peer_id = ? AND m.source = 'obsidian' AND m.is_active = 1
               ORDER BY mp.relevance DESC
               LIMIT 3""",
            [peer_id]
        ).fetchall()
        for m in memories:
            meta = json.loads(m["metadata"]) if m["metadata"] else {}
            results.append({
                "peer": peer_id,
                "title": meta.get("title", "?"),
                "filepath": meta.get("filepath", "?"),
                "score": act.get("total_score", 0),
            })
    conn.close()
    
    return results


def get_index_stats(db_path: str = None) -> dict:
    """Get stats about indexed Obsidian notes."""
    conn = get_connection(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE source = 'obsidian' AND is_active = 1"
    ).fetchone()[0]
    
    by_peer = conn.execute(
        """SELECT mp.peer_id, COUNT(*) as c FROM memories m
           JOIN memory_peers mp ON m.id = mp.memory_id
           WHERE m.source = 'obsidian' AND m.is_active = 1
           GROUP BY mp.peer_id ORDER BY c DESC"""
    ).fetchall()
    
    conn.close()
    
    state = _get_index_state()
    
    return {
        "total_indexed": count,
        "files_tracked": len(state),
        "by_peer": {r["peer_id"]: r["c"] for r in by_peer},
    }


def test():
    """Quick test: index a single file."""
    test_file = "/mnt/d/Documentos/Obsidian Vault/ia/muninn.md"
    if os.path.exists(test_file):
        result = index_note(test_file, force=True)
        print(f"Test index: {json.dumps(result, indent=2)}")
    else:
        print("Test file not found, trying to find any note...")
        for root, dirs, files in os.walk(VAULT_PATH):
            for f in files:
                if f.endswith(".md") and not f.startswith("."):
                    test_file = os.path.join(root, f)
                    result = index_note(test_file, force=True)
                    print(f"Test index {f}: {json.dumps(result, indent=2)}")
                    return
        print("No .md files found!")


if __name__ == "__main__":
    test()