"""
Dreaming LLM — Extracción de hechos durables con LLM + curiosidad.

Se ejecuta DESPUÉS de que el dreaming base procesó eventos (activaciones).
Usa un LLM (primario Zhipu GLM 5.1, fallback NVIDIA NIM) para:
1. Extraer hechos durables de eventos de conversación
2. Generar preguntas de curiosidad desde peers con datos incompletos

Totalmente opcional — dreaming funciona igual sin este módulo.
"""

import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Configuración ───────────────────────────────────────────

LLM_CONFIG = {
    "primary": {
        "api_url": os.environ.get(
            "LLM_API_URL",
            "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
        ),
        "api_key": os.environ.get("ZHIPU_API_KEY") or os.environ.get("LLM_API_KEY", ""),
        "model": os.environ.get("LLM_MODEL", "glm-5.1"),
    },
    "fallback": {
        "api_url": os.environ.get(
            "LLM_FALLBACK_URL",
            "https://integrate.api.nvidia.com/v1/chat/completions"
        ),
        "api_key": os.environ.get("NVIDIA_API_KEY", ""),
        "model": os.environ.get("LLM_FALLBACK_MODEL", "z-ai/glm-5.1"),
    },
}

# Peers que requieren al menos N memorias antes de ser considerados "completos"
MIN_MEMORIES_PER_PEER = 3

# Umbral de confianza para guardar memoria extraída por LLM
MIN_CONFIDENCE_FOR_MEMORY = 0.6

# ══════════════════════════════════════════════════════════════
# LLM CLIENT (resiliente con fallback)
# ══════════════════════════════════════════════════════════════


def _call_llm(prompt: str, system_prompt: str = "") -> Optional[str]:
    """
    Llamar al LLM con fallback resiliente.
    
    Intenta primario (Zhipu GLM 5.1) → si falla → fallback (NVIDIA).
    Reintenta cada provider 2 veces con backoff exponencial.
    """
    providers = [
        ("primary", LLM_CONFIG["primary"]),
        ("fallback", LLM_CONFIG["fallback"]),
    ]

    for provider_name, config in providers:
        if not config["api_key"]:
            logger.warning(f"LLM {provider_name}: no API key configured")
            continue

        for attempt in range(2):  # max 2 attempts per provider
            try:
                t0 = time.time()

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config['api_key']}",
                }

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                body = {
                    "model": config["model"],
                    "messages": messages,
                    "temperature": 0.1,  # bajo para extracción precisa
                    "max_tokens": 500,
                }

                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        config["api_url"],
                        headers=headers,
                        json=body,
                    )

                elapsed = time.time() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    logger.info(
                        f"LLM {provider_name} ({config['model']}) "
                        f"responded in {elapsed:.1f}s"
                    )
                    return content
                else:
                    logger.warning(
                        f"LLM {provider_name} attempt {attempt + 1}: "
                        f"HTTP {resp.status_code} - {resp.text[:200]}"
                    )
                    if attempt == 0:
                        time.sleep(1.0 * (attempt + 1))

            except Exception as e:
                logger.warning(
                    f"LLM {provider_name} attempt {attempt + 1} failed: {e}"
                )
                if attempt == 0:
                    time.sleep(1.0 * (attempt + 1))

    logger.error("All LLM providers failed - returning None")
    return None


# ══════════════════════════════════════════════════════════════
# EXTRACCIÓN DE HECHOS
# ══════════════════════════════════════════════════════════════


EXTRACTION_SYSTEM_PROMPT = """Eres un extractor de conocimiento durable.

Tu tarea: analizar eventos de conversación y determinar si contienen 
información estable sobre el usuario que merezca ser recordada.

REGLAS:
1. SOLO extraer hechos DUPLICABLES (preferencias, datos personales, 
   patrones de comportamiento, insights, decisiones importantes)
2. NO extraer: saludos, confirmaciones, información temporal, 
   fragmentos de conversación sin significado durable
3. Si hay un hecho, extraerlo LIMPIO — sin contexto conversacional
4. Asignar confianza alta (0.8+) solo si el usuario lo dijo explícitamente
5. Confianza media (0.6-0.7) para inferencias razonables
6. Si no hay hecho, responder {"has_fact": false}

Responde SOLO con JSON, nada más."""


def extract_facts_from_event(
    content: str,
    activated_peers: list,
) -> dict:
    """
    Analiza un evento de conversación y extrae hechos durables.
    
    Returns:
        dict con has_fact, fact_text, relevant_peer, confidence, fact_type
        o {"has_fact": false} si no hay hecho
    """
    # No tiene sentido analizar eventos muy cortos
    if len(content.strip()) < 30:
        return {"has_fact": False}

    peers_text = ", ".join(activated_peers[:5]) if activated_peers else "ninguno"

    json_template = '{"has_fact": true/false, "fact_text": "hecho extraído limpio (o vacío)", "relevant_peer": "peer_id más relevante (o vacío)", "confidence": 0.0-1.0, "fact_type": "preferencia|dato|insight|patron|none"}'
    prompt = f"""Analiza este evento de conversación:

"{content}"

Peers activados semánticamente: {peers_text}

Responde SOLO con JSON plano:
{json_template}"""

    result_text = _call_llm(prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT)

    if not result_text:
        return {"has_fact": False}

    # Parsear JSON de la respuesta (el LLM puede añadir markdown)
    try:
        # Limpiar posibles wrappers de markdown
        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            # Extraer bloque JSON de markdown
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            )
        result = json.loads(cleaned)
        return result
    except json.JSONDecodeError:
        logger.warning(f"LLM returned non-JSON: {result_text[:200]}")
        return {"has_fact": False}


# ══════════════════════════════════════════════════════════════
# GENERACIÓN DE PREGUNTAS DE CURIOSIDAD
# ══════════════════════════════════════════════════════════════


CURIOSITY_SYSTEM_PROMPT = """Eres un generador de preguntas curiosas e inteligentes.

Tienes acceso a peers (temas/conceptos sobre el usuario Diego).
Algunos peers tienen pocos datos registrados.

Genera 1 pregunta por peer que ayude a:
1. Llenar vacíos de conocimiento sobre ese tema
2. Entender mejor la perspectiva, preferencias o historia de Diego
3. Conectar conceptos entre peers diferentes

Las preguntas deben ser:
- Específicas (no genéricas como "¿qué piensas de X?")
- Curiosas (que motiven a responder, no deberes)
- Accionables (que Diego pueda responder en 1-2 minutos)

Responde SOLO con JSON array."""


def generate_curiosity_questions(
    conn,
    db_path: Optional[str] = None,
    max_questions: int = 3,
    curiosity_file: Optional[str] = None,
) -> list:
    """
    Analiza peers con pocos datos y genera preguntas de curiosidad.
    
    1. Busca peers con baja cobertura (< MIN_MEMORIES_PER_PEER memorias)
    2. Envía al LLM para generar preguntas relevantes
    3. Las escribe a cola-curiosidad.md (si curiosity_file está definido)
    
    Returns: lista de dicts {peer_id, pregunta}
    """
    if not _is_configured():
        logger.info("LLM no configurado - saltando generación de curiosidad")
        return []

    # 1. Encontrar peers con pocos datos
    peers = conn.execute(
        """
        SELECT p.id, p.name, p.description, p.domain,
               COUNT(mp.memory_id) as memory_count,
               p.activation_count
        FROM peers p
        LEFT JOIN memory_peers mp ON p.id = mp.peer_id
        WHERE p.is_active = 1
        GROUP BY p.id
        HAVING memory_count < ?
        ORDER BY p.activation_count DESC
        LIMIT ?
        """,
        [MIN_MEMORIES_PER_PEER, max_questions + 2],  # +2 por si algunos no generan
    ).fetchall()

    if not peers:
        logger.info("No se encontraron peers con datos incompletos")
        return []

    # 2. Preparar contexto para el LLM
    peers_context = "\n".join(
        f"- {p['id']} ({p['name']}): {p['description'][:100]} "
        f"— {p['memory_count']} memorias, {p['activation_count']} activaciones"
        for p in peers
    )

    prompt = f"""Analiza estos peers (temas/conceptos sobre Diego) que tienen 
pocos datos registrados y genera preguntas para conocerlo mejor:

{peers_context}

Para CADA peer, genera 1 pregunta que ayude a llenar el vacío.
Responde SOLO con JSON array:
[
  {{
    "peer_id": "peer_id",
    "pregunta": "pregunta específica y curiosa",
    "intencion": "qué queremos aprender con esta pregunta"
  }}
]"""

    result_text = _call_llm(prompt, system_prompt=CURIOSITY_SYSTEM_PROMPT)

    if not result_text:
        return []

    # Parsear JSON
    try:
        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )
        questions = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Curiosidad LLM returned non-JSON: {result_text[:200]}")
        return []

    # 3. Escribir al archivo de curiosidad
    if curiosity_file and questions:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            new_entries = "\n\n".join(
                f"### Pregunta (generada {today})\n"
                f"**Peer:** {q['peer_id']}\n"
                f"**Intención:** {q.get('intencion', 'Conocer más')}\n\n"
                f"{q['pregunta']}\n"
                for q in questions
            )

            # Leer archivo existente
            if os.path.exists(curiosity_file):
                with open(curiosity_file, "r") as f:
                    existing = f.read()
            else:
                existing = "# Cola de Curiosidad\n\n"

            # Insertar preguntas nuevas después del título
            # Buscar sección "## Pendientes" o crearla
            if "## Pendientes" in existing:
                # Insertar justo después del título Pendientes
                insert_point = existing.find("## Pendientes")
                insert_point = existing.find("\n", insert_point) + 1
                new_content = (
                    existing[:insert_point]
                    + "\n" + new_entries + "\n"
                    + existing[insert_point:]
                )
            else:
                new_content = existing + "\n## Pendientes\n\n" + new_entries + "\n"

            with open(curiosity_file, "w") as f:
                f.write(new_content)

            logger.info(f"Escritas {len(questions)} preguntas a {curiosity_file}")

        except Exception as e:
            logger.error(f"Error escribiendo curiosidad: {e}")

    return questions


# ══════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════


def _is_configured() -> bool:
    """Verifica si hay al menos un LLM configurado."""
    return bool(
        LLM_CONFIG["primary"]["api_key"]
        or LLM_CONFIG["fallback"]["api_key"]
    )


def is_llm_available() -> bool:
    """
    Health check rápido — verifica si al menos un LLM responde.
    
    Returns: True si hay al menos un LLM disponible
    """
    if not _is_configured():
        return False

    # Prueba simple: pedir al LLM que diga "ok"
    result = _call_llm("Responde SOLO con la palabra OK si me lees.", "Eres un asistente.")
    if result and "ok" in result.strip().lower():
        return True

    return False