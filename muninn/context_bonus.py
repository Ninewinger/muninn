"""Enhanced context bonus for Muninn semantic router.

Replaces the basic _compute_context_bonus with:
- Hour-based: morning (operativo), afternoon (social), night (dreams/shadows)
- Day-of-week: Friday/Saturday (autoevaluacion, social), weekend (casual)
- Recency bonus: boost peers that were recently activated
- Phase-of-day: late night = suenos bonus
"""
import os
from datetime import datetime

# Recency tracking: peer_id -> unix timestamp of last activation
_recency_cache: dict = {}

def compute_context_bonus(peer_id: str, context_hour: int | None = None) -> float:
    """Enhanced context bonus with hour, day-of-week, and recency.
    
    Returns bonus in range [0.0, 0.15]
    """
    bonus = 0.0
    
    # Get current time info
    now = datetime.now()
    hour = context_hour if context_hour is not None else now.hour
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    # ── Hour-based bonuses ──
    
    # Morning (5-9am): operativo, gym_rutina, hermes_sistema
    if 5 <= hour <= 9:
        if peer_id in ("peer_operativo", "gym_rutina", "hermes_sistema"):
            bonus += 0.08
        if peer_id == "autoevaluacion":
            bonus += 0.03  # optional morning review
    
    # Midday (10am-2pm): herramientas, programacion, learning
    elif 10 <= hour <= 14:
        if peer_id in ("peer_herramientas", "programacion", "hermes_sistema"):
            bonus += 0.05
    
    # Afternoon/evening (3-7pm): social, creative, proyecto_juego
    elif 15 <= hour <= 19:
        if peer_id in ("casual_social", "proyecto_juego", "creative"):
            bonus += 0.05
        if peer_id == "relaciones_personales":
            bonus += 0.03
    
    # Night (8-11pm): shadows, dreams, memory consolidation
    elif 20 <= hour <= 23:
        if peer_id.startswith("sombra_"):
            bonus += 0.06
        if peer_id == "suenos_analisis":
            bonus += 0.08
        if peer_id == "memoria_durable":
            bonus += 0.03  # memory consolidation at night
    
    # Late night (0-4am): deep sleep, shadows peak
    elif 0 <= hour <= 4:
        if peer_id.startswith("sombra_"):
            bonus += 0.10
        if peer_id == "suenos_analisis":
            bonus += 0.12
        if peer_id == "memoria_durable":
            bonus += 0.05
    
    # ── Day-of-week bonuses ──
    
    # Monday (0): start of week — operativo, programacion
    if weekday == 0:
        if peer_id in ("peer_operativo", "programacion"):
            bonus += 0.05
    
    # Friday (4): autoevaluacion (fin de semana laboral)
    if weekday == 4:
        if peer_id == "autoevaluacion":
            bonus += 0.10
        if peer_id == "relaciones_personales":
            bonus += 0.03
        if peer_id == "finanzas_patrimonio":
            bonus += 0.03  # weekend spending planning
    
    # Saturday (5): social, leisure, creative
    if weekday == 5:
        if peer_id in ("casual_social", "relaciones_personales"):
            bonus += 0.08
        if peer_id == "proyecto_juego":
            bonus += 0.05
        if peer_id == "gym_rutina":
            bonus += 0.03  # weekend gym
    
    # Sunday (6): rest, shadows, family
    if weekday == 6:
        if peer_id.startswith("sombra_"):
            bonus += 0.05
        if peer_id in ("relaciones_personales", "casual_social"):
            bonus += 0.05
        if peer_id == "autoevaluacion":
            bonus += 0.03  # end-of-week reflection
    
    # ── Keep original bonuses for backward compat ──
    # Atardecer bonus (sunset = shadow hours)
    if peer_id == "sombra_angel_atardecer" and 17 <= hour <= 23:
        bonus += 0.05
    
    # ── Recency bonus (if recency data available) ──
    # This is filled by the route function before calling this
    # recency_seconds is stored as peer-level metadata
    
    return round(min(bonus, 0.15), 4)  # cap at 0.15


def record_activation(peer_id: str):
    """Record that a peer was activated (for recency tracking)."""
    _recency_cache[peer_id] = datetime.now().timestamp()


def get_recency_bonus(peer_id: str, current_time: float = None) -> float:
    """Return bonus if this peer was recently activated."""
    if peer_id not in _recency_cache:
        return 0.0
    if current_time is None:
        current_time = datetime.now().timestamp()
    
    elapsed = current_time - _recency_cache[peer_id]
    
    # Within last minute: strong bonus
    if elapsed < 60:
        return 0.08
    # Within last 5 minutes: moderate bonus
    elif elapsed < 300:
        return 0.05
    # Within last 30 minutes: small bonus
    elif elapsed < 1800:
        return 0.03
    # Within last 2 hours: tiny bonus
    elif elapsed < 7200:
        return 0.01
    
    return 0.0


def get_recency_summary() -> str:
    """Return human-readable summary of recent activations."""
    now = datetime.now().timestamp()
    lines = []
    for pid, ts in sorted(_recency_cache.items(), key=lambda x: x[1], reverse=True):
        elapsed = now - ts
        if elapsed < 7200:  # only last 2 hours
            minutes = int(elapsed / 60)
            lines.append(f"  {pid}: {minutes}m ago")
    return "\n".join(lines) if lines else "  No recent activations"