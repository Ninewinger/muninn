# Plan de Mejoras Muninn v0.3 — Mayo 2026

## Visión General

Integrar los tres sistemas de memoria/aprendizaje de Diego:
1. **Muninn** (memoria semántica persistente)
2. **Curiosidad** (cola de preguntas en Obsidian)
3. **Lanting** (generación de preguntas desde notas incompletas)

Hacia un ciclo orgánico: conversación → extracción → curiosidad → preguntas → respuestas → consolidación.

---

## 1. LLM en Dreaming

### Estado Actual
- `dreaming.py` L3: "Sin LLM externo. Usa embeddings + reglas"
- `dream()` solo clasifica eventos, registra activaciones, descubre co-activaciones, actualiza confianza, olvida
- `_process_event()` desactivó la creación de memorias porque sin LLM guardaba fragmentos de conversación como basura
- Comentario en L240-247: las memorias ahora solo entran por `muninn_add_memory()` (Hermes decide)

### Propuesta
1. **Configurar LLM opcional** en dreaming — no modifica el flujo actual, lo **extiende**
2. Si `LLM_API_URL` está definida, después de procesar cada evento:
   - Enviar el contenido del evento + los peers activados al LLM
   - El LLM extrae: ¿hay un hecho durable aquí? ¿Qué peer es relevante?
   - Si sí, guarda como memoria con `source='dreaming_llm'` y confianza alta
3. Si no hay LLM, el comportamiento sigue exactamente igual (backward compatible)

### Implementación
- Archivo nuevo: `dreaming_llm.py` (mantener dreaming.py limpio)
- Config vía variables de entorno (ya existen como template en `config/env.template`)
- Modelo recomendado: NVIDIA NIM (gratis), mismo patrón que Hermes
- Límite: solo procesar eventos de tipo `user_message` (no system/assistant)
- Filtro: solo eventos con peers activados con score > 0.5

```python
# dreaming_llm.py (borrador)
def extract_facts_with_llm(event_content, activated_peers):
    """Envía evento a LLM → decide si contiene hecho durable"""
    prompt = f"""
    Eres un extractor de conocimiento durable.
    
    Analiza esta interacción y determina si contiene un hecho 
    estable sobre el usuario (preferencia, dato personal, insight, patrón).
    
    Evento: "{event_content}"
    
    Peers activados: {activated_peers}
    
    Responde SOLO con JSON:
    - has_fact: bool
    - fact_text: str (el hecho extraído, o "")
    - relevant_peer: str (peer_id, o "")
    - confidence: float (0-1)
    - type: str (preferencia|dato|insight|patron|none)
    """
    # llamar a LLM API → parsear JSON → retornar
```

---

## 2. Curiosidad → Dreaming

### Estado Actual
- Sistema de curiosidad: cron diario (12:30pm) que lee `cola-curiosidad.md` y entrega pregunta (skill `curiosidad-auto-update`)
- Preguntas escritas a mano en el archivo, prioridades fijas
- No hay generación automática de preguntas
- El cron solo entrega, no detecta respuestas en tiempo real

### Propuesta
1. **Muninn dreaming genera preguntas de curiosidad** — después de consolidar, analiza peers con baja cantidad de memorias o facets
2. **Escribe preguntas directamente a `cola-curiosidad.md`** en Obsidian
3. **Hermes detecta respuestas** durante la conversación y las guarda como memorias en Muninn
4. **El cron de curiosidad solo entrega** la pregunta más prioritaria del día

### Pipeline
```
Conversación (Hermes + Diego)
    ↓ [eventos se guardan en Muninn cada turno]
Dreaming nocturno (o al cerrar sesión)
    ↓ [consolida eventos, registra activaciones]
    ↓ [SI LLM configurado: extrae hechos, guarda memorias]
Fase de Curiosidad (nueva)
    ↓ [analiza peers con facets o memorias incompletas]
    ↓ [genera preguntas → cola-curiosidad.md]
Cron curiosidad (12:30pm) 
    ↓ [lee cola, entrega pregunta más prioritaria]
Diego responde
    ↓ [Hermes detecta, guarda en Muninn, mueve a respondidas]
Dreaming (próximo ciclo)
    ↓ [consolida la respuesta como memoria]
```

### Implementación
- Función nueva en dreaming: `_generate_curiosity_questions(conn)` que se ejecuta SI hay LLM configurado
- Evalúa cada peer: ratio memorias/edad, facets sin explorar, lagunas en conocimiento
- Para cada peer con datos insuficientes, genera 1 pregunta y la escribe al archivo

---

## 3. Lanting → Muninn

### Estado Actual
- Lanting (plugin Obsidian) genera preguntas de notas incompletas
- No hay conexión con Muninn ni con curiosidad

### Propuesta
1. Exportar preguntas de Lanting al mismo archivo `cola-curiosidad.md`
2. Hermes puede consultar esas preguntas durante la conversación
3. Las respuestas se guardan como memorias en Muninn

### Integración
- Cron semanal que ejecuta Lanting (si tiene CLI) o copia sus outputs
- O simplemente unificar el archivo de curiosidad como punto de entrada único
- Hermes mergea preguntas de Lanting + preguntas generadas por dreaming

---

## 4. Skill de Criterios de Memoria

### Propuesta
Skill explícito que Hermes sigue al decidir guardar algo en Muninn:

```yaml
Criterios para muninn_add_memory:
  1. DURABILIDAD: ¿Este hecho será cierto en 3 meses?
  2. COSTO DE PÉRDIDA: ¿Si se olvida, la experiencia empeora?
  3. SINGULARIDAD: ¿Es específico de Diego o genérico?
  4. COMPLETITUD: ¿Llena un vacío en el conocimiento actual?
  5. CURIOSIDAD: ¿Revela algo que no sabía antes?

NO guardar:
  - Resultados de sesiones (ya están en session_search)
  - Confirmaciones triviales ("sí", "ok", "entiendo")
  - Datos obvios o fácilmente rediscoverables
  - TODO state o progreso temporal
```

---

## 5. Hoja de Ruta

### Fase 1 (Ahora — Skill + configuración)
- [ ] Escribir skill de criterios de memoria
- [ ] Auditar memorias existentes en Muninn (limpiar redundancias)
- [ ] Agregar `.env` con LLM_API_URL para dreaming (sin modificar código aún)

### Fase 2 (Corto plazo — LLM en dreaming)
- [ ] Crear `dreaming_llm.py` con función `extract_facts_with_llm()`
- [ ] Modificar `dream()` para llamarlo si LLM configurado
- [ ] Probar con dry_run: verificar calidad de hechos extraídos
- [ ] Ajustar thresholds de confianza

### Fase 3 (Mediano plazo — Curiosidad pipeline)
- [ ] Función `_generate_curiosity_questions()` en dreaming
- [ ] Cambiar ruta del archivo curiosidad (de `asistente/` a `01_sistema/asistente/`)
- [ ] Modificar cron de curiosidad para usar preguntas generadas

### Fase 4 (Largo plazo — Lanting + ciclo completo)
- [ ] Explorar integración con Lanting (tiene API/CLI?)
- [ ] Pipeline completo: evento → LLM → memoria → curiosidad → pregunta → respuesta → aprendizaje

---

## Notas Técnicas

### Configuración recomendada para LLM
- Provider: NVIDIA NIM (gratis, mismo que Hermes)
- Modelo: `meta/llama-3.3-70b-instruct` (rápido, 0.7s, suficiente calidad)
- Costo: $0 (NVIDIA NIM gratuito)
- Contexto: eventos de < 2000 chars, sin necesidad de mucho contexto

### Riesgos
1. Contaminación de representaciones: el LLM puede generar "hechos" incorrectos
   - Mitigación: confianza inicial baja (0.6), requiere refuerzo de múltiples eventos
2. Preguntas genéricas o irrelevantes
   - Mitigación: preguntas solo de peers con > 3 activaciones pero < 5 memorias
3. Dependencia de API externa
   - Mitigación: dreaming funciona sin LLM, es 100% opcional

### Métricas de éxito
- Reducción de eventos sin procesar (S12 ya monitorea esto)
- Aumento de memorias creadas por dreaming (vs solo por Hermes)
- Preguntas de curiosidad con tasa de respuesta > 50%
- Precisión de extracción de hechos > 80% (muestreo manual)