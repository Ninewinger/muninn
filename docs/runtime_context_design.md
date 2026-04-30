# Paso 3: Runtime Context — Inyección Dinámica de Contexto

**Fecha:** 2026-04-10
**Estado:** Diseño
**Autor:** nanobot + Diego

---

## El Problema

Actualmente nanobot carga **todos** estos archivos en CADA mensaje:

| Archivo | Tokens | Se carga? |
|---------|--------|-----------|
| AGENTS.md | ~500 | SIEMPRE |
| SOUL.md | ~200 | SIEMPRE |
| USER.md | ~150 | SIEMPRE |
| TOOLS.md | ~200 | SIEMPRE |
| MEMORY.md | ~2000+ | SIEMPRE |
| Skills SKILL.md | ~500 | SIEMPRE (todas) |
| **Total** | **~3550+** | **Por mensaje** |

Esto significa que **cada mensaje a GLM-5.1 paga ~3550 tokens de overhead**, sin importar si son relevantes. En Telegram con >210k de contexto, esto se multiplica exponencialmente.

**El objetivo:** Solo inyectar lo que Muninn dice que es relevante. Como la memoria humana.

---

## Cómo funciona nanobot v0.1.5 (internals)

### Flujo actual de contexto

```
Mensaje del usuario
    ↓
AgentLoop.process()
    ↓
ContextBuilder.build_messages()
    ├── build_system_prompt()          ← ACA esta el problema
    │   ├── _get_identity()            → runtime, python version, etc.
    │   ├── _load_bootstrap_files()    → AGENTS.md + SOUL.md + USER.md + TOOLS.md
    │   ├── memory.get_memory_context() → MEMORY.md completo
    │   ├── skills.load_always_skills() → Skills "always"
    │   └── skills.build_skills_summary() → Lista de skills
    ├── _build_runtime_context()       → timestamp, channel, chat_id
    └── history (session messages)
    ↓
LLM call
```

### Puntos de extensión disponibles

nanobot v0.1.5 tiene estos hooks:

1. **`AgentHook.before_iteration(context)`** — Se ejecuta antes de cada iteración del loop. `context.messages` es mutable.
2. **`ContextBuilder.build_system_prompt()`** — Construye el system prompt. Es un método que se puede extender.
3. **`MemoryStore.get_memory_context()`** — Retorna MEMORY.md completo. Es un método que se puede override.
4. **`ContextBuilder.BOOTSTRAP_FILES`** — Lista de archivos que carga siempre.

### Limitación clave

nanobot NO tiene un sistema de plugins para reemplazar componentes del core. Los cambios requieren:
- **Opción A:** Modificar código fuente de nanobot (fragil, se pierde en updates)
- **Opción B:** Usar hooks para modificar `context.messages` antes de enviar (limpio, sobrevive updates)
- **Opción C:** Reemplazar archivos estáticos con contenido dinámico generado por Muninn (compatible, no toca código)

---

## Solución: Opción B + C — Hook + Archivos Dinámicos

### Arquitectura propuesta

```
Mensaje del usuario → nanobot AgentLoop
    ↓
before_iteration hook (MuninnContextHook)
    ├── Extrae último mensaje del usuario
    ├── Llama Muninn route() → top-3 activaciones
    ├── Genera contexto dinámico
    └── Inyecta/reemplaza en context.messages[0] (system prompt)
    ↓
LLM recibe SOLO:
    - Identity (runtime, ~100 tokens) — SIEMPRE
    - AGENTS.md minimal (~100 tokens) — SOLO reglas de seguridad
    - Muninn Dynamic Context (~200-500 tokens) — SOLO peers activados
    - Runtime metadata (~30 tokens)
    - History (truncada al contexto disponible)
```

### Implementación en 2 fases

#### Fase 1: Archivos Dinámicos (sin tocar nanobot code)

La idea: un script que se ejecuta ANTES de cada mensaje (via hook `before_iteration`) y reescribe los archivos bootstrap con contenido mínimo/relevante.

**Problema:** Los hooks en nanobot son async Python, no scripts shell. El hook `before_iteration` recibe `AgentHookContext` con `context.messages` mutable.

**Solución práctica:** Crear un hook Python que se carga como skill y modifica los messages directamente.

```python
# skills/muninn/muninn_context_hook.py
class MuninnContextHook(AgentHook):
    async def before_iteration(self, context):
        # 1. Extraer texto del último user message
        user_text = extract_last_user_message(context.messages)
        
        # 2. Route via Muninn (SQLite directo, sin API)
        activations = muninn_route(user_text)
        
        # 3. Generar contexto dinámico
        dynamic_context = build_dynamic_context(activations)
        
        # 4. Reemplazar system prompt
        # context.messages[0] es el system message
        # Reemplazar BOOTSTRAP_FILES content + MEMORY content
        # con solo lo que Muninn activó
        context.messages[0]["content"] = rebuild_system_prompt(
            identity=get_identity(),      # runtime info (~100 tok)
            agents_minimal=get_minimal_agents(),  # solo safety (~100 tok)
            muninn_context=dynamic_context,  # peers activados (~300 tok)
        )
```

**Ahorro estimado:**
- Antes: ~3550 tokens por mensaje
- Después: ~530 tokens por mensaje
- **Ahorro: ~3000 tokens/mensaje (~85%)**

#### Fase 2: Reemplazo Total (Runtime Context Completo)

Cuando Fase 1 funcione, eliminar los archivos bootstrap completamente:

1. **SOUL.md** → `peer_identidad` (solo si conflicto de tono)
2. **USER.md** → `peer_usuario` (solo si contexto personal relevante)
3. **TOOLS.md** → `peer_herramientas` (solo si se necesita una tool)
4. **AGENTS.md** → `peer_operativo` (solo para reglas/recordatorios)
5. **MEMORY.md** → Eliminado, reemplazado por peers de dominio
6. **Skills** → `peer_skills` (solo la skill relevante)

---

## Detalle Técnico: Cargar Hook en nanobot

### Dónde se cargan los hooks

En `agent/loop.py`, `AgentLoop.__init__`:
```python
self._extra_hooks: list[AgentHook] = hooks or []
```

Los hooks se pasan al `AgentLoop` desde... necesito verificar:

```python
# loop.py línea ~250
self._hook = CompositeLoopHook(primary=self, extras=self._extra_hooks)
```

El `_extra_hooks` viene del parámetro `hooks` del constructor. Pero ¿quién pasa hooks al AgentLoop?

Necesitamos buscar dónde se instancia AgentLoop para ver si hay un mecanismo de inyección.

### Alternativa: before_iteration via skill

Si no podemos inyectar hooks fácilmente, la alternativa más limpia es:

1. **Script de pre-procesamiento** que corre en el heartbeat
2. **Reemplazar archivos bootstrap** con versiones mínimas
3. **MEMORY.md** se convierte en una sección dinámica de Muninn

Esto ya lo hacemos parcialmente con `muninn_hook.py --update-memory`. La extensión es:

- AGENTS.md → versión minimal (solo safety rules, ~100 tokens)
- SOUL.md → versión minimal (solo nombre y estilo, ~50 tokens)
- USER.md → versión minimal (solo timezone e idioma, ~30 tokens)
- TOOLS.md → mantener completo (necesario para function calling)
- MEMORY.md → sección Muninn dinámica (ya implementado)

---

## Plan de Implementación

### Paso 3.1: Verificar mecanismo de hooks (HOY)

- [ ] Buscar dónde se instancia `AgentLoop` y si hay config para hooks custom
- [ ] Verificar si skills pueden registrar hooks
- [ ] Si sí → implementar `MuninnContextHook`
- [ ] Si no → ir a Paso 3.2

### Paso 3.2: Archivos Dinámicos vía Heartbeat

- [ ] Crear `muninn_hook.py --slim-bootstrap` que reescribe:
  - `SOUL.md` → versión slim
  - `USER.md` → versión slim  
  - `AGENTS.md` → versión slim (solo safety)
- [ ] Mantener originales como `SOUL.md.full`, `USER.md.full`, etc.
- [ ] Heartbeat ejecuta `--slim-bootstrap` + `--update-memory`
- [ ] **Ahorro: ~1500 tokens/mensaje**

### Paso 3.3: Hook before_iteration

- [ ] Si encontramos cómo registrar hooks custom:
  - [ ] `MuninnContextHook.before_iteration()` intercepta messages
  - [ ] Reemplaza system prompt con contexto Muninn dinámico
  - [ ] **Ahorro: ~3000 tokens/mensaje**

### Paso 3.4: Peers del Sistema (Fase 6 del roadmap)

- [ ] Crear 5 peers nuevos (identidad, usuario, herramientas, operativo, skills)
- [ ] Cada uno con facetas que representan secciones de los archivos
- [ ] El router decide qué secciones cargar
- [ ] **Ahorro total: ~3000+ tokens/mensaje, contexto 100% dinámico**

---

## Tokens Saved (Estimación)

| Fase | Tokens/msg | Ahorro | Complejidad |
|------|-----------|--------|-------------|
| Actual | ~3550 | 0% | — |
| Paso 3.2 (slim files) | ~2000 | 44% | Baja |
| Paso 3.3 (hook) | ~800 | 77% | Media |
| Paso 3.4 (peers) | ~530 | 85% | Alta |

---

## Notas

- El modelo de embeddings ya está cargado en CPU (~240MB, cached después del primer load)
- Route tarda ~50ms una vez cargado el modelo
- No afecta latencia percibida del usuario
- La DB ya tiene `route_with_context_injection()` en `router_v2.py` que genera el texto de inyección
