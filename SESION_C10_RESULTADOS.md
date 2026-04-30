# Sesión C10 — Reranker Test (bge-reranker-v2-m3)

**Fecha:** 2026-04-09  
**Objetivo:** Evaluar si agregar `BAAI/bge-reranker-v2-m3` como CrossEncoder mejora la precisión de routing del sistema Disco Elysium.

---

## Métricas Globales

| Métrica | Sin Reranker | Con Reranker | Δ |
|---------|-------------|-------------|---|
| **R@1** | 29/37 (78.4%) | 33/37 (89.2%) | **+10.8pp** ✅ |
| **R@3** | 33/37 (89.2%) | 33/37 (89.2%) | 0.0pp |
| **R@any** | 37/37 (100.0%) | 37/37 (100.0%) | 0.0pp |

> **Conclusión:** El reranker mejora **R@1 en casi 11 puntos porcentuales**, pasando de 78.4% a 89.2%. R@3 y R@any ya estaban en techo y no cambian.

---

## Configuración

- **Embedding model:** Qwen/Qwen3-Embedding-0.6B (local, CPU)
- **Reranker:** BAAI/bge-reranker-v2-m3 (~560MB, CrossEncoder)
- **DB:** 72 facetas cargadas
- **Test phrases:** 37

---

## Detalle por Frase

Leyenda: ✅✅ = ambos aciertan | ❌❌ = ambos fallan | ✅❌↑ = solo reranker acierta | ❌✅↑ = solo sin-reranker acierta | ✅❌↓ = solo sin-reranker acierta (reranker empeora)

| Status | Frase | Sin Reranker (top-3) | Con Reranker (top-3) |
|--------|-------|---------------------|---------------------|
| ✅✅ | Me siento ignorado, nadie me contesta | sombra_r, sombra_f, sombra_a | sombra_r, sombra_f, casual_s |
| ✅✅ | No me invitaron a la salida del grupo | sombra_r, casual_s, sombra_f | sombra_r, sombra_f, programa |
| ❌❌ | Voy a cargar creatina hoy | peer_usu, gym_ruti, casual_s | casual_s, gym_ruti, peer_usu |
| ✅✅ | Me siento abrumado por todo lo que tengo q… | sombra_a, sombra_f, sombra_r | sombra_a, sombra_f, sombra_r |
| ✅✅ | La nona esta mejor pero sigue en cama | sombra_m, gym_ruti, sombra_a | sombra_m, sombra_f, sombra_a |
| ✅✅ | Si lloro es debilidad | sombra_f, sombra_a, sombra_m | sombra_f, proyecto, sombra_r |
| ✅✅ | Tengo miedo de perder a alguien cercano | sombra_m, sombra_a, sombra_r | sombra_m, sombra_r, sombra_a |
| ✅✅ | Una chica que me gustaba me dejo en visto | sombra_r, sombra_a, sombra_f | sombra_r, proyecto, casual_s |
| ✅✅ | La ansiedad no me deja dormir | sombra_a, sombra_m, peer_ide | sombra_a, sombra_f, sombra_r |
| ❌❌ | El gimnasio es mi templo | peer_usu, peer_ski, gym_ruti | peer_usu, gym_ruti, peer_ski |
| ✅✅ | Siento que no encajo en ninguna parte | sombra_a, sombra_r, peer_ide | sombra_r, sombra_f, sombra_a |
| ✅✅ | Necesito ayuda pero no puedo pedirla | sombra_f, peer_ide, sombra_a | sombra_f, peer_ide, sombra_r |
| ✅✅ | Me da terror abrirme emocionalmente | sombra_r, proyecto, sombra_a | sombra_r, sombra_a, proyecto |
| ✅❌↓ | Cierro los ojos y veo oscuridad | sombra_a, sombra_m, sombra_f | sombra_a, proyecto, sombra_r |
| ✅✅ | La tecnologia me estresa | sombra_a, peer_ski, peer_usu | sombra_a, sombra_r, sombra_f |
| ✅✅ | Las piernas que antes eran mi debilidad | gym_ruti, sombra_f, sombra_m | sombra_f, gym_ruti, sombra_m |
| ✅✅ | El grupo hace planes sin mi | sombra_r, casual_s, peer_ope | sombra_r, casual_s, peer_usu |
| ✅✅ | El doctor dijo que hay que hacer mas exáme… | sombra_m, peer_ide, sombra_a | sombra_m, sombra_a, sombra_f |
| ❌❌ | Saque un nuevo PR en sentadilla hoy | peer_usu, peer_ide, gym_ruti | gym_ruti, casual_s, nanobot_ |
| ✅✅ | Me borraron del grupo de WhatsApp | sombra_r, sombra_f, casual_s | sombra_r, sombra_f, peer_ide |
| ✅✅ | El funeral de mi tio me dejo pensando | sombra_m, valle_al, sombra_a | sombra_m, sombra_a, sombra_f |
| ✅✅ | Me quede scrolleando hasta las 3am | sombra_a, peer_usu, casual_s | sombra_a, sombra_r, peer_usu |
| ✅✅ | Me dieron un abrazo y casi me echo a llora… | sombra_f, sombra_r, sombra_a | sombra_f, sombra_r, proyecto |
| ✅✅ | Mi ex novia cambio su estado | sombra_r, sombra_m, peer_usu | sombra_r, proyecto, sombra_a |
| ✅✅ | Sueno que se me caen los dientes | sombra_m, sombra_a, peer_ide | sombra_m, sombra_a, proyecto |
| ❌✅↑ | Como deberia funcionar el sistema de comba… | peer_ide, sombra_a, peer_ope | proyecto, sombra_a, peer_ski |
| ✅✅ | Tengo un error en la linea 42 | programa, peer_ide, sombra_a | programa, peer_her, peer_ope |
| ✅✅ | Como funcionan los decoradores en Python | programa, peer_ide, peer_ski | programa, peer_ski, sombra_a |
| ✅✅ | Configura un recordatorio para manana | peer_ope, nanobot_, peer_usu | nanobot_, peer_ope, peer_her |
| ✅✅ | Instalar una skill nueva desde clawhub | nanobot_, peer_ski, peer_ope | nanobot_, peer_ski, peer_ope |
| ✅✅ | Las patentes mineras estan vigentes | valle_al, peer_ski, sombra_a | valle_al, peer_ide, nanobot_ |
| ✅✅ | Hoy toca pierna, bulgaras con 35 kilos | gym_ruti, peer_usu, sombra_m | gym_ruti, peer_usu, casual_s |
| ✅✅ | Hola weon, como estai | casual_s, peer_usu, sombra_m | casual_s, sombra_f, sombra_a |
| ✅✅ | Quien soy yo como asistente? | peer_ide, sombra_f, casual_s | peer_ide, peer_usu, proyecto |
| ✅✅ | Que hora es? | casual_s, peer_usu, peer_ide | casual_s, gym_ruti, peer_ope |
| ✅✅ | Necesito ejecutar un comando en la termina… | peer_her, peer_ope, sombra_a | peer_her, peer_ski, peer_ope |
| ✅✅ | Antes de empezar, cual es la vision del pr… | peer_ope, peer_usu, casual_s | peer_ope, peer_ide, peer_usu |

---

## Notas

- **Reranker download:** ~560MB, descargado exitosamente en `D:\hf_cache`.
- **Warning symlink:** Windows sin Developer Mode no soporta symlinks en cache. No afecta funcionalidad.
- **Frase problemática recurrente:** "Voy a cargar creatina hoy" y "El gimnasio es mi templo" fallan en ambos modos — requieren investigación de facetas gym.
- **Caso de regresión:** "Cierro los ojos y veo oscuridad" — sin reranker acierta R@1, con reranker baja (✅❌↓).
