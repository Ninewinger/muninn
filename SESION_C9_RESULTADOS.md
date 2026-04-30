# SESIÓN C9 — Resultados Calibración Qwen3-Embedding

**Fecha:** 2026-04-07  
**Modelo:** Qwen/Qwen3-Embedding-0.6B (fallback: 8B crasheó con exit code 3221225477 / OOM)  
**Test file:** `D:\github\muninn\test_c9_calibrate.py`

---

## Resumen Ejecutivo

El modelo 8B crasheó por falta de VRAM (access violation al cargar shards). Se usó el modelo **0.6B sin cuantización** como alternativa. Los resultados son excelentes y comparables al baseline C7 (MiniLM).

| Métrica | C7 MiniLM th=0.25 | C8 Qwen3 th=0.25 | **C9 Qwen3-0.6B th=0.25** |
|---------|-------------------|-------------------|---------------------------|
| R@1     | 83.3%             | 63.7%             | **81.4%**                 |
| R@3     | 93.1%             | 88.2%             | **92.2%**                 |
| R@any   | 95.1%             | 100%              | **100.0%**                |

**Conclusión: Qwen3-0.6B supera claramente al C8 y es competitivo con MiniLM C7, manteniendo R@any=100%.**

---

## Calibración de Threshold

```
  Threshold      R@1      R@3    R@any
  --------- -------- -------- --------
       0.25   81.4%   92.2%  100.0% <-- BEST
       0.30   81.4%   92.2%  100.0%
       0.35   81.4%   92.2%   99.0%
       0.40   81.4%   92.2%   96.1%
       0.45   81.4%   92.2%   94.1%
       0.50   77.5%   84.3%   85.3%
       0.55   66.7%   69.6%   69.6%
```

**Hallazgo clave:** R@1 y R@3 son estables desde th=0.25 hasta th=0.45. El threshold óptimo es **0.25–0.30** (maximiza R@any sin sacrificar R@1/R@3).

---

## Desglose por Peer (threshold=0.25)

```
  [+]               S:muerte: R@1= 6/8 R@3= 6/8 R@any= 8/8
  [OK]              S:rechazo: R@1=10/11 R@3=11/11 R@any=11/11
  [OK]      S:angel atardecer: R@1= 5/9 R@3= 8/9 R@any= 9/9
  [~]            S:fortaleza: R@1= 4/10 R@3= 5/10 R@any=10/10
  [OK]         proyecto juego: R@1=10/12 R@3=12/12 R@any=12/12
  [OK]           programacion: R@1=11/12 R@3=12/12 R@any=12/12
  [OK]        nanobot sistema: R@1=10/10 R@3=10/10 R@any=10/10
  [OK]             valle alto: R@1= 9/10 R@3=10/10 R@any=10/10
  [OK]             gym rutina: R@1= 9/10 R@3=10/10 R@any=10/10
  [OK]          casual social: R@1= 9/10 R@3=10/10 R@any=10/10
```

### Peers problemáticos:
- **S:fortaleza** (R@3=50%): La faceta más difícil. Muchas frases se confunden con gym_rutina y otros.
- **S:muerte** (R@3=75%): "La nona está mejor" se confunde con proyecto_juego; "Cierro los ojos" con nanobot.

---

## Frases que fallan R@3

```
X "Voy a cargar creatina hoy, se me acabo l" -> S:fortaleza
  got: gym ruti:0.590, nanobot :0.490, casual s:0.441

X "La nona esta mejor pero sigue en cama, m" -> S:muerte
  got: proyecto:0.521, S:rechaz:0.513, casual s:0.486

X "El gimnasio es mi templo, mi cuerpo es m" -> S:fortaleza
  got: nanobot :0.493, gym ruti:0.490, proyecto:0.429

X "Cierro los ojos y veo oscuridad, silenci" -> S:muerte
  got: proyecto:0.486, S:angel :0.466, nanobot :0.464

X "Saque un nuevo PR en sentadilla hoy" -> S:fortaleza
  got: nanobot :0.581, valle al:0.554, gym ruti:0.526

X "Me quede scrolleando hasta las 3am sin p" -> S:angel atardecer
  got: nanobot :0.584, casual s:0.577, gym ruti:0.566

X "Deje de ir al gym tres dias y me senti f" -> S:fortaleza
  got: S:rechaz:0.488, gym ruti:0.462, nanobot :0.422

X "Un nino lloraba en la calle y no supe qu" -> S:fortaleza
  got: casual s:0.434, proyecto:0.432, nanobot :0.406
```

**Patrón:** La sombra de fortaleza (armadura emocional, no pedir ayuda) se confunde frecuentemente con gym_rutina y nanobot. Esto sugiere que las facetas de fortaleza necesitan más especificidad o el instruction necesita dirigir mejor hacia contenido emocional vs. actividad física.

---

## Comparación Final

```
C7 (MiniLM, th=0.25):   R@1=83.3%  R@3=93.1%  R@any=95.1%
C8 (Qwen3, th=0.25):    R@1=63.7%  R@3=88.2%  R@any=100%
C9 (Qwen3-0.6B, th=0.25): R@1=81.4%  R@3=92.2%  R@any=100.0%
```

---

## Notas técnicas

- **8B model crash:** Exit code 3221225477 (Windows STATUS_ACCESS_VIOLATION) al cargar el primer shard. Causa probable: VRAM insuficiente o incompatibilidad de BitsAndBytesConfig en este entorno Windows.
- **0.6B model:** Cargado sin cuantización, ejecutado sin problemas en CPU/GPU.
- El archivo original fue restaurado a su estado original (Qwen3-Embedding-8B con 8-bit quant).
