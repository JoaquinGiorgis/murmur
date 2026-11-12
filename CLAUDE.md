# murmur: contexto para Claude Code

Pipeline para el desafío QualyData "Narrativas y coordinación a escala" (Hackathon Argentina Space 2026, Salta). Equipo: Joaquín Giorgis. Hablale en castellano rioplatense, directo y con opinión.

## Fechas críticas
- **Set final:** se publica el viernes 13/11/2026 a la mañana.
- **Cierre:** viernes 13/11 a las 10:00 (hora Argentina). Hay que commitear y pushear `entrega/` y subir el proyecto en Hacki.
- **Excluyente según las bases:** video de demo y pitch ante el jurado.

## Cómo correr
```bash
git clone https://github.com/qualydata-org/desafio-qualydata ../desafio-qualydata   # bases, scorer y datos de práctica
ln -s ../desafio-qualydata/datos datos                                             # 'datos' está en .gitignore
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
python -m murmur --datos datos/practica --entrega salida/practica              # sin clave: corre sin IA
python ../desafio-qualydata/evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica
```
- Con IA: exportar `ANTHROPIC_API_KEY` (nunca commitearla). El modelo por defecto es `claude-haiku-5-5` (`MURMUR_LLM_MODELO`).
- Caché: la base DuckDB y los embeddings quedan en `.murmur/`. En frío, los embeddings tardan unos 6,5 minutos en 2 CPU.
- **Ojo con pandas 3:** las fechas van en microsegundos. Usá `.astype('datetime64[s, UTC]').astype('int64')`, nunca `//10**9`.

## Estado (set de práctica, scorer oficial)
- **Con IA: 52,5 / 55.** Agrupamiento 0,981, ruido 0,979, postura 0,992, origen 1,0, cuentas coordinadas F1 0,935 (precisión 1,0), grupos 0,876. Costo IA USD 0,17.
- **Sin IA: 51,0.** La postura baja a 0,76, porque elegir el lado que "apoya" es casi una moneda en narrativas 50/50.

## Arquitectura
| Módulo | Qué hace |
|---|---|
| `datos.py` | DuckDB lee los `.csv.gz` y limpia en SQL: duplicados, fechas, respuestas huérfanas, cuentas faltantes. |
| `embeddings.py` | MiniLM multilingüe local sobre textos únicos normalizados; caché Parquet por hash. |
| `narrativas.py` | UMAP + HDBSCAN dan clusters finos. Las respuestas los cosen en narrativas (Leiden) y el 2-coloreo con signo separa los lados (corta = misma postura, larga = opuesta). Reglas de ruido, fusión de fragmentos, origen por densidad en 3 h. IA opcional: elige el lado que afirma con 4 votos alternando el orden (sesgo de posición), con ejemplos limpios de muletillas, y pone los títulos. |
| `coordinacion.py` | Co-acción en la misma publicación dentro de 10 min, normalizada por coseno, más respuestas entre cuentas. Leiden, comunidades chicas y densas, y poda del núcleo (PODA_FRAC=0,5). |
| `dashboard.py` + `dashboard.html` | HTML autocontenido: qué circula, desde cuándo, quién lo empuja. |

## Revisión independiente (Fable 5.2) y plan pendiente, en orden

1. **Coordinación frágil fuera de práctica.** Con recortes (10 días, sin tema litio, solo hospital, 50% de publicaciones) los falsos positivos se disparan (hasta 786) y el total cae a 30–44.
   - Lo único que frena a la masa de usuarios genuinos activos es `TAM_MAX_FRAC=0.02`.
   - **Arreglo validado:** exigir además densidad del grupo ≥ 0,09 (`grado/(n-1)`, medida después de la poda), manteniendo `VENTANA=600`. Las operaciones reales dan 0,12–0,32 y los grupos genuinos ≤ 0,063. En 12 recortes los FP bajaron de 2.145 a 32, sin cambiar el puntaje de práctica.
   - No subir `VENTANA` a 900. No rediseñar el detector.
   - Sumar un **banco de estrés** como test (recortes por tiempo, tema y volumen, puntuados con el scorer). Aceptar cambios de coordinación solo si no suben los FP en ningún recorte.
2. **Ensayo general:** clon limpio, versiones fijas en `requirements.txt`, modelo de embeddings precargado (probar con `HF_HUB_OFFLINE=1`), corrida en frío cronometrada y validación de que la salida tenga todos los ids.
3. **IA robusta:** hoy `LLM.juzgar` traga excepciones y devuelve None sin avisar. Sumar reintentos y backoff ante 429/5xx, un log visible si cae al heurístico, y `map(str)` sobre las frases clave (en `__main__.py`, `"|".join` rompe si viene un string o un no-string).
4. **Umbrales de narrativas relativos a la tasa de respuestas:** `ARISTA_MIN=3`, `TASA_RESP_RUIDO=0.02` y `RUIDO_REPETIDO=20` son absolutos. Con pocas respuestas explotan las narrativas o todo queda como ruido. Sumar una alarma si el ruido supera ~35%.
5. **Entrega y README:**
   - Mover `dashboard.html` fuera de `entrega/` (las bases piden 4 archivos).
   - Corregir "unas 15 llamadas" (fueron unas 50) y verificar los precios por defecto en `costo.json`.
   - Reportar el tiempo en frío y agregar una tabla de escala y memoria por etapa, con extrapolación a 1M comentarios por día.
   - Sumar 3–4 consultas SQL de ejemplo sobre las tablas `res_*`.
   - Fundir las mini-narrativas basura (de 1 a 27 comentarios).
6. **Video y pitch.** Hilo: el problema, por qué no todo con IA, la idea clave (las respuestas cosen las narrativas y la repetición delata a las operaciones), números y costo, y la demo del dashboard. Lema de QualyData: *Model. Integrate. Scale.*; hablar en ese idioma.

**Viernes:** correr sin IA y pushear de inmediato; después con IA y pisar. No tocar código ese día.

**No hacer:** cambiar embeddings o UMAP; reajustar umbrales para pasar de 52,5 a 53 en práctica; IA por comentario; usar la fecha de creación de cuentas como regla dura.
