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
- **Con IA: 53,31 / 55** (8 votos por narrativa, 104 llamadas, USD 0,03). Agrupamiento 0,983, ruido 0,974, postura 0,992, origen 1,0, cuentas coordinadas F1 0,959 (precisión 1,0), grupos 0,925.
- **Sin IA: 52,56.** La postura baja a 0,867, porque elegir el lado que "apoya" es casi una moneda en narrativas 50/50.
- Recorte de 10 días (el más parecido al set final): 52,62 sin IA. Falsos positivos en los 13 recortes del banco: 2.

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
1–3 **hechos** (3374c1d, 5d18137, 9bb4358). Ensayo en clon limpio sin red, en ese momento: 51,75 en 9 min; con IA 52,51 (52 llamadas, USD 0,017).

4. **Umbrales de narrativas relativos: PROBADO Y DESCARTADO.** Escalar `ARISTA_MIN` y `RUIDO_REPETIDO` por volumen y `TASA_RESP_RUIDO` por tasa global (idénticos en práctica) bajó narrativas en 10 de 12 recortes (−2,85 en total, −1,31 en pubs50_s1). Los absolutos rinden mejor; no reintentar. Quedó solo la alarma de ruido > 35% (solo log).
   - Fundir mini-narrativas (< 0,1% de los comentarios; las 13 de práctica, 91 comentarios, son todas pedazos de narrativas grandes): **descartado**. B-cubed sube un poco en todos los recortes, pero sin IA la postura de narrativas casi 50/50 se da vuelta por 10 comentarios: sin_colectivo −0,77, pubs50_s1 −0,59 (−1,27 en total). Solo valdría si la postura del "lado mayor" se decide antes de fundir.
   - Capa de texto casi duplicado en coordinación: también descartada. En este set los genuinos usan plantillas (230 pares de campaña vs 256.000 genuinos a similitud ≥ 0,7). Los hashtags propios son 100% de campaña en práctica, pero no se pueden validar contra el set final.
5. **Entrega y README: hecho** (3995498). Dashboard en `salida/`, costo real USD 0,017 con precios de Haiku 5.5, tabla de tiempo y memoria (10 min en frío, 2,1 GB), SQL de ejemplo.
   - Mover `dashboard.html` fuera de `entrega/` (las bases piden 4 archivos).
   - Corregir "unas 15 llamadas" (fueron unas 50) y verificar los precios por defecto en `costo.json`.
   - Reportar el tiempo en frío y agregar una tabla de escala y memoria por etapa, con extrapolación a 1M comentarios por día.
   - Sumar 3–4 consultas SQL de ejemplo sobre las tablas `res_*`.
   - Fundir las mini-narrativas basura (de 1 a 27 comentarios).
6. **Video y pitch.** Hilo: el problema, por qué no todo con IA, la idea clave (las respuestas cosen las narrativas y la repetición delata a las operaciones), números y costo, y la demo del dashboard. Lema de QualyData: *Model. Integrate. Scale.*; hablar en ese idioma.

**Jueves 12/11, tarde (hecho):**
- Robustez (257c91f): ingesta tolerante a CSV raros, latin-1, .csv sin comprimir y columnas faltantes; fechas no ISO cortan con un mensaje claro; coordinación y narrativas protegidas por separado (sin respuestas: 23,43 en vez de caerse; sale con código 2). Práctica y banco idénticos byte a byte.
- Fragmentos (06ca12f): `FRAG_FRAC` de 0,01 a 0,003, porque las emergentes chicas (H3 0,82%) se fundían con narrativas viejas. dias10 47,85 → 49,69 y dias7 36,10 → 39,14; el resto, idéntico.
- Con IA fuera de práctica: dias10 50,05 y sin_litio 40,28 (postura 0,99).
- Descartados con medición: Leiden con `n_iterations=-1` (dias10 −3,87: parte una operación) y sumar a las operaciones cuentas periféricas (miles de genuinas coinciden igual que las de campaña).
- Origen con umbral fijo (10 en 3 h + 30 en 24 h, sin escalar con el tamaño): **descartado**. Comentarios mal asignados juntaron 10 en 3 h cinco días antes y en sin_colectivo el origen de H3 cayó de 1,0 a 0 (−2,25). El umbral que crece con el tamaño protege contra esas fugas.
- Tope de origen (ORIGEN_TOPE=25): el umbral sigue creciendo con el tamaño, pero hasta 25 comentarios en 3 h. En el set de ~150k el origen de H3 pasa de 0,64 a 1,0 (38,03 → 38,85); el banco queda idéntico. Con tope 15 falla sin_colectivo.
- Satélites (d950f9d, hallado en el análisis con Fable 5.2): cuentas de baja actividad que en ≥2 publicaciones comentan a ≤10 min del núcleo con texto casi igual (≥0,8, ≥4 palabras) y eso es ≥50% de su actividad. Banco +15,14, falsos positivos 2→2. No aflojar (0,7 de similitud o 0,34 de fracción ya meten falsos positivos) ni iterar.
- GRADO_MIN 3,0→2,5 (03bbb5f): pubs50_s1 28,17→36,17, pubs50_s2 29,53→41,20; el resto idéntico. No bajar de 2,5 (a 2,0 entran grupos genuinos).
- N_VOTOS 4→8: con 4, n06 salió 3/4 mal en una de tres corridas (postura 0,90); con 8, 6/8 bien. Costo USD 0,017→0,03.
- Ruido escalable (RUIDO_REP_ESCALA): el umbral de repeticiones crece con el volumen (20 hasta 200k, 100 a 1M). A 1M (práctica ×5): agrupamiento 0,956→0,984, ruido 0,923→0,974, total 49,66→50,38. Banco idéntico.
- Ciclo de experimentos (tests/experimento.py, registro en .murmur/exp/registro.csv). Probado y no aceptado: SAT_SIM 0,75 (+0,52 solo en dias7, afloja la regla) y 0,85 (igual: los satélites son estables), SIM_ASIGNAR 0,75 (−0,21), SIM_COORD 0,6 (−0,05), ORIGEN_TOPE 20 (igual).
- UMAP_COMPAT=False (init recursiva + Adam en paralelo de umap-learn 0.6): ~4,8× más rápido (práctica 449 s → 94 s en el banco con -j 3) pero RECHAZADO: práctica 52,56 → 52,14 (postura sin IA 0,767), dias7 43,85 → 42,01 (origen 0,5); dias10 +0,44, resp20 +3,41; FP iguales. Queda como interruptor apagado (None = comportamiento de siempre) para volúmenes donde el tiempo importe más que medio punto. Las oscilaciones de ±2–3 puntos por recorte muestran cuánto pesa el layout de UMAP: el ruido de semilla no medido es real.
- Las 7 cuentas de C2 que faltan en práctica son activas (15-18 comentarios), no de baja actividad: co-actúan con el núcleo repartido entre muchos miembros (k≤3 por par). Fuerza contra el núcleo, alineamiento narrativo y respuestas con el núcleo no las separan de genuinas activas (el mejor corte suma 3 falsos positivos por 4 aciertos). Es el diseño de la campaña (actividad de cobertura).
- Umbral de ruido relativo a la tasa de respuestas (TASA_RELATIVA): con 50/35/20% de las respuestas, 51,00→51,01, 48,42→48,48 y 45,17→45,49; banco idéntico byte a byte. Activado.
- Capa de vecinos mutuos entre clusters finos en Leiden: **descartada**. Sube B-cubed pero la postura cae de 0,85 a 0,35 (une afirmación y refutación por similitud).
- OJO: con pandas 3, `to_datetime(...).astype("int64")` da microsegundos, no nanosegundos. Los análisis de texto casi duplicado, periferia y sincronía se rehicieron con la unidad corregida, y ninguno separa campaña de genuinas sin perder precisión. dias10: sincronía ≤30 s en ≥3 publicaciones, 7 de campaña contra 62 genuinas; periferia ≥20%, 8 contra 96; texto ≥0,6 en ≥2 publicaciones, 119 pares contra 1.010.
- Límite conocido: las cuentas de campaña con 2 o 3 comentarios no se distinguen por co-acción. Es la mayor pérdida que queda (~4 puntos en dias10).

**Viernes 13/11, madrugada (hecho, con Fable 5.2; práctica con IA 53,31 → 54,89):**
- Aceptados, cada uno con el banco completo (ningún recorte baja, FP iguales):
  - alineados + vocabulario propio (604c31d), coordinación 23,73 → 25,00;
  - RESTO_VECINO (dc59786): nada va a ruido por descarte; ruido 0,974 → 0,995;
  - PARTIR_MIXTOS + ANCLA_POSTURA (9f82a8d); uno sin el otro no pasa (la postura sin IA se da vuelta);
  - NUCLEO_ALIN (d84703a): dias7 +7;
  - FRAG_VECINO con 3 variantes (fffbd84): resp20 +3;
  - FRAG_RESPUESTAS (c832ca5): narrativas fantasma, práctica 18 → 13;
  - RAFAGAS (cdd16fa): ult7dias +3,8, pubs50_s2 +4,2, pubs25 +10,8;
  - ORIGEN_INICIO (3baab98): ult7dias +0,13.
  - COORD_RESTO (272a3cb, bug hallado auditando errores por camino): 105 consignas de C1 sin cluster quedaban como ruido; suma +0,12. Efecto secundario conocido: en sin_hospital fija mal el lado de 237 consignas (postura −0,002); arreglarlo partía en dos lados a una operación del set final y se descartó.
  - VOTO_MAYORIA (ffdc565): el voto de hilo por texto exige mayoría ("Ni más ni menos" iba a una sola narrativa); resp20 +0,05.
- Rechazados con medición; quedan como flags apagadas o fuera del repo:
  - RUIDO_HILO (la idea del rival: clusters fuera de hilos → ruido): suma −2,68. El ruido real nunca está en un hilo y lo que se perdía eran raíces sin cluster.
  - HILO_RESCATE: superado por RESTO_VECINO.
  - RUIDO_LETRAS: resp20 −0,05.
  - Aceptar grupos calibrando contra un nulo barajado: la frontera nula es inestable y deja pasar masas genuinas.
  - PARTIR_PUENTES: resp20 −0,42.
  - kNN para minorías de clusters mezclados: dias7 −0,05 y mueve 2.065 comentarios.
  - Respuestas mutuas del mismo lado para C2: limpia, pero no encuentra nada nuevo.
- Postura sin IA: 11 señales probadas para orientar las narrativas 50/50 (raíces, origen, negación, etc.) y ninguna supera al lado mayor. Cada una acierta siempre en unas narrativas y falla en otras. Sin IA queda como una moneda por narrativa balanceada.
- Pruebas nuevas: `tests/test_nulo.py --completo` (pipeline entero, 115/115 reales y 0 en barajados) y `tests/variantes.py`.
- Límite que queda: C2 (textos variados, sin pares repetidos) en recortes de 7 días o con el 50% de las publicaciones.
- Los 2 FP del banco (dias10 y sin_colectivo) son la misma cuenta genuina muy activa (u015025), que cayó en dos ráfagas de C2 y cruzó el umbral de pares por centésimas. No es un bug: sacarla pediría un tope de actividad nuevo.

**Viernes:** correr sin IA y pushear de inmediato; después con IA y pisar. No tocar código ese día.

**No hacer:** cambiar embeddings o UMAP; reajustar umbrales para pasar de 52,5 a 53 en práctica; IA por comentario; usar la fecha de creación de cuentas como regla dura.
