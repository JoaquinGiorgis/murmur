# murmur

**Qué se dice y quién lo está armando.** Pipeline para el desafío QualyData *Narrativas y coordinación a escala* (Hackathon Argentina Space 2026).

Una *murmuration* es una bandada de miles de estorninos que se mueve como un solo bicho. De afuera ves un patrón y no sabés si alguien lo dirige. Ese es el problema: separar el murmullo genuino de la bandada que alguien está moviendo.

**En el set de práctica: 53,3 de 55 puntos automáticos con IA (52,6 sin IA), precisión 1,0 en cuentas coordinadas (ninguna cuenta genuina acusada), USD 0,03 de IA para 200.000 comentarios y unos 10 minutos en una notebook sin GPU.** Fuera de práctica, en 12 recortes y 7 variantes de datos, no acusa a ninguna cuenta genuina nueva (ver [Robustez y generalización](#robustez-y-generalización)).

## Cómo correrlo

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m murmur --datos datos/final --entrega entrega
```

Python 3.10 o más nuevo (medido con 3.11). Las versiones de `requirements.txt` están fijas: son las del puntaje publicado abajo. El modelo de embeddings (unos 470 MB) se baja solo en la primera corrida y después se carga del disco; para correr sin red, bajarlo antes:

```bash
python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
HF_HUB_OFFLINE=1 python -m murmur --datos datos/final --entrega entrega
```

Un solo comando, de punta a punta. Deja en `entrega/` exactamente los cuatro archivos del formato oficial (`comentarios.csv`, `narrativas.csv`, `cuentas.csv` y `costo.json`). Aparte, en `salida/dashboard.html` (se cambia con `--dashboard`), deja un dashboard autocontenido: qué circula, desde cuándo y quién lo empuja. Se abre en cualquier navegador, sin servidor.

- `--cache .murmur`: dónde quedan la base DuckDB y la caché de embeddings. Por defecto es `.murmur/`.
- IA opcional: con `ANTHROPIC_API_KEY` en el entorno, se hacen **8 llamadas por narrativa** (unas 100 en total) para decidir qué lado afirma y para nombrarla. Sin clave, o con `--sin-ia`, corre igual con heurísticos. El modelo se cambia con `MURMUR_LLM_MODELO`, y los precios para `costo.json` con `MURMUR_USD_MTOK_IN` y `MURMUR_USD_MTOK_OUT` (por defecto, los de Claude Haiku 5.5: USD 0,10 y 0,50 por millón de tokens). Si la API falla, reintenta con espera creciente; si sigue fallando, avisa en el log y en `costo.json` y termina sin IA, sin colgarse.

Para medirse contra el set de práctica:

```bash
python -m murmur --datos datos/practica --entrega salida/practica
python ../desafio-qualydata/evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica
```

## Resultado en el set de práctica

| Criterio | Línea base oficial | murmur sin IA | murmur con IA |
|---|---|---|---|
| Agrupamiento de narrativas (B-cubed F1) | 0,35 | 0,983 | **0,983** |
| Detección de ruido (F1) | 0,67 | 0,974 | **0,974** |
| Postura (macro-F1) | 0,49 | 0,867 | **0,992** |
| Origen de emergentes | 0,33 | 1,00 | **1,00** |
| Cuentas coordinadas (F1) | 0,12 | 0,959 | **0,959** (precisión 1,0) |
| Agrupamiento de operaciones (B-cubed F1) | 0,28 | 0,925 | **0,925** |
| **Total automático** | **16,8 / 55** | **52,56 / 55** | **53,31 / 55** |

Costo en IA: **USD 0** sin IA, **USD 0,03** con IA (104 llamadas a Claude Haiku 5.5 y 165.000 tokens para 200.000 comentarios: menos de dos décimas de centavo de dólar cada mil comentarios). Etiquetar cada comentario con IA costaría cientos de veces más; acá se paga por narrativa, no por comentario. Medido con versiones fijas desde un clon limpio y sin red.

### Tiempo, memoria y disco

Corrida en frío sin IA sobre el set de práctica (200.000 comentarios), en una notebook de 8 CPU y 16 GB, sin GPU:

| Etapa | Tiempo | Pico de memoria acumulado |
|---|---|---|
| Ingesta y limpieza en DuckDB | 2 s | 0,5 GB |
| Coordinación | 6 s | 0,9 GB |
| Embeddings (76.834 textos únicos) | 4–7 min | 1,7 GB |
| Narrativas (UMAP, HDBSCAN, Leiden) | 2,8 min | 2,1 GB |
| **Total** | **unos 10 min** | **2,1 GB** |

Con la caché de embeddings, una segunda corrida tarda unos 3 minutos sin IA y unos 6 a 8 con IA. En disco quedan 33 MB de base DuckDB y 55 MB de caché de embeddings.

**A 1 millón de comentarios (medido):** cinco copias del set de práctica con ids distintos (1.000.000 de comentarios, 80.000 cuentas, 2.500 publicaciones) corren en **3 minutos y 2,6 GB** sin contar embeddings: ingesta 5 s, coordinación 20 s, narrativas 2,4 min. Puntaje 49,7 con precisión 1,0 (medido antes de bajar el grado mínimo a 2,5). Los embeddings quedaron en caché porque los textos se repiten; con 1 millón de comentarios reales y 38% de textos únicos serían unos 380.000 textos nuevos, 20 a 35 minutos de embeddings en CPU (eso sí es extrapolación). La IA no crece: cuesta lo mismo con 200.000 que con 1 millón de comentarios, porque se paga por narrativa. Límite honesto: algunos umbrales absolutos (repeticiones para marcar ruido) están calibrados para 100.000 a 200.000 comentarios; a 1 millón el ruido baja de 0,97 a 0,92.

## Cómo funciona

### Tarea A: narrativas

1. **Embeddings locales** (`paraphrase-multilingual-MiniLM-L12-v2`, corre en CPU) sobre los **textos únicos normalizados**. Los 200.000 comentarios se reducen a unos 77.000 textos distintos: plantillas, copias y respuestas cortas se embeben una sola vez.
2. **Clusters finos** con UMAP más HDBSCAN: unos 500 clusters con 98% de pureza. Cada uno es una variante de un mismo reclamo, casi siempre de una sola postura. HDBSCAN decide cuántos hay; no fijamos *k*.
3. **Las respuestas cosen los clusters en narrativas.** Por las reglas del desafío, una respuesta corta ("Tal cual") hereda la narrativa y la postura del padre, y una respuesta larga es misma narrativa con postura opuesta. Eso arma un grafo entre clusters que agrupamos con Leiden.
   - El signo de cada arista da un **2-coloreo** que separa el lado que afirma del que refuta.
   - No depende del tema ni de palabras clave, y aguanta que la redacción cambie con los días: dos redacciones distintas del mismo reclamo quedan unidas porque las refutan los mismos.
4. **Ruido:** clusters que casi no participan de conversaciones, más textos muy repetidos que nunca reciben ni hacen respuestas (avisos, chistes, "primero!").
5. **Postura:** el 2-coloreo dice qué comentarios están del mismo lado; falta decidir cuál lado **apoya**. Con IA, un modelo chico lo decide mirando 16 ejemplos limpios de cada lado (sin muletillas ni emojis, que confunden). Le preguntamos 8 veces alternando qué grupo va primero, porque estos modelos tienen sesgo de posición, y gana la mayoría; si empata, decide el heurístico. Sin IA: si una operación coordinada se concentra en un lado, ese lado afirma (las operaciones amplifican reclamos); si no, gana el lado más grande. *Este es el punto débil sin IA: en narrativas 50/50 es casi una moneda.*
6. **Fecha de origen:** el primer momento en que la narrativa junta un volumen sostenido en 3 horas (0,5% de su tamaño, entre 10 y 25 comentarios), no el primer comentario suelto, que suele ser un error de asignación. El tope de 25 evita que una narrativa grande que nace lento, como una campaña nocturna, quede fechada horas tarde.
7. **Títulos y frases clave sin IA:** el texto más frecuente y central del lado que afirma, sin las muletillas que se repiten en todas las narrativas ("te digo algo", "lamentable"), y frases clave por c-TF-IDF.

### Tarea B: coordinación

Una ráfaga no alcanza: lo que delata a una operación es **la repetición**.

1. **Co-acción:** dos cuentas comentan la misma publicación dentro de 10 minutos, medido en dos grillas desfasadas para no cortar ráfagas en el borde.
2. Por cada par de cuentas, `k` es cuántas publicaciones distintas comparten así, y `coseno = k / √(pubs_a · pubs_b)`. Eso descuenta a las cuentas muy activas (que coinciden con todos por volumen) y a los virales genuinos (mucha gente distinta, una sola vez).
3. Se suman las respuestas entre cuentas, porque hay operaciones que se responden entre sí.
4. Leiden sobre el grafo de pares sospechosos. Se marcan las comunidades **chicas y densas**: una operación es un grupo cerrado que coincide una y otra vez, mientras que la opinión genuina forma árboles ralos.
5. **Poda del núcleo:** dentro de cada comunidad se descartan, de forma iterativa, las cuentas cuyo lazo interno (co-acciones más respuestas entre miembros) queda por debajo de la mitad de la mediana del grupo. Así salen las cuentas genuinas muy activas que Leiden arrastró por volumen.
6. **Densidad mínima:** el grupo podado tiene que tener densidad (grado medio / (n − 1)) de al menos 0,09 y grado medio de al menos 2,5. Las operaciones reales dan entre 0,12 y 0,32 de densidad; los grupos de usuarios genuinos muy activos, como mucho 0,063.
7. **Satélites:** las cuentas de una campaña que comentan solo 2 o 3 veces no llegan a repetirse lo suficiente. Una cuenta se suma a una operación ya detectada solo si se dan tres cosas juntas: en al menos 2 publicaciones comenta a menos de 10 minutos de un comentario del núcleo, con un texto casi igual (similitud ≥ 0,8, al menos 4 palabras), y eso es al menos la mitad de su actividad. Cada señal sola no separa campaña de genuinos (medido); las tres juntas sí, sin sumar ningún falso positivo en el banco. Nunca crea operaciones nuevas.


El contenido que empuja una operación nunca se marca como ruido, aunque sea una plantilla que nadie responde: se asigna a la narrativa y al lado más cercanos por significado.

No se usan la fecha de creación de las cuentas ni los hashtags. En el set de práctica separan perfecto, pero sería sobreajustar a ese set.

## Robustez y generalización

Para no ajustarnos al set de práctica, cada cambio pasa por un **banco de estrés** (`tests/banco_estres.py`): corre el pipeline sin IA sobre 12 recortes del set (por días, por tema y por fracción de publicaciones), con su verdad recalculada, y los puntúa con el scorer oficial. Un cambio se acepta solo si no baja el puntaje de práctica ni sube los falsos positivos en ningún recorte.

**Test nulo de coordinación** (`tests/test_nulo.py`, sin etiquetas): barajamos las fechas, o los autores, dentro de cada publicación. Eso conserva el volumen, el ritmo, los textos y las respuestas, pero rompe la sincronía entre cuentas. Si el detector marcara a alguien en esos datos, estaría viendo coordinación donde no la hay. En práctica, los datos reales marcan 101 cuentas, todas de campañas; las 6 versiones barajadas marcan 0. Es el modelo nulo por permutación que proponen Pacheco et al. (2021).

| Recorte (sin IA) | Puntaje | Cuentas marcadas por error |
|---|---|---|
| Práctica completa | 52,56 | 0 |
| Primeros 10 días (el más parecido al set final) | 52,62 | 1 |
| Primeros 7 días / últimos 7 días | 43,85 / 38,32 | 0 / 0 |
| Sin un tema (litio / hospital / colectivos) | 38,50 / 52,00 / 53,26 | 0 / 0 / 1 |
| Un solo tema (litio / hospital / colectivos) | 53,30 / 41,71 / 26,31 | 0 / 0 / 0 |
| 50% de las publicaciones (dos sorteos) / 25% | 36,17 / 41,20 / 26,71 | 0 / 0 / 0 |

Los puntajes bajan cuando el recorte deja a una campaña con muy pocos comentarios: el detector prefiere no acusar antes que acusar mal. En total hay **2 falsos positivos en 13 corridas** (antes de agregar el filtro de densidad eran 3.641).

Además lo corrimos sobre variantes del set con cambios de forma y de contenido (todas sin IA, precisión 1,0 en todas):

| Variante | Puntaje |
|---|---|
| Fechas corridas 150 días y en UTC (`Z`) | 52,56 |
| Ids nuevos no secuenciales y filas mezcladas | 52,59 |
| CSV sucio: textos vacíos, filas duplicadas, respuestas a ids inexistentes, fechas ilegibles, cuentas faltantes, columnas extra, CRLF | 52,33 |
| Mitad de las respuestas / 35% / 20% | 51,80 / 49,43 / 46,90 |
| Unos 150.000 comentarios en 28 días con otra mezcla de temas | 40,03 |
| Sin ninguna respuesta | 32,55 (no se cae) |
| Fechas `dd/mm/aaaa` | corta con un mensaje claro que pide fechas ISO 8601 |

La ingesta también acepta `.csv` sin comprimir, columnas faltantes (avisa), texto en latin-1 (lo detecta y lo relee) y comillas mal cerradas sin perder filas, y compara cuántas filas leyó contra el archivo. Si una etapa falla, la otra se entrega igual: los cuatro archivos siempre se escriben, y el proceso termina con código 2 y un aviso visible.

**Límites conocidos:**
- Las cuentas de campaña con un solo comentario no se pueden distinguir por comportamiento: no hay repetición que medir.
- Con pocas publicaciones o un solo tema, una campaña puede quedar sin núcleo detectable y entonces no se marca.
- Sin IA, la postura de narrativas parejas (50/50) es casi una moneda; con IA queda en 0,99.
- El puntaje se midió en macOS arm64; en otra arquitectura UMAP puede mover decimales.

## Dónde guardamos los datos y por qué DuckDB

- **DuckDB, un archivo local** (`.murmur/murmur.duckdb`), lee los `.csv.gz` directo, los tipa y los limpia en SQL: ids duplicados, fechas ilegibles, `respuesta_a` que apunta a comentarios inexistentes, texto nulo, cuentas que comentan pero no están en `cuentas.csv`. No hay servidor que levantar.
- Las consultas del pipeline son analíticas y por columnas (agrupar por publicación, por cuenta, por ventana de tiempo). Eso es justo lo que hace bien un motor columnar; una base transaccional como PostgreSQL rinde peor acá.
- Los resultados también quedan en la base (`res_comentarios`, `res_narrativas`, `res_cuentas`) para consultarlos o armar el dashboard. Algunas consultas de ejemplo (`duckdb .murmur/murmur.duckdb`):

```sql
-- narrativas por volumen y qué fracción de sus comentarios la apoya
SELECT n.narrativa, n.titulo, count(*) AS comentarios, round(avg((r.postura = 'apoya')::INT), 2) AS frac_apoya
FROM res_comentarios r JOIN res_narrativas n USING (narrativa)
GROUP BY ALL ORDER BY comentarios DESC;

-- curva diaria de una narrativa
SELECT date_trunc('day', c.fecha) AS dia, count(*) AS comentarios
FROM res_comentarios r JOIN comentarios c USING (comentario_id)
WHERE r.narrativa = 'n01' GROUP BY dia ORDER BY dia;

-- cada operación: tamaño, antigüedad mediana de sus cuentas y narrativa que más empuja
SELECT g.grupo, count(DISTINCT g.cuenta_id) AS cuentas,
       median(date_diff('day', cu.fecha_creacion, DATE '2026-09-01')) AS dias_antiguedad,
       mode(r.narrativa) AS narrativa_principal
FROM res_cuentas g JOIN cuentas cu USING (cuenta_id) JOIN comentarios c USING (cuenta_id)
JOIN res_comentarios r USING (comentario_id)
WHERE g.grupo <> '' GROUP BY g.grupo ORDER BY cuentas DESC;

-- publicaciones donde más comentan las operaciones
SELECT p.publicacion_id, p.titular, count(*) FILTER (WHERE g.grupo <> '') AS de_operaciones, count(*) AS total
FROM comentarios c JOIN res_cuentas g USING (cuenta_id) JOIN publicaciones p USING (publicacion_id)
GROUP BY ALL ORDER BY de_operaciones DESC LIMIT 10;
```
- **Caché de embeddings en Parquet**, indexada por hash del texto normalizado: una segunda corrida, o un flujo incremental, solo paga por los textos nuevos.

### Cómo escalaría a millones de comentarios por día

- **Ingesta:** Parquet particionado por día en almacenamiento de objetos. DuckDB lo lee igual, y la misma lógica SQL pasa a ClickHouse si hay que servir consultas en tiempo real.
- **Embeddings:** es la única etapa cara y es lineal en *textos únicos*, no en comentarios. Con la caché, el costo diario es solo lo nuevo, y se paraleliza trivialmente (por lotes, con GPU o un servicio de embeddings).
- **Narrativas:** se ajustan los clusters una vez por día y los comentarios nuevos se asignan por similitud al centroide más cercano, que es barato. Las respuestas siguen cosiendo y heredando en línea.
- **Coordinación:** el grafo de co-acción se actualiza con ventanas deslizantes; las matrices son ralas y se pueden acumular por día.
- **IA:** el costo es fijo por narrativa (8 llamadas cada una), no por comentario, así que no crece con el volumen.

## Código y librerías previas al evento

Todo el código de este repo se escribió durante el hackathon. Usamos librerías públicas: DuckDB, pandas, NumPy, SciPy, scikit-learn, UMAP, HDBSCAN, igraph, leidenalg y sentence-transformers, con el modelo público `paraphrase-multilingual-MiniLM-L12-v2`. Nada se etiquetó a mano.

## Licencia

MIT.
