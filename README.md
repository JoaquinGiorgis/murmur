# murmur

**Qué se dice y quién lo está armando.** Pipeline para el desafío QualyData *Narrativas y coordinación a escala* (Hackathon Argentina Space 2026).

Una *murmuration* es una bandada de miles de estorninos que se mueve como un solo bicho. De afuera ves un patrón y no sabés si alguien lo dirige. Ese es el problema: separar el murmullo genuino de la bandada que alguien está moviendo.

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m murmur --datos datos/final --entrega entrega
```

Un solo comando, de punta a punta. Deja en `entrega/` los cuatro archivos del formato oficial (`comentarios.csv`, `narrativas.csv`, `cuentas.csv` y `costo.json`) y un `dashboard.html` autocontenido: qué circula, desde cuándo y quién lo empuja. Se abre en cualquier navegador, sin servidor.

- `--cache .murmur`: dónde quedan la base DuckDB y la caché de embeddings. Por defecto es `.murmur/`.
- IA opcional: con `ANTHROPIC_API_KEY` en el entorno, se hace **una llamada por narrativa** (unas 15 en total) para decidir qué lado afirma y para nombrarla. Sin clave, o con `--sin-ia`, corre igual con heurísticos. El modelo se cambia con `MURMUR_LLM_MODELO`, y los precios para `costo.json` con `MURMUR_USD_MTOK_IN` y `MURMUR_USD_MTOK_OUT`.

Para medirse contra el set de práctica:

```bash
python -m murmur --datos datos/practica --entrega salida/practica
python ../desafio-qualydata/evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica
```

## Resultado en el set de práctica (sin IA)

| Criterio | Línea base oficial | murmur |
|---|---|---|
| Agrupamiento de narrativas (B-cubed F1) | 0,35 | **0,977** |
| Detección de ruido (F1) | 0,67 | **0,979** |
| Postura (macro-F1) | 0,49 | 0,76 |
| Origen de emergentes | 0,33 | **1,00** |
| Cuentas coordinadas (F1) | 0,12 | **0,94** (precisión 1,0) |
| Agrupamiento de operaciones (B-cubed F1) | 0,28 | **0,88** |
| **Total automático** | **16,8 / 55** | **51,0 / 55** |

Costo en IA: **USD 0**. En una máquina de 2 CPU y 8 GB sin GPU, la corrida en frío tarda unos 10 minutos (6,5 de embeddings); con la caché de embeddings, unos 3.

## Cómo funciona

### Tarea A: narrativas

1. **Embeddings locales** (`paraphrase-multilingual-MiniLM-L12-v2`, unos 120 MB, corre en CPU) sobre los **textos únicos normalizados**. Los 200.000 comentarios se reducen a unos 77.000 textos distintos: plantillas, copias y respuestas cortas se embeben una sola vez.
2. **Clusters finos** con UMAP más HDBSCAN: unos 500 clusters con 98% de pureza. Cada uno es una variante de un mismo reclamo, casi siempre de una sola postura. HDBSCAN decide cuántos hay; no fijamos *k*.
3. **Las respuestas cosen los clusters en narrativas.** Por las reglas del desafío, una respuesta corta ("Tal cual") hereda la narrativa y la postura del padre, y una respuesta larga es misma narrativa con postura opuesta. Eso arma un grafo entre clusters que agrupamos con Leiden.
   - El signo de cada arista da un **2-coloreo** que separa el lado que afirma del que refuta.
   - No depende del tema ni de palabras clave, y aguanta que la redacción cambie con los días: dos redacciones distintas del mismo reclamo quedan unidas porque las refutan los mismos.
4. **Ruido:** clusters que casi no participan de conversaciones, más textos muy repetidos que nunca reciben ni hacen respuestas (avisos, chistes, "primero!").
5. **Postura:** el 2-coloreo dice qué comentarios están del mismo lado; falta decidir cuál lado **apoya**. Con IA, un modelo chico lo decide mirando 8 ejemplos de cada lado. Sin IA: si una operación coordinada se concentra en un lado, ese lado afirma (las operaciones amplifican reclamos); si no, gana el lado más grande. *Este es el punto débil sin IA: en narrativas 50/50 es casi una moneda.*
6. **Fecha de origen:** el primer momento en que la narrativa junta un volumen sostenido en 3 horas, no el primer comentario suelto, que suele ser un error de asignación.
7. **Títulos y frases clave sin IA:** el texto más frecuente y central del lado que afirma, sin las muletillas que se repiten en todas las narrativas ("te digo algo", "lamentable"), y frases clave por c-TF-IDF.

### Tarea B: coordinación

Una ráfaga no alcanza: lo que delata a una operación es **la repetición**.

1. **Co-acción:** dos cuentas comentan la misma publicación dentro de 10 minutos, medido en dos grillas desfasadas para no cortar ráfagas en el borde.
2. Por cada par de cuentas, `k` es cuántas publicaciones distintas comparten así, y `coseno = k / √(pubs_a · pubs_b)`. Eso descuenta a las cuentas muy activas (que coinciden con todos por volumen) y a los virales genuinos (mucha gente distinta, una sola vez).
3. Se suman las respuestas entre cuentas, porque hay operaciones que se responden entre sí.
4. Leiden sobre el grafo de pares sospechosos. Se marcan las comunidades **chicas y densas**: una operación es un grupo cerrado que coincide una y otra vez, mientras que la opinión genuina forma árboles ralos.
5. **Poda del núcleo:** dentro de cada comunidad se descartan, de forma iterativa, las cuentas cuyo lazo interno (co-acciones más respuestas entre miembros) queda por debajo de la mitad de la mediana del grupo. Así salen las cuentas genuinas muy activas que Leiden arrastró por volumen.

El contenido que empuja una operación nunca se marca como ruido, aunque sea una plantilla que nadie responde: se asigna a la narrativa y al lado más cercanos por significado.

No se usan la fecha de creación de las cuentas ni los hashtags. En el set de práctica separan perfecto, pero sería sobreajustar a ese set.

## Dónde guardamos los datos y por qué DuckDB

- **DuckDB, un archivo local** (`.murmur/murmur.duckdb`), lee los `.csv.gz` directo, los tipa y los limpia en SQL: ids duplicados, fechas ilegibles, `respuesta_a` que apunta a comentarios inexistentes, texto nulo, cuentas que comentan pero no están en `cuentas.csv`. No hay servidor que levantar.
- Las consultas del pipeline son analíticas y por columnas (agrupar por publicación, por cuenta, por ventana de tiempo). Eso es justo lo que hace bien un motor columnar; una base transaccional como PostgreSQL rinde peor acá.
- Los resultados también quedan en la base (`res_comentarios`, `res_narrativas`, `res_cuentas`) para consultarlos o armar el dashboard.
- **Caché de embeddings en Parquet**, indexada por hash del texto normalizado: una segunda corrida, o un flujo incremental, solo paga por los textos nuevos.

### Cómo escalaría a millones de comentarios por día

- **Ingesta:** Parquet particionado por día en almacenamiento de objetos. DuckDB lo lee igual, y la misma lógica SQL pasa a ClickHouse si hay que servir consultas en tiempo real.
- **Embeddings:** es la única etapa cara y es lineal en *textos únicos*, no en comentarios. Con la caché, el costo diario es solo lo nuevo, y se paraleliza trivialmente (por lotes, con GPU o un servicio de embeddings).
- **Narrativas:** se ajustan los clusters una vez por día y los comentarios nuevos se asignan por similitud al centroide más cercano, que es barato. Las respuestas siguen cosiendo y heredando en línea.
- **Coordinación:** el grafo de co-acción se actualiza con ventanas deslizantes; las matrices son ralas y se pueden acumular por día.
- **IA:** el costo es fijo por narrativa (unas 15 llamadas), no por comentario, así que no crece con el volumen.

## Código y librerías previas al evento

Todo el código de este repo se escribió durante el hackathon. Usamos librerías públicas: DuckDB, pandas, NumPy, SciPy, scikit-learn, UMAP, HDBSCAN, igraph, leidenalg y sentence-transformers, con el modelo público `paraphrase-multilingual-MiniLM-L12-v2`. Nada se etiquetó a mano.

## Licencia

MIT.
