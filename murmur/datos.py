"""Ingesta: CSV comprimidos -> DuckDB (tablas limpias) + Parquet.

DuckDB lee los .csv.gz directamente, tipa y limpia en SQL, y deja todo en un
archivo local. El resto del pipeline lee de ahí en columnas (vía Arrow), así
que no hay base que levantar ni servidor que mantener.
"""
import csv, gzip
from pathlib import Path
import duckdb
import pandas as pd

ZONA = "America/Argentina/Buenos_Aires"
COLUMNAS = ["comentario_id", "publicacion_id", "cuenta_id", "fecha", "texto", "respuesta_a", "me_gusta"]


def _csv(path: Path, encoding: str = "utf-8") -> str:
    # strict_mode=false: una comilla rara en un texto no hace descartar el resto del archivo
    return (f"read_csv('{path.as_posix()}', header=true, all_varchar=true, ignore_errors=true, "
            f"quote='\"', escape='\"', strict_mode=false, encoding='{encoding}')")


def _archivo(datos: Path, nombre: str) -> Path:
    """nombre.csv.gz, o nombre.csv si viene sin comprimir."""
    gz = datos / f"{nombre}.csv.gz"
    return gz if gz.exists() or not (datos / f"{nombre}.csv").exists() else datos / f"{nombre}.csv"


def _contar_registros(path: Path):
    """Registros CSV según el lector de Python (independiente de DuckDB), o None si no se puede leer."""
    try:
        abrir = gzip.open if path.suffix == ".gz" else open
        with abrir(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
            return sum(1 for fila in csv.reader(f) if fila) - 1
    except Exception:
        return None


def cargar(datos: Path, db_path: Path, log=print) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    # fechas sin zona horaria = hora argentina, aunque la máquina esté en UTC
    con.execute(f"SET TimeZone='{ZONA}'")
    com = _archivo(datos, "comentarios")
    cue = _archivo(datos, "cuentas")
    pub = _archivo(datos, "publicaciones")
    if not com.exists():
        raise SystemExit(f"ERROR: no encuentro {datos}/comentarios.csv.gz (ni .csv)")

    # Comentarios: ids únicos, fechas válidas, texto nunca nulo, respuesta_a solo si apunta a algo que existe.
    con.execute(f"""
        CREATE OR REPLACE TABLE comentarios_raw AS SELECT * FROM {_csv(com)};
    """)
    # control de filas: DuckDB descarta en silencio las que no puede leer
    n_raw = con.execute("SELECT count(*) FROM comentarios_raw").fetchone()[0]
    n_csv = _contar_registros(com)
    if n_csv is not None and n_csv > n_raw:  # ¿otra codificación? (latin-1 corta la lectura en el primer acento)
        try:
            con.execute(f"CREATE OR REPLACE TABLE comentarios_l1 AS SELECT * FROM {_csv(com, 'latin-1')}")
            n_l1 = con.execute("SELECT count(*) FROM comentarios_l1").fetchone()[0]
            if n_l1 > n_raw:
                log(f"  AVISO: comentarios no está en UTF-8; se leyó como latin-1 ({n_l1} filas en vez de {n_raw})")
                con.execute("DROP TABLE comentarios_raw; ALTER TABLE comentarios_l1 RENAME TO comentarios_raw")
                n_raw = n_l1
            else:
                con.execute("DROP TABLE comentarios_l1")
        except Exception:
            pass
    if n_csv is not None and n_csv != n_raw:
        log(f"  AVISO: el archivo tiene {n_csv} registros y se leyeron {n_raw}; "
            f"{n_csv - n_raw} filas ilegibles quedan fuera de la entrega")
    cols = {r[0].lower() for r in con.execute("DESCRIBE comentarios_raw").fetchall()}
    if "comentario_id" not in cols:
        raise SystemExit(f"ERROR: comentarios sin columna comentario_id (columnas: {sorted(cols)})")
    for col in COLUMNAS:
        if col not in cols:
            log(f"  AVISO: falta la columna {col} en comentarios; se toma vacía")
            con.execute(f"ALTER TABLE comentarios_raw ADD COLUMN {col} VARCHAR")
    con.execute(f"""
        CREATE OR REPLACE TABLE comentarios AS
        WITH base AS (
            SELECT trim(comentario_id) AS comentario_id,
                   coalesce(nullif(trim(publicacion_id), ''), '_sin_pub_' || trim(comentario_id)) AS publicacion_id,
                   trim(cuenta_id) AS cuenta_id,
                   try_cast(fecha AS TIMESTAMPTZ) AS fecha,
                   coalesce(texto, '') AS texto,
                   nullif(trim(respuesta_a), '') AS respuesta_a,
                   coalesce(try_cast(me_gusta AS INTEGER), 0) AS me_gusta,
                   row_number() OVER (PARTITION BY trim(comentario_id) ORDER BY fecha, texto) AS rn
            FROM comentarios_raw
            WHERE comentario_id IS NOT NULL AND trim(comentario_id) <> ''
        )
        SELECT b.comentario_id, b.publicacion_id, coalesce(b.cuenta_id, 'desconocida') AS cuenta_id,
               b.fecha, b.texto,
               CASE WHEN b.respuesta_a IN (SELECT comentario_id FROM base) THEN b.respuesta_a END AS respuesta_a,
               b.me_gusta
        FROM base b WHERE rn = 1;
    """)
    n_ok = con.execute("SELECT count(*) FROM comentarios").fetchone()[0]
    if n_ok < n_raw:
        log(f"  {n_raw - n_ok} filas descartadas por id vacío o duplicado")
    n_mal = con.execute("SELECT count(*) FROM comentarios WHERE fecha IS NULL").fetchone()[0]
    if n_ok and n_mal == n_ok:
        ej = con.execute("SELECT fecha FROM comentarios_raw WHERE fecha IS NOT NULL LIMIT 1").fetchone()
        raise SystemExit(f"ERROR: ninguna fecha se pudo leer (ejemplo: {ej[0] if ej else None!r}); "
                         f"se esperan fechas ISO 8601, p. ej. 2026-09-01T07:32:30-03:00")
    if n_mal:
        log(f"  AVISO: {n_mal} fechas ilegibles; se imputan con la mediana de su publicación")
    # Fechas ilegibles: se imputan con la mediana de la publicación.
    con.execute("""
        CREATE OR REPLACE TABLE comentarios AS
        SELECT c.* REPLACE (coalesce(c.fecha, p.f, (SELECT min(fecha) FROM comentarios)) AS fecha)
        FROM comentarios c
        LEFT JOIN (SELECT publicacion_id, to_timestamp(median(epoch(fecha))) AS f FROM comentarios GROUP BY 1) p
        USING (publicacion_id);
    """)
    if cue.exists():
        con.execute(f"""
            CREATE OR REPLACE TABLE cuentas AS
            SELECT trim(cuenta_id) AS cuenta_id, any_value(usuario) AS usuario,
                   any_value(try_cast(fecha_creacion AS DATE)) AS fecha_creacion,
                   any_value(coalesce(try_cast(seguidores AS INTEGER), 0)) AS seguidores,
                   any_value(coalesce(try_cast(siguiendo AS INTEGER), 0)) AS siguiendo
            FROM {_csv(cue)} WHERE cuenta_id IS NOT NULL GROUP BY 1;
        """)
    else:
        con.execute("CREATE OR REPLACE TABLE cuentas (cuenta_id VARCHAR, usuario VARCHAR, fecha_creacion DATE, seguidores INTEGER, siguiendo INTEGER)")
    # Toda cuenta que comentó tiene fila, aunque falte en cuentas.csv.
    con.execute("""
        INSERT INTO cuentas SELECT DISTINCT cuenta_id, NULL, NULL, 0, 0 FROM comentarios
        WHERE cuenta_id NOT IN (SELECT cuenta_id FROM cuentas);
    """)
    if pub.exists():
        con.execute(f"CREATE OR REPLACE TABLE publicaciones AS SELECT * FROM {_csv(pub)}")
    con.execute("DROP TABLE IF EXISTS comentarios_raw")
    return con


def comentarios_df(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT comentario_id, publicacion_id, cuenta_id, epoch(fecha)::BIGINT AS ts,
               strftime(fecha AT TIME ZONE 'America/Argentina/Buenos_Aires', '%Y-%m-%dT%H:%M:%S') || '-03:00' AS fecha,
               texto, respuesta_a, me_gusta
        FROM comentarios ORDER BY comentario_id
    """).df()
    return df


def exportar_parquet(con, carpeta: Path):
    carpeta.mkdir(parents=True, exist_ok=True)
    for t in ("comentarios", "cuentas"):
        con.execute(f"COPY {t} TO '{(carpeta / (t + '.parquet')).as_posix()}' (FORMAT PARQUET)")
