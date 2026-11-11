"""Ingesta: CSV comprimidos -> DuckDB (tablas limpias) + Parquet.

DuckDB lee los .csv.gz directamente, tipa y limpia en SQL, y deja todo en un
archivo local. El resto del pipeline lee de ahí en columnas (vía Arrow), así
que no hay base que levantar ni servidor que mantener.
"""
from pathlib import Path
import duckdb
import pandas as pd


def _csv(path: Path) -> str:
    return f"read_csv('{path.as_posix()}', header=true, all_varchar=true, ignore_errors=true, quote='\"', escape='\"')"


def cargar(datos: Path, db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    com = datos / "comentarios.csv.gz"
    cue = datos / "cuentas.csv.gz"
    pub = datos / "publicaciones.csv.gz"

    # Comentarios: ids únicos, fechas válidas, texto nunca nulo, respuesta_a solo si apunta a algo que existe.
    con.execute(f"""
        CREATE OR REPLACE TABLE comentarios_raw AS SELECT * FROM {_csv(com)};
        CREATE OR REPLACE TABLE comentarios AS
        WITH base AS (
            SELECT trim(comentario_id) AS comentario_id,
                   trim(publicacion_id) AS publicacion_id,
                   trim(cuenta_id) AS cuenta_id,
                   try_cast(fecha AS TIMESTAMPTZ) AS fecha,
                   coalesce(texto, '') AS texto,
                   nullif(trim(respuesta_a), '') AS respuesta_a,
                   coalesce(try_cast(me_gusta AS INTEGER), 0) AS me_gusta,
                   row_number() OVER (PARTITION BY trim(comentario_id) ORDER BY fecha) AS rn
            FROM comentarios_raw
            WHERE comentario_id IS NOT NULL AND trim(comentario_id) <> ''
        )
        SELECT b.comentario_id, b.publicacion_id, coalesce(b.cuenta_id, 'desconocida') AS cuenta_id,
               b.fecha, b.texto,
               CASE WHEN b.respuesta_a IN (SELECT comentario_id FROM base) THEN b.respuesta_a END AS respuesta_a,
               b.me_gusta
        FROM base b WHERE rn = 1;
    """)
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
