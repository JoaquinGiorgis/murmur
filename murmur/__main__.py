"""murmur: narrativas y coordinación a escala.

Uso:
    python -m murmur --datos datos/final --entrega entrega
"""
import argparse, json, os, platform, resource, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from . import datos, embeddings, coordinacion, narrativas, dashboard


def _hardware():
    try:
        import psutil
        ram = f"{psutil.virtual_memory().total / 2**30:.0f} GB RAM"
    except Exception:
        ram = f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30:.0f} GB RAM" if hasattr(os, "sysconf") else "RAM ?"
    return f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPU, {ram}, sin GPU"


def _pico_mb():
    """Pico de memoria residente del proceso hasta ahora (MB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2**20 if sys.platform == "darwin" else r / 2**10)


def main():
    ap = argparse.ArgumentParser(prog="murmur")
    ap.add_argument("--datos", required=True, type=Path, help="carpeta con comentarios.csv.gz, cuentas.csv.gz, ...")
    ap.add_argument("--entrega", required=True, type=Path, help="carpeta de salida")
    ap.add_argument("--cache", default=Path(".murmur"), type=Path, help="base DuckDB y caché de embeddings")
    ap.add_argument("--dashboard", default=Path("salida/dashboard.html"), type=Path,
                    help="dónde escribir el dashboard (fuera de entrega/, que lleva solo los 4 archivos oficiales)")
    ap.add_argument("--sin-ia", action="store_true", help="no usar LLM aunque haya ANTHROPIC_API_KEY")
    a = ap.parse_args()

    t0 = time.time()
    tiempos, memoria = {}, {}
    log = lambda s: print(s, flush=True)
    a.entrega.mkdir(parents=True, exist_ok=True)

    log("[1/4] ingesta en DuckDB")
    t = time.time()
    con = datos.cargar(a.datos, a.cache / "murmur.duckdb")
    c = datos.comentarios_df(con)
    cuentas = con.execute("SELECT cuenta_id FROM cuentas ORDER BY cuenta_id").df().cuenta_id
    log(f"  {len(c)} comentarios, {len(cuentas)} cuentas")
    tiempos["ingesta"] = round(time.time() - t, 1)
    memoria["ingesta"] = _pico_mb()

    log("[2/4] coordinación")
    t = time.time()
    grupos, ops = coordinacion.detectar(c, cuentas, log=log)
    tiempos["coordinacion"] = round(time.time() - t, 1)
    memoria["coordinacion"] = _pico_mb()

    log("[3/4] embeddings")
    t = time.time()
    E, kid, _, est = embeddings.embeber(c.texto, a.cache, log=log)
    tiempos["embeddings"] = round(time.time() - t, 1)
    memoria["embeddings"] = _pico_mb()

    log("[4/4] narrativas")
    t = time.time()
    llm = None
    if os.environ.get("ANTHROPIC_API_KEY") and not a.sin_ia:
        llm = narrativas.LLM(log=log)
        log(f"  IA activada ({llm.modelo}) para elegir postura y nombrar narrativas")
    df, nd, uso = narrativas.narrativas(c, E, kid, grupos=grupos, log=log, llm=llm)
    tiempos["narrativas"] = round(time.time() - t, 1)
    memoria["narrativas"] = _pico_mb()

    # --- entrega
    df[["comentario_id", "narrativa", "postura"]].to_csv(a.entrega / "comentarios.csv", index=False)
    out_n = nd[["narrativa", "titulo", "frases_clave", "fecha_origen"]].copy()
    out_n["frases_clave"] = out_n.frases_clave.map(
        lambda l: "|".join(map(str, l)) if isinstance(l, (list, tuple)) else ("" if l is None else str(l)))
    out_n.to_csv(a.entrega / "narrativas.csv", index=False)
    grupos.to_csv(a.entrega / "cuentas.csv", index=False)

    # --- resultados también a DuckDB (para el dashboard y para consultar)
    con.register("r_com", df[["comentario_id", "narrativa", "postura"]])
    con.register("r_nar", out_n.merge(nd[["narrativa", "comentarios", "metodo_postura"]], on="narrativa"))
    con.register("r_cue", grupos)
    con.execute("CREATE OR REPLACE TABLE res_comentarios AS SELECT * FROM r_com")
    con.execute("CREATE OR REPLACE TABLE res_narrativas AS SELECT * FROM r_nar")
    con.execute("CREATE OR REPLACE TABLE res_cuentas AS SELECT * FROM r_cue")
    info = con.execute("SELECT cuenta_id, fecha_creacion, seguidores FROM cuentas").df()
    con.close()

    a.dashboard.parent.mkdir(parents=True, exist_ok=True)
    dashboard.generar(df, nd, grupos, ops, info, {
        "comentarios": len(c), "cuentas_marcadas": int((grupos.grupo != "").sum()),
        "segundos": round(time.time() - t0, 1), "usd": round(uso["usd"], 4)}, a.dashboard)

    seg = round(time.time() - t0, 1)
    n = len(c)
    costo = {
        "comentarios_procesados": n,
        "segundos": seg,
        "tokens_ia": int(uso["tokens"]),
        "usd_ia": round(uso["usd"], 4),
        "usd_por_mil_comentarios": round(uso["usd"] / n * 1000, 6) if n else 0,
        "memoria_pico_mb": _pico_mb(),
        "hardware": _hardware(),
        "almacenamiento": "DuckDB local (archivo único) + Parquet para caché de embeddings",
        "notas": (f"embeddings locales ({embeddings.MODELO.split('/')[-1]}) sobre {est['textos_unicos']} textos únicos; "
                  f"IA {'solo para elegir postura y nombrar narrativas (' + str(uso['llamadas']) + ' llamadas)' if uso['llamadas'] else 'no usada'}; "
                  + (f"IA con fallos ({llm.fallos} llamadas{', ' + llm.abortado if llm.abortado else ''}): esas narrativas usan postura heurística; "
                     if llm is not None and (llm.fallos or llm.abortado) else "") +
                  f"tiempos por etapa (s): {tiempos}; pico de memoria acumulado al terminar cada etapa (MB): {memoria}"),
    }
    (a.entrega / "costo.json").write_text(json.dumps(costo, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"listo en {seg}s (pico de memoria {_pico_mb()} MB) -> {a.entrega}/  dashboard: {a.dashboard}")


if __name__ == "__main__":
    main()
