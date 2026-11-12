"""Banco de estrés: murmur fuera del set de práctica.

Arma recortes del set de práctica (por tiempo, por tema y por volumen) con su verdad
consistente, corre `python -m murmur --sin-ia` en cada uno y lo puntúa con el scorer
oficial. Sirve para aceptar o rechazar cambios: ninguno puede bajar el puntaje de
práctica ni subir los falsos positivos de coordinación en ningún recorte.

Uso:
    python tests/banco_estres.py --etiqueta antes            # corre todo y guarda .murmur/banco/antes.json
    python tests/banco_estres.py --etiqueta despues
    python tests/banco_estres.py --comparar antes despues    # tabla y veredicto (exit 1 si falla la regla)

Recortes, datos y salidas quedan en .murmur/banco/ (fuera de git).
"""
import argparse, gzip, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
BANCO = RAIZ / ".murmur" / "banco"
SCORER = RAIZ.parent / "desafio-qualydata" / "evaluacion" / "puntuar.py"


def _dias(n, desde_final=False):
    def f(c, pub, v):
        t = pd.to_datetime(c.fecha, utc=True, format="ISO8601")
        lim = (t.max() - pd.Timedelta(days=n)) if desde_final else (t.min() + pd.Timedelta(days=n))
        return (t >= lim) if desde_final else (t < lim)
    return f


def _temas(keep):
    def f(c, pub, v):
        return c.publicacion_id.map(pub).isin(keep).values
    return f


def _pubs(frac, seed):
    def f(c, pub, v):
        ps = np.sort(c.publicacion_id.unique())
        rng = np.random.default_rng(seed)
        elegidas = set(rng.choice(ps, int(len(ps) * frac), replace=False))
        return c.publicacion_id.isin(elegidas).values
    return f


RECORTES = {
    "dias10": _dias(10),
    "dias7": _dias(7),
    "ult7dias": _dias(7, desde_final=True),
    "sin_litio": _temas({"hospital", "colectivo"}),
    "sin_hospital": _temas({"litio", "colectivo"}),
    "sin_colectivo": _temas({"litio", "hospital"}),
    "solo_litio": _temas({"litio"}),
    "solo_hospital": _temas({"hospital"}),
    "solo_colectivo": _temas({"colectivo"}),
    "pubs50_s1": _pubs(0.5, 1),
    "pubs50_s2": _pubs(0.5, 2),
    "pubs25_s1": _pubs(0.25, 1),
}


def armar(origen: Path, nombre: str, filtro) -> Path:
    d = BANCO / "recortes" / nombre
    if (d / "verdad" / "narrativas.csv").exists():
        return d
    (d / "verdad").mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(origen / "comentarios.csv.gz", dtype=str, keep_default_na=False)
    vc = pd.read_csv(origen / "verdad" / "comentarios.csv.gz", dtype=str, keep_default_na=False)
    vn = pd.read_csv(origen / "verdad" / "narrativas.csv", dtype=str, keep_default_na=False)
    vcu = pd.read_csv(origen / "verdad" / "cuentas.csv.gz", dtype=str, keep_default_na=False)
    # tema de cada publicación = tema mayoritario de sus comentarios con narrativa
    tema = vc.merge(vn[["narrativa_id", "tema"]], on="narrativa_id").merge(c[["comentario_id", "publicacion_id"]], on="comentario_id")
    pub = tema.groupby("publicacion_id").tema.agg(lambda s: s.mode().iat[0])

    m = filtro(c, pub, vc)
    cc = c[m]
    cc.to_csv(d / "comentarios.csv.gz", index=False, compression="gzip")
    p = pd.read_csv(origen / "publicaciones.csv.gz", dtype=str, keep_default_na=False)
    p[p.publicacion_id.isin(set(cc.publicacion_id))].to_csv(d / "publicaciones.csv.gz", index=False, compression="gzip")
    for f in ("cuentas.csv.gz", "fuentes.csv"):
        shutil.copy(origen / f, d / f)

    v = vc[vc.comentario_id.isin(set(cc.comentario_id))]
    v.to_csv(d / "verdad" / "comentarios.csv.gz", index=False, compression="gzip")
    # una cuenta de campaña sin comentarios en el recorte es indetectable: deja de contar como positiva
    activas = set(cc.cuenta_id)
    vcu = vcu.copy()
    vcu.loc[~vcu.cuenta_id.isin(activas), "campana_id"] = ""
    vcu.to_csv(d / "verdad" / "cuentas.csv.gz", index=False, compression="gzip")
    # narrativas presentes, con origen y conteo recalculados dentro del recorte
    x = v[v.narrativa_id != "ninguna"].merge(cc[["comentario_id", "fecha", "publicacion_id"]], on="comentario_id")
    x["t"] = pd.to_datetime(x.fecha, utc=True, format="ISO8601")
    prim = x.sort_values("t").groupby("narrativa_id").first()
    n = vn[vn.narrativa_id.isin(prim.index)].copy()
    n["fecha_origen"] = n.narrativa_id.map(prim.fecha)
    n["publicacion_origen"] = n.narrativa_id.map(prim.publicacion_id)
    n["comentarios"] = n.narrativa_id.map(x.narrativa_id.value_counts()).astype(int)
    n.to_csv(d / "verdad" / "narrativas.csv", index=False)
    shutil.copy(origen / "verdad" / "campanas.csv", d / "verdad" / "campanas.csv")
    return d


def correr(nombre: str, datos: Path, etiqueta: str) -> dict:
    out = BANCO / etiqueta / nombre
    cache = BANCO / "cache" / nombre
    cache.mkdir(parents=True, exist_ok=True)
    emb = RAIZ / ".murmur" / "embeddings.parquet"   # caché por hash de texto: los recortes son subconjuntos
    if emb.exists() and not (cache / "embeddings.parquet").exists():
        (cache / "embeddings.parquet").symlink_to(emb)
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "murmur", "--datos", str(datos), "--entrega", str(out),
                        "--cache", str(cache), "--sin-ia"], cwd=RAIZ, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{nombre}: murmur falló\n{r.stderr[-2000:]}")
    seg = time.time() - t
    js = out / "puntaje.json"
    subprocess.run([sys.executable, str(SCORER), "--verdad", str(datos / "verdad"), "--entrega", str(out),
                    "--json", str(js)], check=True, capture_output=True)
    s = json.loads(js.read_text())
    co = s["coordinacion"]
    tp = round(co["precision"] * co["cuentas_marcadas"])
    pos = int((pd.read_csv(datos / "verdad" / "cuentas.csv.gz", dtype=str, keep_default_na=False).campana_id != "").sum())
    return {"recorte": nombre, "total": round(s["narrativas"]["puntos"] + co["puntos"], 2),
            "narr": s["narrativas"]["puntos"], "coord": co["puntos"], "marcadas": co["cuentas_marcadas"],
            "tp": tp, "fp": co["cuentas_marcadas"] - tp, "positivas": pos, "f1": round(co["f1"], 3), "seg": round(seg)}


def tabla(filas, cols):
    w = {c: max(len(c), *(len(str(f.get(c, ""))) for f in filas)) for c in cols}
    lin = ["| " + " | ".join(c.ljust(w[c]) for c in cols) + " |", "|" + "|".join("-" * (w[c] + 2) for c in cols) + "|"]
    lin += ["| " + " | ".join(str(f.get(c, "")).ljust(w[c]) for c in cols) + " |" for f in filas]
    return "\n".join(lin)


def comparar(a: str, b: str) -> int:
    A = {f["recorte"]: f for f in json.loads((BANCO / f"{a}.json").read_text())}
    B = {f["recorte"]: f for f in json.loads((BANCO / f"{b}.json").read_text())}
    filas, falla = [], []
    for k in A:
        if k not in B:
            continue
        fa, fb = A[k], B[k]
        filas.append({"recorte": k, f"total {a}": fa["total"], f"total {b}": fb["total"],
                      f"FP {a}": fa["fp"], f"FP {b}": fb["fp"], f"TP {a}": fa["tp"], f"TP {b}": fb["tp"]})
        if fb["fp"] > fa["fp"]:
            falla.append(f"{k}: FP {fa['fp']} -> {fb['fp']}")
        if k == "practica" and fb["total"] < fa["total"]:
            falla.append(f"practica: total {fa['total']} -> {fb['total']}")
    print(tabla(filas, list(filas[0])))
    print(f"\nFP totales: {sum(A[k]['fp'] for k in A if k in B)} -> {sum(B[k]['fp'] for k in B if k in A)}")
    print("VEREDICTO:", "OK" if not falla else "FALLA\n  " + "\n  ".join(falla))
    return 1 if falla else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=RAIZ / "datos" / "practica")
    ap.add_argument("--etiqueta", help="nombre de esta corrida (p.ej. antes, despues)")
    ap.add_argument("--solo", nargs="*", help="correr solo estos recortes")
    ap.add_argument("-j", type=int, default=2, help="recortes en paralelo")
    ap.add_argument("--comparar", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.comparar:
        sys.exit(comparar(*a.comparar))
    if not a.etiqueta:
        ap.error("falta --etiqueta o --comparar")
    if not SCORER.exists():
        sys.exit(f"no encuentro el scorer en {SCORER}")
    nombres = a.solo or ["practica", *RECORTES]
    dirs = {k: (a.datos if k == "practica" else armar(a.datos, k, RECORTES[k])) for k in nombres}
    with ThreadPoolExecutor(a.j) as ex:
        filas = list(ex.map(lambda k: correr(k, dirs[k], a.etiqueta), nombres))
    (BANCO / f"{a.etiqueta}.json").write_text(json.dumps(filas, indent=1))
    print(tabla(filas, ["recorte", "total", "narr", "coord", "marcadas", "tp", "fp", "positivas", "f1", "seg"]))
    print(f"\nFP totales en recortes: {sum(f['fp'] for f in filas if f['recorte'] != 'practica')}")


if __name__ == "__main__":
    main()
