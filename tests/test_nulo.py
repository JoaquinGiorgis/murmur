"""Test nulo de coordinación (no usa la verdad para decidir nada, solo para contar).

Si se destruye la sincronía entre cuentas, el detector no debería marcar a nadie. Es la forma de comprobar
falsos positivos sin etiquetas, siguiendo el modelo nulo por permutación que proponen Pacheco et al. (2021).

  N0  datos reales (control)
  N1  fechas barajadas dentro de cada publicación: conserva volumen y ritmo de cada post, textos y respuestas;
      rompe quién comenta junto a quién
  N2  autores barajados dentro de cada publicación: conserva los horarios; rompe qué cuenta escribió cuándo

Uso:  python tests/test_nulo.py [carpeta_datos] [--semillas 3]
      python tests/test_nulo.py [carpeta_datos] --completo [--semillas 2]
Sale con código 1 si alguna versión barajada marca cuentas.

Sin --completo solo se ejercita coordinacion.detectar (segundos). Con --completo cada variante corre el pipeline
entero sin IA (`python -m murmur --sin-ia`) sobre los datos barajados y se cuentan las cuentas con grupo en
cuentas.csv: así quedan cubiertos también los pasos que suman cuentas después de detectar (satélites, núcleos
alineados, ráfagas, alineados y vocabulario propio), que necesitan embeddings y narrativas. Tarda lo que tarda
el pipeline por variante; usa un único directorio temporal (datos, caché y entrega) que se borra al terminar.
"""
import argparse, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from murmur import datos, coordinacion  # noqa: E402


def _archivo(datos: Path, nombre: str) -> Path:
    gz = datos / f"{nombre}.csv.gz"
    return gz if gz.exists() else datos / f"{nombre}.csv"


def completo(datos: Path, semillas: int) -> int:
    """Pipeline entero sobre N0 (real), N1 (fechas barajadas) y N2 (autores barajados). Devuelve el código de salida."""
    datos = Path(datos).resolve()
    com = pd.read_csv(_archivo(datos, "comentarios"), dtype=str, keep_default_na=False)
    coord = None
    v = datos / "verdad" / "cuentas.csv.gz"
    if v.exists():
        v = pd.read_csv(v, dtype=str).fillna("")
        coord = set(v.loc[v.campana_id != "", "cuenta_id"])
    env = {k: val for k, val in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env.setdefault("HF_HUB_OFFLINE", "1")   # los textos no cambian al barajar: con la caché alcanza

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        d, cache, entrega = tmp / "datos", tmp / "cache", tmp / "entrega"
        d.mkdir(); cache.mkdir()
        for f in datos.iterdir():   # todo menos los comentarios (y la verdad) se enlaza tal cual
            if f.is_file() and not f.name.startswith("comentarios."):
                os.symlink(f, d / f.name)
        emb = RAIZ / ".murmur" / "embeddings.parquet"
        if emb.exists():            # enlace, no copia: embeddings.py escribe a un temporal y reemplaza
            os.symlink(emb, cache / "embeddings.parquet")

        def medir(df, nombre):
            t = time.time()
            df.to_csv(d / "comentarios.csv.gz", index=False, compression="gzip")
            r = subprocess.run([sys.executable, "-m", "murmur", "--datos", str(d), "--entrega", str(entrega),
                                "--cache", str(cache), "--dashboard", str(entrega / "dashboard.html"), "--sin-ia"],
                               cwd=RAIZ, env=env, capture_output=True, text=True)
            if r.returncode not in (0, 2):
                raise RuntimeError(f"{nombre}: murmur falló\n{r.stderr[-2000:]}")
            g = pd.read_csv(entrega / "cuentas.csv", dtype=str, keep_default_na=False)
            m = set(g.loc[g.grupo != "", "cuenta_id"])
            pasos = [l.strip() for l in r.stdout.splitlines()
                     if any(k in l for k in ("operaciones:", "satélites", "núcleos", "ráfagas", "alineados sumados", "vocabulario"))]
            extra = f"  de campaña={len(m & coord):4d}  genuinas={len(m - coord):3d}" if coord is not None else ""
            print(f"{nombre:26s} marcadas={len(m):4d}  operaciones={g.grupo[g.grupo != ''].nunique():2d}{extra}  "
                  f"({time.time() - t:.0f}s)", flush=True)
            for l in pasos:
                print(f"    {l}", flush=True)
            return len(m)

        print(f"{len(com)} comentarios (pipeline completo, sin IA)")
        medir(com, "N0 real (control)")
        falla = 0
        for s in range(semillas):
            rng = np.random.default_rng(s)
            x = com.copy()
            x["fecha"] = x.groupby("publicacion_id").fecha.transform(lambda y: rng.permutation(y.values))
            falla += medir(x, f"N1 fechas barajadas s{s}")
        for s in range(semillas):
            rng = np.random.default_rng(100 + s)
            x = com.copy()
            x["cuenta_id"] = x.groupby("publicacion_id").cuenta_id.transform(lambda y: rng.permutation(y.values))
            falla += medir(x, f"N2 autores barajados s{s}")
    print("VEREDICTO:", "OK (0 cuentas marcadas en datos sin coordinación, pipeline completo)" if falla == 0
          else f"FALLA ({falla} marcas en datos barajados)")
    return 1 if falla else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datos", nargs="?", default=RAIZ / "datos/practica", type=Path)
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--completo", action="store_true", help="correr el pipeline entero sin IA en cada variante")
    a = ap.parse_args()
    if a.completo:
        sys.exit(completo(a.datos, a.semillas))

    with tempfile.TemporaryDirectory() as tmp:
        con = datos.cargar(a.datos, Path(tmp) / "nulo.duckdb", log=lambda s: None)
        c = datos.comentarios_df(con)
        cuentas = con.execute("SELECT cuenta_id FROM cuentas ORDER BY cuenta_id").df().cuenta_id
        con.close()

    coord = None
    v = a.datos / "verdad" / "cuentas.csv.gz"
    if v.exists():
        v = pd.read_csv(v, dtype=str).fillna("")
        coord = set(v.loc[v.campana_id != "", "cuenta_id"])

    def medir(df, nombre):
        t = time.time()
        g, ops = coordinacion.detectar(df, cuentas, log=lambda s: None)
        m = set(g.loc[g.grupo != "", "cuenta_id"])
        extra = f"  de campaña={len(m & coord):4d}  genuinas={len(m - coord):3d}" if coord is not None else ""
        print(f"{nombre:26s} marcadas={len(m):4d}  operaciones={len(ops):2d}{extra}  ({time.time() - t:.1f}s)", flush=True)
        return len(m)

    print(f"{len(c)} comentarios, {len(cuentas)} cuentas")
    medir(c, "N0 real (control)")
    falla = 0
    for s in range(a.semillas):
        rng = np.random.default_rng(s)
        d = c.copy()
        d["ts"] = d.groupby("publicacion_id").ts.transform(lambda x: rng.permutation(x.values))
        falla += medir(d, f"N1 fechas barajadas s{s}")
    for s in range(a.semillas):
        rng = np.random.default_rng(100 + s)
        d = c.copy()
        d["cuenta_id"] = d.groupby("publicacion_id").cuenta_id.transform(lambda x: rng.permutation(x.values))
        falla += medir(d, f"N2 autores barajados s{s}")
    print("VEREDICTO:", "OK (0 cuentas marcadas en datos sin coordinación)" if falla == 0
          else f"FALLA ({falla} marcas en datos barajados)")
    sys.exit(1 if falla else 0)


if __name__ == "__main__":
    main()
