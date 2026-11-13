"""Test nulo de coordinación (no usa la verdad para decidir nada, solo para contar).

Si se destruye la sincronía entre cuentas, el detector no debería marcar a nadie. Es la forma de comprobar
falsos positivos sin etiquetas, siguiendo el modelo nulo por permutación que proponen Pacheco et al. (2021).

  N0  datos reales (control)
  N1  fechas barajadas dentro de cada publicación: conserva volumen y ritmo de cada post, textos y respuestas;
      rompe quién comenta junto a quién
  N2  autores barajados dentro de cada publicación: conserva los horarios; rompe qué cuenta escribió cuándo

Uso:  python tests/test_nulo.py [carpeta_datos] [--semillas 3]
Sale con código 1 si alguna versión barajada marca cuentas.
"""
import argparse, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from murmur import datos, coordinacion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datos", nargs="?", default=RAIZ / "datos/practica", type=Path)
    ap.add_argument("--semillas", type=int, default=3)
    a = ap.parse_args()

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
