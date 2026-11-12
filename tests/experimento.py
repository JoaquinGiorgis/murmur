"""Ciclo de experimentos: probar una idea cambiando constantes al vuelo, sin tocar el código.

Corre murmur sin IA sobre un conjunto de recortes (en paralelo), lo puntúa con el scorer oficial,
anota todo en un registro y da un veredicto contra la línea base con las reglas del banco:
práctica no baja, ningún recorte suma falsos positivos y la suma total sube.

Uso:
    python tests/experimento.py --nombre base                       # línea base (sin cambios)
    python tests/experimento.py --nombre sat07 coordinacion.SAT_SIM=0.7
    python tests/experimento.py --nombre x --set completo narrativas.HDB_MCS=50
    python tests/experimento.py --registro                          # tabla de todo lo probado

Conjuntos:
    rapido   práctica, dias10, dias7 y resp20 (20% de las respuestas)   ~8-10 min con -j 3
    completo práctica y los 12 recortes del banco, más resp20           ~30 min
    reserva  rutas en MURMUR_RESERVA (separadas por ':'): datos que NO se usan para elegir,
             solo para comprobar al final que lo elegido generaliza.

Los cambios son módulo.CONSTANTE=valor (módulos: narrativas, coordinacion, embeddings, datos).
Todo queda en .murmur/exp/ (fuera de git). El código del repo no se modifica.
"""
import argparse, ast, csv, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from banco_estres import RAIZ, SCORER, RECORTES, armar  # noqa: E402

EXP = RAIZ / ".murmur" / "exp"
REGISTRO = EXP / "registro.csv"
PRACTICA = RAIZ / "datos" / "practica"
CONJUNTOS = {
    "rapido": ["practica", "dias10", "dias7", "resp20"],
    "completo": ["practica", *RECORTES, "resp20"],
}
COLS = ["fecha", "nombre", "cambios", "commit", "dataset", "total", "narr", "coord", "bcubed", "ruido",
        "postura", "origen", "f1_coord", "marcadas", "tp", "fp", "seg"]


def _resp20() -> Path:
    """Práctica con solo el 20% de los enlaces de respuesta: simula un set con poca conversación."""
    d = EXP / "datos" / "resp20"
    if (d / "verdad" / "narrativas.csv").exists():
        return d
    (d / "verdad").mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(PRACTICA / "comentarios.csv.gz", dtype=str, keep_default_na=False)
    rng = np.random.default_rng(20)
    tiene = c.respuesta_a != ""
    borrar = tiene & (rng.random(len(c)) >= 0.2)
    c.loc[borrar, "respuesta_a"] = ""
    c.to_csv(d / "comentarios.csv.gz", index=False, compression="gzip")
    for f in ("publicaciones.csv.gz", "cuentas.csv.gz", "fuentes.csv"):
        shutil.copy(PRACTICA / f, d / f)
    for f in PRACTICA.joinpath("verdad").iterdir():
        shutil.copy(f, d / "verdad" / f.name)
    return d


def _datos(nombre: str) -> Path:
    if nombre == "practica":
        return PRACTICA
    if nombre == "resp20":
        return _resp20()
    if nombre in RECORTES:
        return armar(PRACTICA, nombre, RECORTES[nombre])
    return Path(nombre)  # ruta de reserva


def _correr(nombre_exp: str, cambios: list, ds: str) -> dict:
    datos = _datos(ds)
    etiqueta = Path(ds).name if "/" in ds else ds
    out = EXP / nombre_exp / etiqueta
    cache = EXP / "cache" / f"{nombre_exp}__{etiqueta}"   # una caché por corrida: DuckDB no admite dos procesos
    cache.mkdir(parents=True, exist_ok=True)
    emb = RAIZ / ".murmur" / "embeddings.parquet"
    if emb.exists() and not (cache / "embeddings.parquet").exists():
        shutil.copy(emb, cache / "embeddings.parquet")   # copia, no symlink: la caché se reescribe si hay textos nuevos
    parches = "".join(f"import murmur.{m} as _m; _m.{k} = {v!r}\n" for m, k, v in cambios)
    codigo = (f"import sys, runpy\n{parches}"
              f"sys.argv = ['murmur', '--datos', {str(datos)!r}, '--entrega', {str(out)!r}, '--cache', {str(cache)!r}, "
              f"'--dashboard', {str(out / 'dashboard.html')!r}, '--sin-ia']\n"
              f"runpy.run_module('murmur', run_name='__main__')\n")
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["HF_HUB_OFFLINE"] = "1"
    t = time.time()
    r = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, env=env, capture_output=True, text=True)
    seg = round(time.time() - t)
    if r.returncode not in (0, 2):
        raise RuntimeError(f"{etiqueta}: murmur falló\n{r.stderr[-1500:]}")
    js = out / "puntaje.json"
    subprocess.run([sys.executable, str(SCORER), "--verdad", str(datos / "verdad"), "--entrega", str(out),
                    "--json", str(js)], check=True, capture_output=True)
    s = json.loads(js.read_text())
    n, co = s["narrativas"], s["coordinacion"]
    tp = round(co["precision"] * co["cuentas_marcadas"])
    return {"dataset": etiqueta, "total": round(n["puntos"] + co["puntos"], 2), "narr": n["puntos"], "coord": co["puntos"],
            "bcubed": round(n["bcubed_f1"], 3), "ruido": round(n["ruido_f1"], 3), "postura": round(n["postura_macro_f1"], 3),
            "origen": round(n["origen"], 3), "f1_coord": round(co["f1"], 3), "marcadas": co["cuentas_marcadas"],
            "tp": tp, "fp": co["cuentas_marcadas"] - tp, "seg": seg}


def _leer_registro() -> pd.DataFrame:
    return pd.read_csv(REGISTRO, dtype={"commit": str}) if REGISTRO.exists() else pd.DataFrame(columns=COLS)


def _commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True, text=True)
    sucio = subprocess.run(["git", "status", "--porcelain", "murmur/"], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + ("+cambios" if sucio else "")


def veredicto(filas: list, base: pd.DataFrame) -> int:
    b = {r.dataset: r for r in base.itertuples()}
    print(f"\n{'dataset':12s} {'base':>7s} {'nuevo':>7s} {'Δ':>7s}  {'FP base':>7s} {'FP nuevo':>8s}")
    falla, suma = [], 0.0
    for f in filas:
        r = b.get(f["dataset"])
        if r is None:
            print(f"{f['dataset']:12s} {'—':>7s} {f['total']:7.2f}")
            continue
        d = f["total"] - r.total
        suma += d
        print(f"{f['dataset']:12s} {r.total:7.2f} {f['total']:7.2f} {d:+7.2f}  {int(r.fp):7d} {f['fp']:8d}")
        if f["fp"] > r.fp:
            falla.append(f"{f['dataset']}: falsos positivos {int(r.fp)} -> {f['fp']}")
        if f["dataset"] == "practica" and f["total"] < r.total:
            falla.append(f"practica: {r.total} -> {f['total']}")
    print(f"\nsuma de diferencias: {suma:+.2f}")
    if falla:
        print("VEREDICTO: RECHAZADO\n  " + "\n  ".join(falla))
        return 1
    if suma <= 0:
        print("VEREDICTO: SIN MEJORA (no rompe nada, pero tampoco suma)")
        return 1
    print("VEREDICTO: ACEPTABLE -> confirmar con --set completo antes de pasarlo al código")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", help="nombre del experimento ('base' para la línea base)")
    ap.add_argument("--set", default="rapido", choices=["rapido", "completo", "reserva"])
    ap.add_argument("-j", type=int, default=3)
    ap.add_argument("--registro", action="store_true", help="mostrar el registro y salir")
    ap.add_argument("cambios", nargs="*", help="modulo.CONSTANTE=valor")
    a = ap.parse_args()
    EXP.mkdir(parents=True, exist_ok=True)
    if a.registro:
        r = _leer_registro()
        print(r.pivot_table(index="nombre", columns="dataset", values="total", aggfunc="last").round(2).to_string() if len(r) else "registro vacío")
        return
    if not a.nombre:
        ap.error("falta --nombre")
    cambios = []
    for c in a.cambios:
        izq, val = c.split("=", 1)
        mod, const = izq.split(".")
        cambios.append((mod, const, ast.literal_eval(val)))
    if a.set == "reserva":
        dss = [p for p in os.environ.get("MURMUR_RESERVA", "").split(":") if p]
        if not dss:
            sys.exit("definí MURMUR_RESERVA con las rutas de reserva (separadas por ':')")
    else:
        dss = CONJUNTOS[a.set]
    for ds in dss:  # armar los recortes antes de paralelizar
        _datos(ds)
    with ThreadPoolExecutor(a.j) as ex:
        filas = list(ex.map(lambda ds: _correr(a.nombre, cambios, ds), dss))
    meta = {"fecha": datetime.now().isoformat(timespec="seconds"), "nombre": a.nombre,
            "cambios": " ".join(a.cambios), "commit": _commit()}
    reg = _leer_registro()
    nuevo = pd.DataFrame([{**meta, **f} for f in filas])[COLS]
    pd.concat([reg, nuevo], ignore_index=True).to_csv(REGISTRO, index=False)
    print(nuevo[["dataset", "total", "narr", "coord", "postura", "origen", "f1_coord", "fp", "seg"]].to_string(index=False))
    if a.nombre == "base":
        print("\nLínea base registrada.")
        return
    base = reg[(reg.nombre == "base") & reg.dataset.isin([f["dataset"] for f in filas])]
    base = base.groupby("dataset").tail(1)
    if base.empty:
        print("\nNo hay línea base para estos datasets: corré antes --nombre base con el mismo --set.")
        return
    sys.exit(veredicto(filas, base))


if __name__ == "__main__":
    main()
