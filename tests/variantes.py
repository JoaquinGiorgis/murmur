"""Variantes de forma y contenido del set de práctica (tabla de "Robustez y generalización" del README).
Corre murmur --sin-ia sobre cada una en un único directorio temporal (embeddings enlazados) y puntúa con el scorer oficial.
uso: variantes.py [nombres...]"""
import gzip, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; P = R/"datos"/"practica"; SCORER = R.parent/"desafio-qualydata"/"evaluacion"/"puntuar.py"
PY = sys.executable
def leer(f): return pd.read_csv(f, dtype=str, keep_default_na=False)
C = leer(P/"comentarios.csv.gz"); CU = leer(P/"cuentas.csv.gz"); PU = leer(P/"publicaciones.csv.gz")
VC = leer(P/"verdad/comentarios.csv.gz"); VCU = leer(P/"verdad/cuentas.csv.gz"); VN = leer(P/"verdad/narrativas.csv")
def base(): return dict(c=C.copy(), cu=CU.copy(), pu=PU.copy(), vc=VC.copy(), vcu=VCU.copy(), vn=VN.copy())
def v_utc150():
    d = base()
    def mover(s): return (pd.to_datetime(s, utc=True, format="ISO8601") + pd.Timedelta(days=150)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    d["c"]["fecha"] = mover(d["c"].fecha); d["vn"]["fecha_origen"] = mover(d["vn"].fecha_origen)
    for col in d["pu"].columns:
        if "fecha" in col: d["pu"][col] = mover(d["pu"][col])
    return d
def v_ids():
    d = base()
    h = lambda pref: (lambda x: pref + hashlib.blake2b(x.encode(), digest_size=6).hexdigest() if x else x)
    mc, ma, mp = h("k"), h("z"), h("w")
    c = d["c"]; c["comentario_id"] = c.comentario_id.map(mc); c["respuesta_a"] = c.respuesta_a.map(mc); c["cuenta_id"] = c.cuenta_id.map(ma); c["publicacion_id"] = c.publicacion_id.map(mp)
    d["c"] = c.sample(frac=1, random_state=7).reset_index(drop=True)
    d["cu"]["cuenta_id"] = d["cu"].cuenta_id.map(ma); d["cu"] = d["cu"].sample(frac=1, random_state=8)
    d["pu"]["publicacion_id"] = d["pu"].publicacion_id.map(mp); d["pu"] = d["pu"].sample(frac=1, random_state=9)
    d["vc"]["comentario_id"] = d["vc"].comentario_id.map(mc); d["vcu"]["cuenta_id"] = d["vcu"].cuenta_id.map(ma)
    if "publicacion_origen" in d["vn"]: d["vn"]["publicacion_origen"] = d["vn"].publicacion_origen.map(mp)
    return d
def v_sucio():
    d = base(); c = d["c"]; rng = np.random.default_rng(11); n = len(c)
    c.loc[rng.random(n) < 0.005, "texto"] = ""                                   # 0,5% textos vacíos
    rep = (c.respuesta_a != "").values
    rota = rep & (rng.random(n) < 0.005); c.loc[rota, "respuesta_a"] = ["c_no_existe_%d" % i for i in range(rota.sum())]   # 0,5% de respuestas a ids inexistentes
    mala = rng.random(n) < 0.003; c.loc[mala, "fecha"] = rng.choice(["ayer", "", "32/13/2026", "sin fecha"], mala.sum())   # 0,3% fechas ilegibles
    c = pd.concat([c, c.sample(frac=0.01, random_state=12)], ignore_index=True)   # 1% filas duplicadas
    c["extra"] = "x"; c["idioma"] = "es"                                          # columnas extra
    d["c"] = c; d["cu"] = d["cu"].sample(frac=0.99, random_state=13)              # 1% de cuentas faltantes
    d["crlf"] = True
    return d
def v_resp(frac):
    def f():
        d = base(); c = d["c"]; rng = np.random.default_rng(int(frac*100))
        c.loc[(c.respuesta_a != "") & (rng.random(len(c)) >= frac), "respuesta_a"] = ""
        return d
    return f
def v_150k28():
    d = base(); c = d["c"]
    tema = d["vc"].merge(d["vn"][["narrativa_id","tema"]], on="narrativa_id").merge(c[["comentario_id","publicacion_id"]], on="comentario_id")
    pub = tema.groupby("publicacion_id").tema.agg(lambda s: s.mode().iat[0])
    rng = np.random.default_rng(28); lit = np.sort(pub.index[pub == "litio"]); n_por = c.publicacion_id.value_counts()
    orden = rng.permutation(lit); sobra = len(c) - 150_000; quitar = set(); acum = 0
    for p in orden:
        if acum >= sobra: break
        quitar.add(p); acum += n_por[p]
    c = c[~c.publicacion_id.isin(quitar)].copy()
    t = pd.to_datetime(c.fecha, utc=True, format="ISO8601"); t0 = t.min()
    c["fecha"] = (t0 + (t - t0) * 2).dt.tz_convert("America/Argentina/Buenos_Aires").map(lambda x: x.isoformat(timespec="seconds"))
    d["c"] = c; d["pu"] = d["pu"][d["pu"].publicacion_id.isin(set(c.publicacion_id))]
    v = d["vc"][d["vc"].comentario_id.isin(set(c.comentario_id))]; d["vc"] = v
    x = v[v.narrativa_id != "ninguna"].merge(c[["comentario_id","fecha","publicacion_id"]], on="comentario_id"); x["t"] = pd.to_datetime(x.fecha, utc=True, format="ISO8601")
    prim = x.sort_values("t").groupby("narrativa_id").first()
    vn = d["vn"][d["vn"].narrativa_id.isin(prim.index)].copy(); vn["fecha_origen"] = vn.narrativa_id.map(prim.fecha); vn["comentarios"] = vn.narrativa_id.map(x.narrativa_id.value_counts()).astype(int).astype(str); d["vn"] = vn
    d["vcu"].loc[~d["vcu"].cuenta_id.isin(set(c.cuenta_id)), "campana_id"] = ""
    return d
def v_sinresp():
    d = base(); d["c"]["respuesta_a"] = ""; return d
def v_ddmm():
    d = base(); d["c"]["fecha"] = pd.to_datetime(d["c"].fecha, utc=True, format="ISO8601").dt.strftime("%d/%m/%Y %H:%M"); return d
V = {"utc150": v_utc150, "ids": v_ids, "sucio": v_sucio, "resp50": v_resp(0.5), "resp35": v_resp(0.35), "resp20": v_resp(0.2), "150k28": v_150k28, "sinresp": v_sinresp, "ddmm": v_ddmm}
def escribir(df, path, crlf=False):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f: df.to_csv(f, index=False, lineterminator="\r\n" if crlf else "\n")
env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}; env["HF_HUB_OFFLINE"] = "1"
res = []
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp); D, cache, out = tmp/"datos", tmp/"cache", tmp/"entrega"; (D/"verdad").mkdir(parents=True); cache.mkdir()
    os.symlink(R/".murmur"/"embeddings.parquet", cache/"embeddings.parquet"); shutil.copy(P/"fuentes.csv", D/"fuentes.csv")
    for nombre in (sys.argv[1:] or list(V)):
        d = V[nombre](); crlf = d.get("crlf", False)
        escribir(d["c"], D/"comentarios.csv.gz", crlf); escribir(d["cu"], D/"cuentas.csv.gz", crlf); escribir(d["pu"], D/"publicaciones.csv.gz", crlf)
        escribir(d["vc"], D/"verdad/comentarios.csv.gz"); escribir(d["vcu"], D/"verdad/cuentas.csv.gz"); d["vn"].to_csv(D/"verdad/narrativas.csv", index=False)
        if out.exists(): shutil.rmtree(out)
        t = time.time()
        r = subprocess.run([PY, "-m", "murmur", "--datos", str(D), "--entrega", str(out), "--cache", str(cache), "--dashboard", str(out/"dashboard.html"), "--sin-ia"], cwd=R, env=env, capture_output=True, text=True)
        fila = {"variante": nombre, "comentarios": len(d["c"]), "codigo": r.returncode, "seg": round(time.time()-t)}
        avisos = [l.strip() for l in (r.stdout + r.stderr).splitlines() if any(k in l for k in ("AVISO", "ERROR", "ATENCIÓN", "ISO", "fechas"))]
        fila["avisos"] = avisos[:4]
        if r.returncode in (0, 2) and (out/"cuentas.csv").exists() and (out/"comentarios.csv").exists():
            js = tmp/"p.json"
            s = subprocess.run([PY, str(SCORER), "--verdad", str(D/"verdad"), "--entrega", str(out), "--json", str(js)], capture_output=True, text=True)
            if s.returncode == 0:
                j = json.loads(js.read_text()); n, co = j["narrativas"], j["coordinacion"]; tp = round(co["precision"]*co["cuentas_marcadas"])
                pc = pd.read_csv(out/"comentarios.csv", dtype=str, keep_default_na=False); tam = pc[pc.narrativa != "ninguna"].narrativa.value_counts()
                fila.update(total=round(n["puntos"]+co["puntos"], 2), narr=n["puntos"], coord=co["puntos"], bc=round(n["bcubed_f1"],3), ruido=round(n["ruido_f1"],3), post=round(n["postura_macro_f1"],3), orig=round(n["origen"],3),
                            marcadas=co["cuentas_marcadas"], tp=tp, fp=co["cuentas_marcadas"]-tp, positivas=int((d["vcu"].campana_id != "").sum()), f1=round(co["f1"],3),
                            narr_pred=int(len(tam)), narr_pred_100=int((tam >= 100).sum()), narr_real=int(d["vn"].narrativa_id.nunique()))
            else: fila["scorer"] = s.stderr[-300:] + s.stdout[-300:]
        else: fila["stderr"] = r.stderr[-400:]
        res.append(fila); print(json.dumps(fila, ensure_ascii=False), flush=True)
json.dump(res, open(Path(__file__).parent/"variantes_res.json", "w"), ensure_ascii=False, indent=1)
