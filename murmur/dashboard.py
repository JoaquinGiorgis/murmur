"""Dashboard estático: un solo HTML con los datos embebidos, sin servidor.

Responde en un minuto las tres preguntas del jurado: qué circula, desde cuándo y
quién lo empuja.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

BIN = 6 * 3600  # 6 horas
TZ = -3 * 3600


def generar(df, nd, grupos, ops, cuentas_info, meta, salida: Path):
    t0 = int((df.ts.min() + TZ) // 86400 * 86400 - TZ)
    t1 = int(df.ts.max())
    nb = int((t1 - t0) // BIN + 1)
    df = df.assign(b=((df.ts - t0) // BIN).astype(int))
    g = dict(zip(grupos.cuenta_id, grupos.grupo))
    df["grupo"] = df.cuenta_id.map(g).fillna("")
    total_n = (df.n >= 0).sum()

    # quién empuja qué: comentarios propios (no respuestas) de cada operación por narrativa
    raiz_op = df[(df.grupo != "") & ~df.rep & (df.n >= 0)]
    op_n = raiz_op.groupby(["grupo", "narrativa"]).size()
    op_tot = raiz_op.groupby("grupo").size()

    def ejemplo(textos):
        """El texto más repetido (normalizado por espacios), recortado: muestra la postura con palabras reales."""
        t = textos.astype(str).str.strip()
        t = t[t.str.split().str.len() >= 5]
        if not len(t):
            return ""
        x = t.value_counts().index[0]
        return x if len(x) <= 160 else x[:157].rstrip() + "…"

    narrs = []
    for _, r in nd.sort_values("comentarios", ascending=False).iterrows():
        sub = df[df.narrativa == r.narrativa]
        if len(sub) < 0.005 * total_n:
            continue
        a = np.bincount(sub[sub.postura == "apoya"].b, minlength=nb)[:nb]
        x = np.bincount(sub[sub.postura == "rechaza"].b, minlength=nb)[:nb]
        empuja = []
        for grp in op_tot.index:
            k = int(op_n.get((grp, r.narrativa), 0))
            if k >= 20 and k / op_tot[grp] >= 0.25:
                s = raiz_op[(raiz_op.grupo == grp) & (raiz_op.narrativa == r.narrativa)]
                empuja.append({"grupo": grp, "comentarios": k, "postura": "apoya" if (s.postura == "apoya").mean() >= 0.5 else "rechaza"})
        narrs.append({
            "id": r.narrativa, "titulo": r.titulo, "claves": list(r.frases_clave)[:5],
            "origen": int(r.origen), "emergente": bool(r.origen > t0 + 86400 + 12 * 3600),
            "total": int(len(sub)), "apoya": int(a.sum()), "rechaza": int(x.sum()),
            "a": a.tolist(), "r": x.tolist(), "empuja": empuja,
            "ej_apoya": ejemplo(sub[(sub.postura == "apoya") & ~sub.rep].texto),
            "ej_rechaza": ejemplo(sub[(sub.postura == "rechaza") & ~sub.rep].texto),
        })

    ci = cuentas_info.set_index("cuenta_id")
    autor = dict(zip(df.comentario_id, df.cuenta_id))
    lista_ops = []
    for _, o in ops.iterrows():
        mem = grupos.cuenta_id[grupos.grupo == o.grupo]
        sub = df[df.cuenta_id.isin(mem)]
        horas = np.bincount(((sub.ts + TZ) % 86400 // 3600).astype(int), minlength=24).tolist()
        info = ci.reindex(mem)
        fc = pd.to_datetime(info.fecha_creacion, errors="coerce")
        edad = (pd.Timestamp(t0, unit="s") - fc).dt.days.median()
        rr = raiz_op[raiz_op.grupo == o.grupo]
        foco = rr.narrativa.value_counts()
        foco_id = foco.index[0] if len(foco) else None
        titulo = nd.set_index("narrativa").titulo.get(foco_id, "") if foco_id else ""
        # respuestas entre miembros de la misma operación
        rep = sub[sub.rep & sub.respuesta_a.notna()]
        autor_padre = rep.respuesta_a.map(autor)
        entre = int((autor_padre.isin(set(mem)) & (autor_padre != rep.cuenta_id)).sum())
        lista_ops.append({
            "grupo": o.grupo, "cuentas": int(o.cuentas), "comentarios": int(len(sub)),
            "foco": foco_id, "foco_titulo": titulo,
            "foco_pct": round(float(foco.iloc[0] / foco.sum()), 2) if len(foco) else 0,
            "horas": horas,
            "edad_dias": None if pd.isna(edad) else int(edad),
            "seguidores": None if info.seguidores.isna().all() else int(info.seguidores.median()),
            "respuestas_internas": entre,
            "textos": [{"texto": t if len(t) <= 140 else t[:137].rstrip() + "…", "veces": int(v)}
                       for t, v in (lambda t: t[t.str.split().str.len() >= 4])(sub[~sub.rep].texto.astype(str).str.strip())
                       .value_counts().head(3).items()],
        })

    data = {"meta": {**meta, "t0": t0, "t1": t1, "bin": BIN, "nb": nb}, "narrativas": narrs, "ops": lista_ops}
    plantilla = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
    js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # un texto con </script> no rompe la página
    salida.write_text(plantilla.replace("/*__DATA__*/null", js), encoding="utf-8")
