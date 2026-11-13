"""Dashboard estático: un solo HTML con los datos embebidos, sin servidor.

Responde en un minuto las tres preguntas del jurado: qué circula, desde cuándo y
quién lo empuja.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .narrativas import _segmentos

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

    # textos que aparecen en más de una narrativa ("Y la plata dónde está?"): no sirven de ejemplo de ninguna
    con_n = df[(df.n >= 0) & ~df.rep]
    en_varias = con_n.assign(t=con_n.texto.astype(str).str.strip()).groupby("t").narrativa.nunique()
    en_varias = set(en_varias[en_varias > 1].index)

    def ejemplo(textos):
        """El texto más repetido que es propio de esta narrativa, recortado: muestra la postura con palabras reales."""
        t = textos.astype(str).str.strip()
        t = t[t.str.split().str.len() >= 5]
        if not len(t):
            return ""
        propios = t[~t.isin(en_varias)]
        x = (propios if len(propios) else t).value_counts().index[0]
        return x if len(x) <= 160 else x[:157].rstrip() + "…"

    # muletillas: segmentos ("te digo algo", "es así") que aparecen en muchas narrativas, mismo criterio que los títulos
    seg_n = {}
    for nar, tx in con_n.groupby("narrativa").texto:
        for x in tx.astype(str).drop_duplicates():
            for sg in _segmentos(x):
                seg_n.setdefault(sg, set()).add(nar)
    n_narr = max(con_n.narrativa.nunique(), 1)
    muletillas = {sg for sg, ns in seg_n.items() if len(ns) >= max(3, 0.25 * n_narr)}

    def clave(t):
        """Texto sin mayúsculas, emojis, hashtags ni muletillas: dos variantes de la misma consigna dan la misma clave."""
        return " | ".join(sg for sg in _segmentos(" ".join(w for w in str(t).split() if not w.startswith("#"))) if sg not in muletillas)

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
        # lo que repite: sus comentarios (no respuestas) en la narrativa que empuja, contados sin muletillas ni emojis
        tf = rr[rr.narrativa == foco_id].texto.astype(str).str.strip() if foco_id else pd.Series([], dtype=str)
        tf = tf[tf.str.split().str.len() >= 4]
        kk = tf.map(clave)
        tf, kk = tf[kk != ""], kk[kk != ""]
        cuenta_k = kk.value_counts()
        textos_op = []
        for k_, v in cuenta_k.head(3).items():
            vc = tf[kk == k_].value_counts()            # la variante más escrita de esa consigna; si empatan, la más corta
            t = sorted(vc.index[vc.values == vc.values[0]], key=len)[0]
            textos_op.append({"texto": t if len(t) <= 140 else t[:137].rstrip() + "…", "veces": int(v)})
        rotulo = "Lo que más repiten" if len(cuenta_k) and cuenta_k.iloc[0] > 2 else "Ejemplos de lo que publican"
        lista_ops.append({
            "grupo": o.grupo, "cuentas": int(o.cuentas), "comentarios": int(len(sub)),
            "foco": foco_id, "foco_titulo": titulo,
            "foco_pct": round(float(foco.iloc[0] / foco.sum()), 2) if len(foco) else 0,
            "horas": horas,
            "edad_dias": None if pd.isna(edad) else int(edad),
            "seguidores": None if info.seguidores.isna().all() else int(info.seguidores.median()),
            "respuestas_internas": entre,
            "textos": textos_op, "textos_rotulo": rotulo,
        })

    ruido_pct = round(float((df.n < 0).mean() * 100), 1) if len(df) else 0.0   # comentarios sin narrativa ("ninguna")
    data = {"meta": {**meta, "t0": t0, "t1": t1, "bin": BIN, "nb": nb, "ruido_pct": ruido_pct, "narrativas_total": int(len(nd))},
            "narrativas": narrs, "ops": lista_ops}
    plantilla = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
    js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # un texto con </script> no rompe la página
    salida.write_text(plantilla.replace("/*__DATA__*/null", js), encoding="utf-8")
