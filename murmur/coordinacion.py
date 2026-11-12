"""Tarea B: detección de operaciones coordinadas.

Idea: una ráfaga no alcanza; lo que delata a una operación es la repetición.
1. Co-acción: dos cuentas que comentan la misma publicación dentro de 10 minutos
   (dos grillas desfasadas 5 min para no cortar ráfagas en el borde).
2. Por par de cuentas: k = cuántas publicaciones distintas comparten así, y
   coseno = k / sqrt(pubs_a * pubs_b), que descuenta a las cuentas muy activas
   (que coinciden con todos por puro volumen) y a los virales genuinos (mucha
   gente distinta, una sola vez).
3. Se suman las respuestas entre cuentas (operaciones que se responden entre sí).
4. Leiden sobre el grafo de pares sospechosos; se marcan las comunidades chicas y
   densas (grado interno medio alto). Una operación es un grupo cerrado que
   coincide muchas veces; la opinión genuina forma árboles ralos.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
import igraph as ig
import leidenalg as la

VENTANA = 600          # segundos
K_MIN, COS_MIN = 3, 0.15
K_MIN_REP, COS_MIN_REP = 2, 0.08
TAM_MIN, GRADO_MIN = 8, 3.0
PESO_RESP, PODA_FRAC = 3.0, 0.5
TAM_MAX_FRAC = 0.02   # una "operación" que abarca >2% de las cuentas es la masa de usuarios activos, no una operación
DENS_MIN = 0.09       # densidad del grupo podado (grado medio / (n-1)): operaciones 0,12-0,32; grupos genuinos <= 0,063


def detectar(c: pd.DataFrame, todas_cuentas: pd.Series, log=print):
    a, acc = pd.factorize(c.cuenta_id)
    N = len(acc)
    ts = c.ts.values
    npub = pd.Series(c.publicacion_id.values).groupby(a).nunique().reindex(range(N)).fillna(1).values

    # co-acción por publicación en ventana corta
    K = None
    for off in (0, VENTANA // 2):
        item = pd.factorize(c.publicacion_id.astype(str) + "_" + ((ts + off) // VENTANA).astype(str))[0]
        X = sp.csr_matrix((np.ones(len(c), np.float32), (a, item)), shape=(N, item.max() + 1))
        X.data[:] = 1
        X.sum_duplicates()
        X.data[:] = 1
        Kc = sp.triu(X @ X.T, 1).tocsr()
        K = Kc if K is None else K.maximum(Kc)

    # respuestas entre cuentas
    idx = pd.Series(np.arange(len(c)), index=c.comentario_id)
    pi = c.respuesta_a.map(idx)
    m = pi.notna().values
    ra = a[m]; rb = a[pi[m].astype(int).values]
    k = ra != rb
    lo, hi = np.minimum(ra[k], rb[k]), np.maximum(ra[k], rb[k])
    R = sp.coo_matrix((np.ones(len(lo)), (lo, hi)), shape=(N, N)).tocsr()

    M = (K + R).tocoo()
    df = pd.DataFrame({"x": M.row, "y": M.col})
    df["k"] = np.asarray(K[df.x.values, df.y.values]).ravel()
    df["r"] = np.asarray(R[df.x.values, df.y.values]).ravel()
    df["cos"] = df.k / np.sqrt(npub[df.x] * npub[df.y])
    E = df[((df.k >= K_MIN) & (df.cos >= COS_MIN)) | ((df.k >= K_MIN_REP) & (df.r >= 1) & (df.cos >= COS_MIN_REP))]
    log(f"  pares sospechosos: {len(E)}")

    g = ig.Graph(n=N, edges=list(zip(E.x, E.y)), edge_attrs={"weight": E.cos.values})
    part = la.find_partition(g, la.ModularityVertexPartition, weights="weight", seed=1)
    memb = np.array(part.membership)

    # fuerza de lazo entre pares: co-acciones + respuestas (pesan más: responderse es interacción directa)
    W = (K + K.T + PESO_RESP * (R + R.T)).tocsr()

    grupo = {}
    ops = []
    for cid in np.unique(memb):
        ii = np.nonzero(memb == cid)[0]
        if len(ii) < TAM_MIN or len(ii) > TAM_MAX_FRAC * N:
            continue
        # poda del núcleo: una operación es un grupo donde todos están muy ligados entre sí;
        # las cuentas genuinas que Leiden arrastró quedan en la periferia con lazos internos débiles
        while len(ii) >= TAM_MIN:
            fuerza = np.asarray(W[ii][:, ii].sum(1)).ravel()
            keep = fuerza >= PODA_FRAC * np.median(fuerza)
            if keep.all():
                break
            ii = ii[keep]
        if len(ii) < TAM_MIN:
            continue
        sub = g.subgraph(ii)
        grado = 2 * sub.ecount() / len(ii)
        densidad = grado / (len(ii) - 1)
        if grado >= GRADO_MIN and densidad >= DENS_MIN:
            nombre = f"op{len(ops) + 1:02d}"
            ops.append({"grupo": nombre, "cuentas": int(len(ii)), "grado_medio": round(grado, 2), "densidad": round(densidad, 3)})
            for i in ii:
                grupo[acc[i]] = nombre
    log(f"  operaciones: {len(ops)}  cuentas marcadas: {len(grupo)}")
    out = pd.DataFrame({"cuenta_id": todas_cuentas.values})
    out["grupo"] = out.cuenta_id.map(grupo).fillna("")
    return out, pd.DataFrame(ops)


# --- satélites: cuentas de baja actividad de una operación ya detectada
SAT_SIM, SAT_PUBS, SAT_FRAC = 0.8, 2, 0.5


def sumar_satelites(c: pd.DataFrame, grupos: pd.DataFrame, ops: pd.DataFrame, E, kid, log=print):
    """Suma a cada operación ya detectada las cuentas con 2-5 comentarios que la co-acción no alcanza a ver.

    Una cuenta entra solo si se dan las tres cosas a la vez (cada una por separado no separa campaña de
    genuinas): (1) en al menos SAT_PUBS publicaciones distintas comenta a menos de VENTANA de un comentario
    del núcleo, (2) con un texto casi igual al de ese comentario (coseno >= SAT_SIM), y (3) al menos
    SAT_FRAC de las publicaciones en las que participa coinciden con el núcleo.
    No crea operaciones nuevas ni toca el núcleo: si no hay núcleo, no hace nada.
    """
    g = dict(zip(grupos.cuenta_id, grupos.grupo))
    grp = c.cuenta_id.map(g).fillna("").values
    if not (grp != "").any():
        return grupos, ops, 0
    nw = c.texto.fillna("").str.split().str.len().values
    ts = c.ts.values
    cuenta = c.cuenta_id.values
    pub = c.publicacion_id.values
    npub = c.groupby("cuenta_id").publicacion_id.nunique()
    por_pub = pd.Series(np.arange(len(c))).groupby(pub).apply(lambda s: s.values)
    mejor = {}   # cuenta -> (publicaciones con texto casi igual, operación)
    for op in sorted(set(grp) - {""}):
        cerca, igual = defaultdict(set), defaultdict(set)
        for p in pd.unique(pub[grp == op]):
            ii = por_pub[p]
            nuc = ii[grp[ii] == op]
            cand = ii[grp[ii] == ""]
            if not len(cand):
                continue
            m = np.abs(ts[cand][:, None] - ts[nuc][None, :]) <= VENTANA
            fila = m.any(1)
            if not fila.any():
                continue
            S = np.where(m[fila], E[kid[cand[fila]]] @ E[kid[nuc]].T, -1.0)
            parecido = (S.max(1) >= SAT_SIM) & (nw[cand[fila]] >= 4)
            for cu, ok in zip(cuenta[cand[fila]], parecido):
                cerca[cu].add(p)
                if ok:
                    igual[cu].add(p)
        for cu, ps in igual.items():
            if len(ps) >= SAT_PUBS and len(cerca[cu]) / npub[cu] >= SAT_FRAC and (cu not in mejor or len(ps) > mejor[cu][0]):
                mejor[cu] = (len(ps), op)
    if mejor:
        grupos = grupos.copy()
        grupos["grupo"] = [mejor[cu][1] if cu in mejor else gr for cu, gr in zip(grupos.cuenta_id, grupos.grupo)]
        if len(ops):
            ops = ops.copy()
            ops["cuentas"] = ops.grupo.map(grupos[grupos.grupo != ""].groupby("grupo").size()).astype(int)
    log(f"  satélites sumados a operaciones: {len(mejor)}")
    return grupos, ops, len(mejor)
