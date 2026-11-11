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
        if grado >= GRADO_MIN:
            nombre = f"op{len(ops) + 1:02d}"
            ops.append({"grupo": nombre, "cuentas": int(len(ii)), "grado_medio": round(grado, 2)})
            for i in ii:
                grupo[acc[i]] = nombre
    log(f"  operaciones: {len(ops)}  cuentas marcadas: {len(grupo)}")
    out = pd.DataFrame({"cuenta_id": todas_cuentas.values})
    out["grupo"] = out.cuenta_id.map(grupo).fillna("")
    return out, pd.DataFrame(ops)
