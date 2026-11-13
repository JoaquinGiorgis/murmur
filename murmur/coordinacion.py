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
import re, unicodedata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import igraph as ig
import leidenalg as la

VENTANA = 600          # segundos
K_MIN, COS_MIN = 3, 0.15
K_MIN_REP, COS_MIN_REP = 2, 0.08
TAM_MIN, GRADO_MIN = 8, 2.5   # grado medio del grupo podado; grupos genuinos densos llegan a 2,0, campañas con pocas co-acciones a 2,17-2,77
PESO_RESP, PODA_FRAC = 3.0, 0.5
TAM_MAX_FRAC = 0.02   # una "operación" que abarca >2% de las cuentas es la masa de usuarios activos, no una operación
DENS_MIN = 0.09       # densidad del grupo podado (grado medio / (n-1)): operaciones 0,12-0,32; grupos genuinos <= 0,063


ULTIMOS_DESCARTADOS = []   # grupos podados que no pasaron grado/densidad en la última llamada a detectar (no cambia su salida)


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
    descartados = ULTIMOS_DESCARTADOS
    descartados.clear()
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
        else:
            descartados.append((acc[ii], round(grado, 2), round(densidad, 3)))
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


# --- alineados: cuentas activas de una operación que la co-acción par a par no alcanza a ver
ALIN_PUBS, ALIN_NUC = 2, 3   # ALIN_PUBS = 0: apagado
ALIN_FRAC = 0.1              # y al menos esta parte de todas las publicaciones de la cuenta: sin esto, las cuentas
                             # hiperactivas (100-300 comentarios) entran por puro volumen (modelo nulo: 0,61 -> 0,06 por set)


def sumar_alineados(c: pd.DataFrame, grupos: pd.DataFrame, ops: pd.DataFrame, narr, lado, log=print):
    """Suma a cada operación las cuentas que empujan lo mismo que el núcleo, a la vez que el núcleo, más de una vez.

    Cada operación empuja un lado de una narrativa (el más frecuente entre los comentarios del núcleo). Una cuenta
    entra si en al menos ALIN_PUBS publicaciones distintas tiene un comentario de esa narrativa y ese lado a menos
    de VENTANA de comentarios, también de esa narrativa y lado, de al menos ALIN_NUC cuentas distintas del núcleo.
    Las cuentas con actividad de cobertura reparten sus coincidencias entre muchos miembros (k <= 3 por par) y no
    forman pares sospechosos, pero caen una y otra vez dentro de la ráfaga; una cuenta genuina coincide a lo sumo
    una vez. `narr` y `lado` son la narrativa (-1 = ruido) y el lado de cada comentario, en el orden de `c`.
    No crea operaciones ni cambia las narrativas.
    """
    if ALIN_PUBS <= 0 or not len(c):
        return grupos, ops, 0
    g = dict(zip(grupos.cuenta_id, grupos.grupo))
    grp = c.cuenta_id.map(g).fillna("").values
    narr, lado = np.asarray(narr), np.asarray(lado)
    ts, cuenta, pub = c.ts.values, c.cuenta_id.values, c.publicacion_id.values
    npub = c.groupby("cuenta_id").publicacion_id.nunique()
    mejor = {}   # cuenta -> (publicaciones, operación)
    for op in sorted(set(grp) - {""}):
        del_op = (grp == op) & (narr >= 0)
        if not del_op.any():
            continue
        claves, cuantos = np.unique(np.stack([narr[del_op], lado[del_op]], 1), axis=0, return_counts=True)
        n_dom, s_dom = claves[cuantos.argmax()]
        dom = (narr == n_dom) & (lado == s_dom)
        nuc_i = np.nonzero(dom & (grp == op))[0]
        cand_i = np.nonzero(dom & (grp == ""))[0]
        if not len(cand_i):
            continue
        nuc_por_pub = pd.Series(nuc_i).groupby(pub[nuc_i]).apply(lambda s: s.values)
        cand_por_pub = pd.Series(cand_i).groupby(pub[cand_i]).apply(lambda s: s.values)
        pubs = defaultdict(set)
        for p, nn in nuc_por_pub.items():
            if p not in cand_por_pub.index or len(set(cuenta[nn])) < ALIN_NUC:
                continue
            cc = cand_por_pub[p]
            m = np.abs(ts[cc][:, None] - ts[nn][None, :]) <= VENTANA
            for i in np.nonzero(m.sum(1) >= ALIN_NUC)[0]:
                if len(set(cuenta[nn[m[i]]])) >= ALIN_NUC:
                    pubs[cuenta[cc[i]]].add(p)
        for cu, ps in pubs.items():
            if len(ps) >= max(ALIN_PUBS, ALIN_FRAC * npub[cu]) and (cu not in mejor or len(ps) > mejor[cu][0]):
                mejor[cu] = (len(ps), op)
    if mejor:
        grupos = grupos.copy()
        grupos["grupo"] = [mejor[cu][1] if cu in mejor else gr for cu, gr in zip(grupos.cuenta_id, grupos.grupo)]
        if len(ops):
            ops = ops.copy()
            ops["cuentas"] = ops.grupo.map(grupos[grupos.grupo != ""].groupby("grupo").size()).astype(int)
    log(f"  alineados sumados a operaciones: {len(mejor)}")
    return grupos, ops, len(mejor)


# --- vocabulario propio: cuentas de poca actividad que usan las consignas de una operación ya detectada
PROPIO_FRAC = 0.6                   # 0 = apagado. Parte mínima de las cuentas que usan un término que debe ser del núcleo
PROPIO_NUC, PROPIO_MIN = 0.2, 5     # el término lo usa al menos el 20% del núcleo (y al menos 5 cuentas)


def _terminos(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return set(re.findall(r"#?[a-z0-9_]{3,}", t))


def sumar_vocabulario(c: pd.DataFrame, grupos: pd.DataFrame, ops: pd.DataFrame, log=print):
    """Suma a cada operación las cuentas que usan su vocabulario propio dentro de una de sus ráfagas.

    Un término (palabra o hashtag) es propio de una operación si lo usa buena parte del núcleo y casi nadie más
    (el núcleo es >= PROPIO_FRAC de todas las cuentas que lo usan): eso lo miden los datos, no una lista. Una
    cuenta sin marcar entra si escribió un comentario con un término propio a menos de VENTANA de un comentario
    del núcleo en la misma publicación. Si los genuinos usan las mismas consignas, ningún término califica y la
    regla no hace nada. No crea operaciones.
    """
    if PROPIO_FRAC <= 0 or not len(c):
        return grupos, ops, 0
    g = dict(zip(grupos.cuenta_id, grupos.grupo))
    grp = c.cuenta_id.map(g).fillna("").values
    if not (grp != "").any():
        return grupos, ops, 0
    ts, cuenta, pub = c.ts.values, c.cuenta_id.values, c.publicacion_id.values
    tk = [_terminos(t) for t in c.texto.fillna("").values]
    del_nucleo = set().union(*(tk[i] for i in np.nonzero(grp != "")[0]))   # solo importan los términos que usa algún núcleo
    tk = [x & del_nucleo for x in tk]
    ii = np.repeat(np.arange(len(c)), [len(x) for x in tk])
    tc = pd.DataFrame({"i": ii, "t": [w for x in tk for w in x]})
    tc["cuenta"] = cuenta[tc.i.values]; tc["g"] = grp[tc.i.values]
    usan = tc.drop_duplicates(["t", "cuenta"])
    total = usan.groupby("t").size()
    nuevo = {}
    for op in sorted(set(grp) - {""}):
        n_nuc = len(set(cuenta[grp == op]))
        del_nuc = usan[usan.g == op].groupby("t").size()
        propios = del_nuc[(del_nuc >= max(PROPIO_MIN, PROPIO_NUC * n_nuc)) & (del_nuc / total.reindex(del_nuc.index) >= PROPIO_FRAC)].index
        if not len(propios):
            continue
        cand = np.unique(tc.i.values[tc.t.isin(propios).values & (tc.g.values == "")])
        nuc_i = np.nonzero(grp == op)[0]
        t_nuc = pd.Series(ts[nuc_i]).groupby(pub[nuc_i]).apply(lambda s: np.sort(s.values))
        for i in cand:
            if cuenta[i] in nuevo or pub[i] not in t_nuc.index:
                continue
            tn = t_nuc[pub[i]]
            if np.searchsorted(tn, ts[i] + VENTANA, "right") > np.searchsorted(tn, ts[i] - VENTANA, "left"):
                nuevo[cuenta[i]] = op
    if nuevo:
        grupos = grupos.copy()
        grupos["grupo"] = [nuevo.get(cu, gr) for cu, gr in zip(grupos.cuenta_id, grupos.grupo)]
        if len(ops):
            ops = ops.copy()
            ops["cuentas"] = ops.grupo.map(grupos[grupos.grupo != ""].groupby("grupo").size()).astype(int)
    log(f"  sumadas por vocabulario propio: {len(nuevo)}")
    return grupos, ops, len(nuevo)


# --- núcleos alineados: segunda oportunidad para los grupos que no pasaron grado/densidad
NUCLEO_ALIN = True    # True: usa ALIN_NUC, ALIN_PUBS y ALIN_FRAC (los mismos números que sumar_alineados) y TAM_MIN


def rescatar_nucleos(c: pd.DataFrame, grupos: pd.DataFrame, ops: pd.DataFrame, narr, lado, log=print):
    """Acepta como operación el núcleo de un grupo descartado si sus cuentas empujan lo mismo, juntas, más de una vez.

    En sets chicos una operación deja pocos pares sospechosos y el grupo podado no llega a GRADO_MIN/DENS_MIN (o
    arrastra genuinas que lo diluyen). Dentro de cada grupo descartado se busca el núcleo que se sostiene solo:
    cuentas que en al menos ALIN_PUBS publicaciones (y ALIN_FRAC de las suyas) comentan el lado dominante del grupo
    a menos de VENTANA de otras ALIN_NUC cuentas del núcleo que comentan lo mismo. Se itera hasta que no cambia; si
    quedan al menos TAM_MIN cuentas, es una operación. Las genuinas arrastradas no cumplen y quedan afuera.
    """
    if not NUCLEO_ALIN or not ULTIMOS_DESCARTADOS or not len(c):
        return grupos, ops, 0
    narr, lado = np.asarray(narr), np.asarray(lado)
    ts, cuenta, pub = c.ts.values, c.cuenta_id.values, c.publicacion_id.values
    npub = c.groupby("cuenta_id").publicacion_id.nunique()
    ya = set(grupos.cuenta_id[grupos.grupo != ""])
    nuevos = {}
    filas = ops.to_dict("records") if len(ops) else []
    for cuentas_g, grado, dens in ULTIMOS_DESCARTADOS:
        mi = set(cuentas_g) - ya
        while len(mi) >= TAM_MIN:
            m = np.isin(cuenta, list(mi)) & (narr >= 0)
            if not m.any():
                mi = set(); break
            claves, cuantos = np.unique(np.stack([narr[m], lado[m]], 1), axis=0, return_counts=True)
            n_dom, s_dom = claves[cuantos.argmax()]
            idx = np.nonzero(m & (narr == n_dom) & (lado == s_dom))[0]
            pubs = defaultdict(set)
            for p, ii in pd.Series(idx).groupby(pub[idx]):
                ii = ii.values
                if len(set(cuenta[ii])) <= ALIN_NUC:
                    continue
                cerca = np.abs(ts[ii][:, None] - ts[ii][None, :]) <= VENTANA
                for j in range(len(ii)):
                    if len(set(cuenta[ii[cerca[j]]]) - {cuenta[ii[j]]}) >= ALIN_NUC:
                        pubs[cuenta[ii[j]]].add(p)
            ok = {cu for cu, ps in pubs.items() if len(ps) >= max(ALIN_PUBS, ALIN_FRAC * npub[cu], 1)}
            if ok == mi:
                break
            mi = ok
        if len(mi) >= TAM_MIN:
            nombre = f"op{len(filas) + 1:02d}"
            filas.append({"grupo": nombre, "cuentas": len(mi), "grado_medio": grado, "densidad": dens})
            for cu in mi:
                nuevos[cu] = nombre
            ya |= mi
    if nuevos:
        grupos = grupos.copy()
        grupos["grupo"] = [nuevos.get(cu, gr) for cu, gr in zip(grupos.cuenta_id, grupos.grupo)]
        ops = pd.DataFrame(filas)
    log(f"  núcleos alineados rescatados: {len(set(nuevos.values()))} operaciones, {len(nuevos)} cuentas")
    return grupos, ops, len(nuevos)


# --- ráfagas alineadas: operaciones de cuentas con tan poca actividad que no llegan a formar pares sospechosos
RAFAGAS = True    # True: usa VENTANA, SAT_SIM, ALIN_NUC, ALIN_PUBS, ALIN_FRAC, TAM_MIN y TAM_MAX_FRAC (ningún número nuevo)


def rafagas_alineadas(c: pd.DataFrame, grupos: pd.DataFrame, ops: pd.DataFrame, narr, lado, E, kid, log=print):
    """Arma operaciones sin pasar por el grafo de pares: cuentas que repiten juntas el mismo texto del mismo lado.

    Un comentario está "en ráfaga" si a menos de VENTANA, en la misma publicación y con la misma narrativa y lado,
    hay comentarios casi iguales (coseno >= SAT_SIM, >= 4 palabras) de al menos ALIN_NUC cuentas distintas. Una
    cuenta queda si eso le pasa en al menos ALIN_PUBS publicaciones (y ALIN_FRAC de las suyas), contando solo a las
    cuentas que también quedan (se itera hasta que no cambia). Los sobrevivientes se agrupan por quién repite con
    quién. La masa genuina que usa las mismas plantillas, cuando aparece, forma un solo grupo enorme y se descarta
    con el mismo tope de siempre (TAM_MAX_FRAC); las operaciones son grupos chicos y cerrados. Un grupo que toca
    una operación ya detectada le suma sus cuentas; uno que no toca ninguna es una operación nueva.
    """
    if not RAFAGAS or not len(c):
        return grupos, ops, 0
    narr, lado = np.asarray(narr), np.asarray(lado)
    ts, cuenta, pub = c.ts.values, c.cuenta_id.values, c.publicacion_id.values
    nw = c.texto.fillna("").str.split().str.len().values
    npub = c.groupby("cuenta_id").publicacion_id.nunique()
    sirve = (narr >= 0) & (nw >= 4)
    clave = narr * 3 + lado + 1
    vec = {}   # comentario -> cuentas distintas con texto casi igual, cerca en el tiempo
    for _, ii in pd.Series(np.nonzero(sirve)[0]).groupby([pub[sirve], clave[sirve]]):
        ii = ii.values
        if len(ii) <= ALIN_NUC:
            continue
        ii = ii[np.argsort(ts[ii], kind="stable")]
        t = ts[ii]
        lo = np.searchsorted(t, t - VENTANA, "left"); hi = np.searchsorted(t, t + VENTANA, "right")
        for j in np.nonzero(hi - lo > ALIN_NUC)[0]:
            jj = ii[lo[j]:hi[j]]
            cer = jj[((E[kid[jj]] @ E[kid[ii[j]]]) >= SAT_SIM) & (cuenta[jj] != cuenta[ii[j]])]
            if len(cer) >= ALIN_NUC:
                vec[ii[j]] = set(cuenta[cer])
    vivos = None
    while True:
        pubs = defaultdict(set)
        for i, cs in vec.items():
            if vivos is not None and cuenta[i] not in vivos:
                continue
            if len(cs if vivos is None else cs & vivos) >= ALIN_NUC:
                pubs[cuenta[i]].add(pub[i])
        ok = {cu for cu, ps in pubs.items() if len(ps) >= max(ALIN_PUBS, ALIN_FRAC * npub[cu], 1)}
        if ok == vivos or len(ok) < TAM_MIN:
            break
        vivos = ok
    if len(ok) < TAM_MIN:
        log("  ráfagas alineadas: nada")
        return grupos, ops, 0
    padre = {u: u for u in ok}
    def raiz(u):
        while padre[u] != u:
            padre[u] = padre[padre[u]]; u = padre[u]
        return u
    for i, cs in vec.items():
        if cuenta[i] in ok:
            for v in cs & ok:
                padre[raiz(cuenta[i])] = raiz(v)
    comp = defaultdict(set)
    for u in sorted(ok):
        comp[raiz(u)].add(u)
    g = dict(zip(grupos.cuenta_id, grupos.grupo))
    tope = TAM_MAX_FRAC * c.cuenta_id.nunique()
    filas = ops.to_dict("records") if len(ops) else []
    nuevos = {}
    for miembros in sorted(comp.values(), key=lambda m: (-len(m), min(m))):
        if len(miembros) < TAM_MIN or len(miembros) > tope:
            continue
        de_op = pd.Series([g.get(u, "") for u in miembros])
        de_op = de_op[de_op != ""]
        if len(de_op):
            nombre = de_op.value_counts().index[0]
        else:
            nombre = f"op{len(filas) + 1:02d}"
            filas.append({"grupo": nombre, "cuentas": len(miembros), "grado_medio": 0.0, "densidad": 0.0})
        for u in miembros:
            if not g.get(u, ""):
                nuevos[u] = nombre
    if nuevos:
        grupos = grupos.copy()
        grupos["grupo"] = [nuevos.get(cu, gr) for cu, gr in zip(grupos.cuenta_id, grupos.grupo)]
        ops = pd.DataFrame(filas)
        ops["cuentas"] = ops.grupo.map(grupos[grupos.grupo != ""].groupby("grupo").size()).astype(int)
    log(f"  ráfagas alineadas: {len(nuevos)} cuentas nuevas en {len(set(nuevos.values()))} operaciones")
    return grupos, ops, len(nuevos)
