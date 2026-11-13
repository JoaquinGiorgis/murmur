"""Tarea A: narrativas, ruido, postura y fecha de origen.

1. Clusters finos: UMAP + HDBSCAN sobre embeddings de los textos únicos con
   contenido (>= 5 palabras). Salen cientos de clusters muy puros: cada uno es
   una variante de un mismo reclamo, casi siempre con una sola postura.
2. Las respuestas cosen los clusters en narrativas. Por las reglas del desafío,
   una respuesta corta ("Tal cual") hereda narrativa y postura del padre, y una
   larga es la misma narrativa con la postura opuesta. Eso arma un grafo entre
   clusters (Leiden) y, con el signo de cada arista, un 2-coloreo que separa el
   lado que afirma del que refuta. No depende del tema ni de palabras clave, y
   aguanta que la redacción cambie con los días.
3. Ruido: clusters que casi no participan de conversaciones, y textos muy
   repetidos que nunca reciben ni hacen respuestas (avisos, chistes, "primero!").
4. Postura: el 2-coloreo dice qué comentarios están del mismo lado; falta decidir
   qué lado "apoya". Con IA (opcional) se le pregunta a un modelo chico una vez
   por narrativa. Sin IA: si una operación coordinada se concentra en un lado,
   ese lado afirma (las operaciones amplifican reclamos); si no, el lado más grande.
5. Origen: primer momento en que la narrativa junta un volumen sostenido en 3 h
   (no el primer comentario suelto, que suele ser un error de asignación).
"""
import heapq, json, os, random, re, threading, time, unicodedata, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la

MIN_PALABRAS = 5
HDB_MCS, HDB_MS = 40, 10
ARISTA_MIN = 3
RESOLUCION = 1.0
TASA_RESP_RUIDO = 0.02
SIM_ASIGNAR = 0.80
SIM_COORD = 0.55
FRAG_FRAC, SIM_FRAG = 0.003, 0.5   # fragmento = narrativa de < 0,3%: la real más chica en práctica y recortes pesa 0,82%; con 1% se fundían emergentes nuevas (H3, B4) en otras viejas
IA_PARALELO = 4   # llamadas a la IA simultáneas (los 429 se reintentan con espera)
UMAP_COMPAT = None   # None = pipeline 0.5 de umap-learn (spectral, 1 hilo con semilla); False = init recursiva + Adam en paralelo
N_EJ, N_VOTOS = 16, 8   # 8 votos: con 4, una narrativa dudosa (la IA erra 1 de cada 4) salía mal ~1 de cada 20 corridas
RUIDO_REPETIDO = 20
RUIDO_REP_ESCALA = True       # el umbral de repeticiones crece con el volumen (nunca baja de 20): calibrado para ~200k
REF_COMENTARIOS = 200_000
REF_TASA_RESP = 0.1403        # tasa de respuestas de práctica
TASA_RELATIVA = True          # si el set responde menos que práctica, el umbral de ruido por tasa de respuesta baja en
                              # proporción (nunca sube): con pocas respuestas, menos clusters genuinos quedan como ruido
RUIDO_ALARMA = 0.35   # solo avisa; no cambia la salida
RUIDO_HILO = 0.0      # 0 = apagado. Un cluster fino pasa a ruido si la parte de sus comentarios que está en un hilo (es
                      # respuesta o recibió respuesta) es menor que RUIDO_HILO veces la del set entero (relativo, no absoluto)
FRAG_RESPUESTAS = True   # True: una narrativa chica hecha solo de respuestas (sin raíces) también se une a una grande
FRAG_VARIANTES = 3    # con FRAG_VECINO: cuántas variantes parecidas de un mismo lado se promedian (1 = la más parecida sola)
FRAG_VECINO = True    # True: el fragmento se une al lado de la narrativa grande que tiene la variante (cluster fino) más
                      # parecida, no al promedio de todo el lado: el promedio de una narrativa grande y variada queda lejos
                      # de cada variante y una narrativa chica y compacta del mismo tema se la lleva
PARTIR_MIXTOS = True  # True: si las respuestas largas a comentarios de un cluster fino votan otra narrativa (al menos
                      # ARISTA_MIN votos), el cluster mezcla dos temas: cada comentario con votos toma la narrativa que le
                      # votan y los demás siguen al compañero de cluster con votos que más se les parece
RUIDO_LETRAS = False  # True: la clave del texto repetido ignora letras traspuestas dentro de cada palabra ('Alguein' = 'Alguien'):
                      # las variantes con errores de tipeo de un aviso repetido que nadie contesta también son ruido
ANCLA_POSTURA = True  # True: con alguna de las opciones de abajo prendidas, el lado que 'apoya' sin IA se decide igual que
                      # sin ellas (primera pasada, sin IA) y se traslada por solapamiento: en narrativas 50/50 el 'lado mayor'
                      # depende de un puñado de comentarios y cualquier mejora de asignación lo daba vuelta al azar
RESTO_VECINO = True   # True: los comentarios raíz que HDBSCAN dejó sin cluster y no entraron por SIM_ASIGNAR no van a ruido
                      # por descarte. Si el texto no es ruido conocido (repetido que nadie contesta), toma la narrativa que
                      # votan las respuestas largas a ese mismo texto; si no tiene, la del texto con cluster más parecido,
                      # siempre que se parezca más a una narrativa que al ruido conocido. Sin umbral de similitud.
HILO_RESCATE = False  # True: un comentario raíz sin cluster que recibió respuestas no es ruido (el ruido no se contesta):
                      # toma la narrativa que votan sus respuestas largas o, si no hay, la del cluster más parecido
ORIGEN_VENTANA = 3 * 3600
ORIGEN_MIN, ORIGEN_FRAC = 10, 0.005
ORIGEN_TOPE = 25   # el umbral crece con el tamaño (frena comentarios mal asignados) pero con tope: una
                   # narrativa grande que nace lento (29 en 3 h) no queda fechada horas tarde

STOP = set("""a al algo ante asi aun bien cada como con contra cual de del desde donde el ella ellos en entre era es esa ese
eso esta estan este esto fue ha hace hay la las le les lo los mas me mi muy nada ni no nos o otra otro para pero poco por porque
que q se ser si sin sobre son su sus tambien te tiene todo todos tu un una uno unos ya y yo x d vos che mira jaja ojo bueno
verdad digo decirlo hay escuchen sinceramente nadie dice lo que siempre igual te juro posta
dia dias ano anos ahora aca alla haya seria sea mismo mismos misma nuevo nueva nuevas nuevos poder puede pueden hacer tienen tener vez vamos van estar estamos solo hoy mañana ayer gente""".split())


def _k2(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^a-z ]+", " ", t)
    t = re.sub(r"(.)\1+", r"\1", t)
    return " ".join(sorted(t.split()))


def _k3(t):
    return " ".join(sorted("".join(sorted(w)) for w in _k2(t).split()))


def _tokens(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    w = [x for x in re.findall(r"[a-z]{3,}", t) if x not in STOP]
    return w + [a + " " + b for a, b in zip(w, w[1:])]


def _norm_seg(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z ]+", "", s).strip()


def _segmentos(t):
    return [x for x in (_norm_seg(p) for p in re.split(r"[.:,;!?¡¿]+", str(t))) if x]


def _limpiar_titulo(t, muletillas):
    t = "".join(ch for ch in str(t) if ch.isalnum() or ch.isspace() or ch in ",.:;?!¿¡'\"-")
    partes = [p.strip() for p in re.split(r"([.:,;!?]+)", t)]
    piezas = [partes[i] for i in range(0, len(partes), 2) if partes[i]]
    piezas = [p for p in piezas if _norm_seg(p) not in muletillas]
    out = ", ".join(piezas).strip(" ,")
    return (out[:1].upper() + out[1:]) if out else str(t).strip()


def clusters_finos(E, nw_por_kid, seed=42, log=print):
    import umap, hdbscan
    cand = np.nonzero(nw_por_kid >= MIN_PALABRAS)[0]
    X = umap.UMAP(n_components=8, n_neighbors=30, min_dist=0.0, metric="cosine",
                  random_state=seed, low_memory=True, compatibility_layout=UMAP_COMPAT).fit_transform(E[cand])
    cl = hdbscan.HDBSCAN(min_cluster_size=HDB_MCS, min_samples=HDB_MS).fit(X)
    lab = np.full(len(E), -1)
    lab[cand] = cl.labels_
    log(f"  clusters finos: {cl.labels_.max() + 1}")
    return lab


def narrativas(c, E, kid, grupos=None, log=print, llm=None):
    """Punto de entrada. Con ANCLA_POSTURA y alguna opción de asignación prendida corre dos pasadas (UMAP una sola vez)."""
    global RESTO_VECINO, PARTIR_MIXTOS, FRAG_VECINO, HILO_RESCATE, FRAG_RESPUESTAS
    opciones = (RESTO_VECINO, PARTIR_MIXTOS, FRAG_VECINO, HILO_RESCATE, FRAG_RESPUESTAS)
    if not (ANCLA_POSTURA and any(opciones)):
        return _narrativas(c, E, kid, grupos=grupos, log=log, llm=llm)
    nw = c.texto.fillna("").str.split().str.len().values
    nw_kid = np.zeros(len(E)); np.maximum.at(nw_kid, kid, nw)
    lab_kid = clusters_finos(E, nw_kid, log=log)
    try:
        RESTO_VECINO = PARTIR_MIXTOS = FRAG_VECINO = HILO_RESCATE = FRAG_RESPUESTAS = False
        df0, _, _ = _narrativas(c, E, kid, grupos=grupos, log=lambda s: None, llm=None, _lab=lab_kid)
    finally:
        RESTO_VECINO, PARTIR_MIXTOS, FRAG_VECINO, HILO_RESCATE, FRAG_RESPUESTAS = opciones
    ancla = np.where(df0.postura.values == "apoya", 1, np.where(df0.postura.values == "rechaza", -1, 0))
    return _narrativas(c, E, kid, grupos=grupos, log=log, llm=llm, _lab=lab_kid, _ancla=ancla)


def _narrativas(c, E, kid, grupos=None, log=print, llm=None, _lab=None, _ancla=None):
    N = len(c)
    nw = c.texto.fillna("").str.split().str.len().values
    nw_kid = np.zeros(len(E)); np.maximum.at(nw_kid, kid, nw)
    lab_kid = clusters_finos(E, nw_kid, log=log) if _lab is None else _lab
    F = lab_kid[kid]
    nfc = max(F.max() + 1, 1)

    idx = pd.Series(np.arange(N), index=c.comentario_id)
    pi = c.respuesta_a.map(idx).fillna(-1).astype(int).values
    isrep = pi >= 0
    corto = nw < MIN_PALABRAS

    # grafo de clusters desde respuestas
    pf = np.where(isrep, F[np.maximum(pi, 0)], -1)
    m = isrep & (F >= 0) & (pf >= 0)
    w = defaultdict(float); sgn = defaultdict(float)
    toc = np.zeros(nfc)
    for x, y, s in zip(F[m], pf[m], corto[m]):
        toc[x] += 1; toc[y] += 1
        if x == y:
            continue
        kk = (min(x, y), max(x, y)); w[kk] += 1; sgn[kk] += 1 if s else -1
    tam = np.bincount(F[F >= 0], minlength=nfc)
    aristas = [e for e in w if w[e] >= ARISTA_MIN]
    pesos = [w[e] for e in aristas]
    tasa = isrep.mean() if N else 0.0
    g = ig.Graph(n=nfc, edges=aristas, edge_attrs={"weight": pesos})
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                             resolution_parameter=RESOLUCION, seed=1)
    narr = np.array(part.membership)
    tasa_ruido = TASA_RESP_RUIDO * min(1.0, tasa / REF_TASA_RESP) if TASA_RELATIVA else TASA_RESP_RUIDO
    narr[toc / np.maximum(tam, 1) < tasa_ruido] = -1
    recibio = np.zeros(N, bool); recibio[pi[isrep]] = True
    if RUIDO_HILO > 0:
        en_hilo = isrep | recibio
        fh = np.bincount(F[F >= 0], weights=en_hilo[F >= 0], minlength=nfc) / np.maximum(tam, 1)
        narr[fh < RUIDO_HILO * en_hilo.mean()] = -1

    # 2-coloreo con signo (árbol de máximo peso)
    lado = np.zeros(nfc, int)
    ady = defaultdict(list)
    for (x, y), ww in w.items():
        if ww >= ARISTA_MIN and narr[x] >= 0 and narr[x] == narr[y]:
            s = 1 if sgn[(x, y)] > 0 else -1
            ady[x].append((ww, y, s)); ady[y].append((ww, x, s))
    visto = np.zeros(nfc, bool)
    for st in np.argsort(-tam):
        if visto[st] or narr[st] < 0:
            continue
        visto[st] = True; lado[st] = 1
        h = [(-ww, st, v, s) for ww, v, s in ady[st]]; heapq.heapify(h)
        while h:
            _, u, v, s = heapq.heappop(h)
            if visto[v]:
                continue
            visto[v] = True; lado[v] = lado[u] * s
            for ww, nb, s2 in ady[v]:
                if not visto[nb]:
                    heapq.heappush(h, (-ww, v, nb, s2))

    # asignación de comentarios raíz
    cn = np.full(N, -1); cs = np.zeros(N, int)
    raiz = (F >= 0) & ~isrep
    cn[raiz] = narr[F[raiz]]; cs[raiz] = lado[F[raiz]]
    if PARTIR_MIXTOS:
        vot = defaultdict(list)   # comentario raíz con cluster -> clusters de sus respuestas largas con narrativa
        for i in np.nonzero(isrep & ~corto & (F >= 0))[0]:
            p_ = pi[i]
            if raiz[p_] and narr[F[i]] >= 0 and narr[F[p_]] >= 0:
                vot[p_].append(F[i])
        por_cl = defaultdict(dict)
        for p_, fs in vot.items():
            ns, cu = np.unique(narr[fs], return_counts=True)
            nv = ns[cu.argmax()]
            sv = -1 if sum(lado[f] for f in fs if narr[f] == nv) > 0 else 1
            por_cl[F[p_]][p_] = (nv, sv) if nv != narr[F[p_]] else (narr[F[p_]], lado[F[p_]])
        n_part = n_mov = 0
        for f, pv in por_cl.items():
            otros = defaultdict(int)
            for nv, _ in pv.values():
                if nv != narr[f]:
                    otros[nv] += 1
            if not otros or max(otros.values()) < ARISTA_MIN:
                continue
            mem = np.nonzero(raiz & (F == f))[0]
            sem = np.array(list(pv))
            cerca = sem[(E[kid[mem]] @ E[kid[sem]].T).argmax(1)]
            for i, sm in zip(mem, cerca):
                nv, sv = pv[sm]
                if nv != narr[f]:
                    cn[i], cs[i] = nv, sv; n_mov += 1
            n_part += 1
        log(f"  clusters mixtos partidos: {n_part} ({n_mov} comentarios cambian de narrativa)")
    cent = np.zeros((nfc, E.shape[1]), np.float32)
    np.add.at(cent, F[F >= 0], E[kid[F >= 0]])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    sinc = np.nonzero((~isrep) & (F < 0) & (nw >= 4))[0]
    for i0 in range(0, len(sinc), 20000):  # por lotes para acotar memoria
        ii = sinc[i0:i0 + 20000]
        S = E[kid[ii]] @ cent.T
        b = S.argmax(1); bs = S.max(1)
        ok = (bs >= SIM_ASIGNAR) & (narr[b] >= 0)
        cn[ii[ok]] = narr[b[ok]]; cs[ii[ok]] = lado[b[ok]]

    if HILO_RESCATE:
        hijos = defaultdict(list)
        for i in np.nonzero(isrep & ~corto & (F >= 0))[0]:
            if narr[F[i]] >= 0:
                hijos[pi[i]].append(i)
        vivos = np.nonzero(narr >= 0)[0]
        for i in np.nonzero((~isrep) & (cn < 0) & (F < 0) & recibio)[0]:
            hh = hijos.get(i)
            if hh:   # el hilo vota: la narrativa más frecuente entre sus respuestas largas; lado opuesto al de esas respuestas
                ns, cu = np.unique([narr[F[h]] for h in hh], return_counts=True)
                cn[i] = ns[cu.argmax()]
                ld = sum(lado[F[h]] for h in hh if narr[F[h]] == cn[i])
                cs[i] = -1 if ld > 0 else 1
            elif len(vivos):
                b = vivos[int((cent[vivos] @ E[kid[i]]).argmax())]
                cn[i] = narr[b]; cs[i] = lado[b]

    # ruido: textos repetidos que nunca entran en una conversación
    k2 = c.texto.map(_k3 if RUIDO_LETRAS else _k2).values
    en_resp = set(k2[isrep]) | set(k2[pi[isrep]])
    cnt = pd.Series(k2).value_counts()
    umbral_rep = RUIDO_REPETIDO * max(1.0, N / REF_COMENTARIOS) if RUIDO_REP_ESCALA else RUIDO_REPETIDO
    ruidosos = set(cnt[cnt >= umbral_rep].index) - en_resp
    nk = np.isin(k2, list(ruidosos)) & ~isrep
    cn[nk] = -1

    extra = np.zeros(N, bool)   # asignados por RESTO_VECINO (y lo que hereda de ellos): no deciden qué lado es el mayor
    if RESTO_VECINO:
        # lo de las cuentas coordinadas sigue por su propio camino (SIM_COORD, más abajo), que mira el lado completo
        de_op = c.cuenta_id.isin(set(grupos.cuenta_id[grupos.grupo != ""])).values if grupos is not None else np.zeros(N, bool)
        resto = np.nonzero((~isrep) & (cn < 0) & (F < 0) & (nw >= 4) & ~nk & ~de_op)[0]
        if len(resto):
            kid_ruidoso = np.zeros(len(E), bool); kid_ruidoso[kid[nk]] = True
            con_cl = lab_kid >= 0
            ej_narr = np.nonzero(con_cl & (narr[np.maximum(lab_kid, 0)] >= 0) & ~kid_ruidoso)[0]
            ej_ruido = np.nonzero((con_cl & (narr[np.maximum(lab_kid, 0)] < 0)) | kid_ruidoso)[0]
            # 1) voto del hilo, por texto: respuestas largas con narrativa a cualquier comentario con ese mismo texto
            votos = defaultdict(list)
            for i in np.nonzero(isrep & ~corto & (F >= 0))[0]:
                if narr[F[i]] >= 0:
                    votos[kid[pi[i]]].append(F[i])
            sin_voto = []
            for i in resto:
                fs = votos.get(kid[i])
                if fs:
                    ns, cu = np.unique(narr[fs], return_counts=True)
                    cn[i] = ns[cu.argmax()]
                    cs[i] = -1 if sum(lado[f] for f in fs if narr[f] == cn[i]) > 0 else 1
                else:
                    sin_voto.append(i)
            # 2) vecino más cercano: narrativa del texto con cluster más parecido, si gana contra el ruido conocido
            sin_voto = np.array(sin_voto, dtype=int)
            if len(sin_voto) and len(ej_narr):
                uk, inv = np.unique(kid[sin_voto], return_inverse=True)
                mejor = np.zeros(len(uk), int); gana = np.zeros(len(uk), bool)
                paso = max(64, int(2e8 // max(len(ej_narr), 1)))
                for i0 in range(0, len(uk), paso):
                    q = E[uk[i0:i0 + paso]]
                    S = q @ E[ej_narr].T
                    mejor[i0:i0 + paso] = ej_narr[S.argmax(1)]
                    sz = (q @ E[ej_ruido].T).max(1) if len(ej_ruido) else np.full(len(q), -1.0)
                    gana[i0:i0 + paso] = S.max(1) > sz
                f = lab_kid[mejor[inv]]; ok = gana[inv]
                cn[sin_voto[ok]] = narr[f[ok]]; cs[sin_voto[ok]] = lado[f[ok]]
            extra[resto] = cn[resto] >= 0
            log(f"  resto sin cluster: {len(resto)}; con voto de hilo {len(resto) - len(sin_voto)}, quedan como ruido {int((cn[resto] < 0).sum())}")

    # respuestas: heredan del padre, en orden temporal
    for i in np.argsort(c.ts.values, kind="stable"):
        if not isrep[i]:
            continue
        p = pi[i]
        if cn[p] < 0 and not corto[i] and F[i] >= 0 and narr[F[i]] >= 0:
            cn[i] = narr[F[i]]; cs[i] = lado[F[i]]; continue
        cn[i] = cn[p]; extra[i] = extra[p] and (corto[i] or F[i] < 0 or narr[F[i]] < 0)   # una respuesta larga con cluster propio ya contaba
        cs[i] = cs[p] if corto[i] else -cs[p]
        if not corto[i] and F[i] >= 0 and narr[F[i]] == cn[p] and lado[F[i]] != 0:
            cs[i] = lado[F[i]]

    # lo que empuja una operación coordinada no es ruido aunque sea una plantilla que nadie responde:
    # se asigna a la narrativa y al lado más cercanos por significado
    if grupos is not None:
        coord_c = set(grupos.cuenta_id[grupos.grupo != ""])
        es_coord = c.cuenta_id.isin(coord_c).values
        asign = (cn >= 0) & ~isrep & ~extra
        claves_ns = sorted(set(zip(cn[asign], cs[asign])))
        if claves_ns:
            C = np.zeros((len(claves_ns), E.shape[1]), np.float32)
            pos = {k: j for j, k in enumerate(claves_ns)}
            j_of = np.array([pos[(n, s)] for n, s in zip(cn[asign], cs[asign])], dtype=int)
            np.add.at(C, j_of, E[kid[asign]])
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
            ii = np.nonzero(es_coord & (cn < 0) & ~isrep & (nw >= MIN_PALABRAS))[0]
            if len(ii):
                S = E[kid[ii]] @ C.T
                b = S.argmax(1); ok = S.max(1) >= SIM_COORD
                for i, jj in zip(ii[ok], b[ok]):
                    cn[i], cs[i] = claves_ns[jj]
                # sus respuestas cortas heredan
                for i in np.argsort(c.ts.values, kind="stable"):
                    if isrep[i] and cn[i] < 0 and cn[pi[i]] >= 0:
                        cn[i] = cn[pi[i]]; cs[i] = cs[pi[i]] if corto[i] else -cs[pi[i]]

    # renumerar narrativas por tamaño
    vals, cnts = np.unique(cn[cn >= 0], return_counts=True)
    orden = vals[np.argsort(-cnts)]
    ren = {v: i + 1 for i, v in enumerate(orden)}
    cn = np.array([ren.get(v, -1) for v in cn])

    # fragmentos: narrativas chicas que son pedazos de una grande (una variante de reclamo o de refutación
    # que quedó sin cosido suficiente). Cada lado del fragmento se une al lado más parecido de una narrativa grande.
    tot = (cn >= 0).sum()
    tam_n = pd.Series(cn[cn >= 0]).value_counts()
    grandes = set(tam_n[tam_n >= FRAG_FRAC * tot].index)
    chicas = set(tam_n.index) - grandes
    if grandes and chicas:
        base = (cn >= 0) & ~isrep & ~extra
        def centroides(mask):
            keys = sorted(set(zip(cn[mask], cs[mask])))
            C = np.zeros((len(keys), E.shape[1]), np.float32)
            pos = {k: j for j, k in enumerate(keys)}
            np.add.at(C, np.array([pos[k] for k in zip(cn[mask], cs[mask])], dtype=int), E[kid[mask]])
            return keys, C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        kg, Cg = centroides(base & np.isin(cn, list(grandes)))
        if FRAG_VECINO:
            mg = base & np.isin(cn, list(grandes)) & (F >= 0)
            tri = pd.DataFrame({"f": F[mg], "n": cn[mg], "s": cs[mg]})
            dom = tri.groupby("f")[["n", "s"]].agg(lambda v: v.mode().iat[0])   # narrativa y lado mayoritarios de cada cluster fino
            if len(dom) and FRAG_VARIANTES <= 1:
                kg = [(int(a), int(b)) for a, b in zip(dom.n, dom.s)]
                Cg = cent[dom.index.values]
            elif len(dom):
                # una sola variante parecida puede ser una muletilla ("es mentira", "es falso") compartida entre temas:
                # el puntaje de cada lado es el promedio de sus FRAG_VARIANTES variantes más parecidas al fragmento
                claves_v = [(int(a), int(b)) for a, b in zip(dom.n, dom.s)]
                kg = sorted(set(claves_v))
                Cg = None
        m_ch = np.isin(cn, list(chicas))
        if FRAG_RESPUESTAS:   # un lado de una narrativa chica hecho solo de respuestas no tiene raíces: sin esto nunca se fusiona
            con_raiz = set(zip(cn[base & m_ch], cs[base & m_ch]))
            sin_raiz = m_ch & np.array([(a, b) not in con_raiz for a, b in zip(cn, cs)])
            kc, Cc = centroides((base & m_ch) | sin_raiz)
        else:
            kc, Cc = centroides(base & m_ch)
        if Cg is not None:
            S = Cc @ Cg.T
        else:
            Sv = Cc @ cent[dom.index.values].T
            S = np.full((len(kc), len(kg)), -1.0, np.float32)
            for j, kk in enumerate(kg):
                cols = [q for q, cv in enumerate(claves_v) if cv == kk]
                S[:, j] = np.sort(Sv[:, cols], axis=1)[:, -FRAG_VARIANTES:].mean(1)
        destino = {kc[i]: kg[int(S[i].argmax())] for i in range(len(kc)) if S[i].max() >= SIM_FRAG}
        for i in np.nonzero(np.isin(cn, list(chicas)))[0]:
            k = (cn[i], cs[i])
            if k in destino:
                cn[i], cs[i] = destino[k]
        vals, cnts = np.unique(cn[cn >= 0], return_counts=True)
        ren = {v: i + 1 for i, v in enumerate(vals[np.argsort(-cnts)])}
        cn = np.array([ren.get(v, -1) for v in cn])

    df = pd.DataFrame({"comentario_id": c.comentario_id.values, "n": cn, "s": cs, "ts": c.ts.values,
                       "texto": c.texto.values, "cuenta_id": c.cuenta_id.values, "rep": isrep, "kid": kid,
                       "respuesta_a": c.respuesta_a.values})
    df["extra"] = extra
    coord = set()
    if grupos is not None:
        coord = set(grupos.cuenta_id[grupos.grupo != ""])
    df["coord"] = df.cuenta_id.isin(coord)

    filas = []
    tokdf = []
    for n, gd in df[df.n >= 0].groupby("n"):
        gb = gd[~gd.extra] if gd.extra.any() and not gd.extra.all() else gd
        s1, s2 = (gb.s == 1).sum(), (gb.s == -1).sum()
        apo, metodo = (1 if s1 >= s2 else -1), "lado mayor"
        cg = gd[gd.coord & ~gd.rep]
        if len(cg) >= 30:
            f1 = (cg.s == 1).mean()
            if f1 >= 0.8: apo, metodo = 1, "operación coordinada"
            elif f1 <= 0.2: apo, metodo = -1, "operación coordinada"
        ts = np.sort(gd.ts.values)
        thr = max(ORIGEN_MIN, min(int(ORIGEN_FRAC * len(ts)), ORIGEN_TOPE))
        j = np.searchsorted(ts, ts + ORIGEN_VENTANA, side="right") - np.arange(len(ts))
        hit = np.nonzero(j >= thr)[0]
        origen = int(ts[hit[0]] if len(hit) else ts[0])
        filas.append({"n": n, "apo": apo, "metodo_postura": metodo, "origen": origen, "comentarios": len(gd)})
    nd = pd.DataFrame(filas, columns=["n", "apo", "metodo_postura", "origen", "comentarios"])
    if _ancla is not None and len(nd):
        df["ancla"] = _ancla
        for i, r in nd.iterrows():
            gd = df[(df.n == r.n) & (df.ancla != 0)]
            if len(gd):
                acuerdo = (gd.s.values * gd.ancla.values).sum()   # > 0: el lado +1 es el que la primera pasada llamó "apoya"
                if acuerdo != 0:
                    nd.loc[i, "apo"] = 1 if acuerdo > 0 else -1
    if nd.empty:
        raise RuntimeError("no quedó ninguna narrativa (¿el set casi no tiene respuestas?)")

    # títulos y frases clave sin IA: c-TF-IDF sobre el lado que afirma + texto más central
    apo_map = dict(zip(nd.n, nd.apo))
    lado_a = df[(df.n >= 0) & ~df.rep & (df.s == df.n.map(apo_map))]
    docs = lado_a.groupby("n").texto.apply(lambda s: [t for x in s.sample(min(len(s), 3000), random_state=0) for t in _tokens(x)])
    tf = {n: pd.Series(v).value_counts() for n, v in docs.items()}
    dfreq = pd.Series([t for v in tf.values() for t in v.index]).value_counts()
    nN = len(tf)
    # muletillas: segmentos ("te digo algo", "lamentable") que aparecen en muchas narrativas distintas
    seg_n = defaultdict(set)
    for n, s in lado_a.groupby("n").texto:
        for x in s.sample(min(len(s), 2000), random_state=0):
            for sg in _segmentos(x):
                seg_n[sg].add(n)
    muletillas = {sg for sg, ns in seg_n.items() if len(ns) >= max(3, 0.25 * nN)}
    claves, titulos = {}, {}
    for n, s in tf.items():
        sc = (s / s.sum()) * np.log(1 + nN / dfreq.reindex(s.index).values)
        claves[n] = list(sc.sort_values(ascending=False).index[:5])
        sub = lado_a[lado_a.n == n]
        vc = sub.kid.value_counts()
        ks = vc.index[:200]
        v = E[ks]; cen = v.mean(0) / (np.linalg.norm(v.mean(0)) + 1e-9)
        top = ks[:30][np.argsort(-(E[ks[:30]] @ cen))]  # textos frecuentes (sin typos) y centrales
        cands = [_limpiar_titulo(sub[sub.kid == k].texto.iloc[0], muletillas) for k in top[:10]]
        cands = [t for t in cands if len(t.split()) >= 4] or cands
        titulos[n] = cands[0]
    nd["titulo"] = nd.n.map(titulos)
    nd["frases_clave"] = nd.n.map(lambda n: claves.get(n, []))

    uso_ia = {"tokens": 0, "usd": 0.0, "llamadas": 0}
    if llm is not None:
        def ej(p, k):
            # ejemplos limpios: sin muletillas ni emojis, que confunden ("hay trabajo... Vergüenza.")
            out = []
            for t in p.sample(frac=1, random_state=k):
                t2 = _limpiar_titulo(t, muletillas)
                if len(t2.split()) >= 4 and t2 not in out:
                    out.append(t2)
                if len(out) == N_EJ:
                    break
            return out
        # 1) armar todas las preguntas: varias por narrativa, alternando qué grupo va primero (sesgo de posición)
        tareas = []
        for i, r in nd.iterrows():
            gd = df[(df.n == r.n) & ~df.rep]
            pa = gd[gd.s == 1].texto.drop_duplicates()
            pb = gd[gd.s == -1].texto.drop_duplicates()
            if len(pa) < 3 or len(pb) < 3:
                continue
            for k in range(N_VOTOS):
                invertido = k % 2 == 1
                A, B = (ej(pb, k), ej(pa, k)) if invertido else (ej(pa, k), ej(pb, k))
                tareas.append((i, invertido, A, B))

        def preguntar(t):
            try:
                return llm.juzgar(t[2], t[3], uso_ia)
            except Exception as e:  # una respuesta rara de la API no tumba la corrida
                log(f"  IA: respuesta inesperada ({e!r}); sigo")
                return None
        # 2) hacerlas en paralelo (el orden de los resultados es el de las tareas: el voto no depende de quién termina antes)
        with ThreadPoolExecutor(IA_PARALELO) as ex:
            resultados = list(ex.map(preguntar, tareas))
        # 3) contar votos por narrativa
        por_narr = defaultdict(list)
        for (i, invertido, _, _), res in zip(tareas, resultados):
            if res:
                por_narr[i].append((1 if (res["lado"] == "A") != invertido else -1, res))
        for i, r in nd.iterrows():
            votos = [v for v, _ in por_narr.get(i, [])]
            respuestas = [rr for _, rr in por_narr.get(i, [])]
            if votos and sum(votos) != 0:
                apo = 1 if sum(votos) > 0 else -1
                nd.loc[i, "apo"] = apo
                nd.loc[i, "metodo_postura"] = f"ia ({votos.count(apo)}/{len(votos)} votos)"
                # el título sale de una respuesta que eligió el mismo lado que la mayoría
                nombre = next(rr for v, rr in zip(votos, respuestas) if v == apo)
                nd.loc[i, "titulo"] = str(nombre.get("titulo") or "").strip() or r.titulo
                fc = nombre.get("frases_clave")
                fc = [fc] if isinstance(fc, str) else (fc if isinstance(fc, list) else [])
                fc = [str(x).replace("|", " ").strip() for x in fc if str(x).strip()]
                if fc:
                    nd.at[i, "frases_clave"] = fc[:6]
        if llm.cortes:
            log(f"  IA diagnóstico: stop_reason {dict(llm.cortes)}, con razonamiento {llm.pensadas}")
        if llm.fallos or llm.abortado:
            log(f"  AVISO IA: {llm.fallos} llamadas fallidas{' — ' + llm.abortado if llm.abortado else ''}; "
                f"las narrativas sin votos usan la postura heurística")

    # restos sin título (narrativas mínimas): el texto más repetido de la narrativa
    for i, r in nd[nd.titulo.isna() | (nd.titulo.astype(str).str.strip() == "")].iterrows():
        tx = df[df.n == r.n].texto
        nd.loc[i, "titulo"] = _limpiar_titulo(tx.value_counts().index[0], muletillas) if len(tx) else f"narrativa {r.n}"
    apo_map = dict(zip(nd.n, nd.apo))
    df["narrativa"] = np.where(df.n >= 0, "n" + df.n.astype(str).str.zfill(2), "ninguna")
    df["postura"] = [("apoya" if s == apo_map.get(n, 1) else "rechaza") if n >= 0 else "" for n, s in zip(df.n, df.s)]
    nd["narrativa"] = "n" + nd.n.astype(str).str.zfill(2)
    nd["fecha_origen"] = pd.to_datetime(nd.origen, unit="s", utc=True).dt.tz_convert("America/Argentina/Buenos_Aires").map(lambda t: t.isoformat())
    ruido = (df.n < 0).mean()
    log(f"  narrativas: {len(nd)}  ruido: {ruido:.1%}")
    if ruido > RUIDO_ALARMA:
        log(f"  AVISO: {ruido:.0%} de los comentarios quedó como ruido (práctica: ~20%). Revisar el set: "
            f"pocas respuestas o textos muy cortos dejan clusters sin conversación.")
    return df, nd, uso_ia


class LLM:
    """Juez opcional: decide qué lado afirma la narrativa y la nombra. Una llamada por narrativa."""

    REINTENTOS = 6                  # por llamada, con espera creciente (429, 5xx, 529, red)
    CORTE_SEGUIDOS = 4              # llamadas seguidas que fallan del todo -> se deja de usar la IA en esta corrida
    TRANSITORIOS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
    PRESUPUESTO_S = 300             # tiempo total perdido en fallos antes de seguir sin IA (la corrida no se cuelga)

    def __init__(self, log=print):
        self.key = os.environ["ANTHROPIC_API_KEY"]
        self.modelo = os.environ.get("MURMUR_LLM_MODELO", "claude-haiku-5-5")
        self.usd_in = float(os.environ.get("MURMUR_USD_MTOK_IN", "0.10"))
        self.usd_out = float(os.environ.get("MURMUR_USD_MTOK_OUT", "0.50"))
        self.log = log
        self.fallos = 0             # llamadas que terminaron sin respuesta usable
        self.seguidos = 0
        self.perdido = 0.0          # segundos gastados en intentos fallidos
        self.abortado = ""          # motivo si se dejó de usar la IA
        self.lock = threading.Lock()  # las llamadas van en paralelo: los contadores se actualizan de a una
        self.cortes = defaultdict(int)  # diagnóstico: stop_reason de cada respuesta (solo se informa, no decide nada)
        self.pensadas = 0               # respuestas que trajeron bloques de razonamiento (gastan max_tokens)

    def _cortar(self, motivo):
        if not self.abortado:
            self.abortado = motivo
            self.log(f"  AVISO IA: {motivo}; sigo sin IA (postura heurística)")

    def _post(self, body):
        """POST a la API con reintentos. Devuelve el JSON o None (y cuenta el fallo)."""
        espera = 2.0
        for intento in range(self.REINTENTOS):
            t0 = time.time()
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
                "x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            try:
                return json.loads(urllib.request.urlopen(req, timeout=60).read())
            except urllib.error.HTTPError as e:
                try:
                    detalle = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    detalle = ""
                if e.code in (401, 403):
                    self._cortar(f"clave rechazada (HTTP {e.code})")
                    return None
                if e.code == 404:
                    self._cortar(f"modelo {self.modelo} no encontrado (HTTP 404)")
                    return None
                if e.code == 400 and re.search(r"credit|billing|limit|spend", detalle, re.I):
                    self._cortar("sin saldo o tope de gasto alcanzado (HTTP 400); esperar no lo arregla")
                    return None
                if e.code not in self.TRANSITORIOS:
                    self.log(f"  IA: HTTP {e.code} {detalle[:120]}")
                    return None
                ra = e.headers.get("retry-after") if e.headers else None
                pausa = float(ra) if ra and ra.replace(".", "", 1).isdigit() else espera
                motivo = f"HTTP {e.code}"
            except Exception as e:  # timeout, conexión cortada, DNS
                pausa, motivo = espera, type(e).__name__
            with self.lock:
                self.perdido += time.time() - t0
            if self.perdido >= self.PRESUPUESTO_S:
                self._cortar(f"{self.perdido:.0f}s perdidos en fallos de la API ({motivo})")
                return None
            if intento + 1 < self.REINTENTOS:
                pausa = min(pausa, 60.0, self.PRESUPUESTO_S - self.perdido) * (0.75 + 0.5 * random.random())
                with self.lock:
                    self.perdido += pausa
                self.log(f"  IA: {motivo}, reintento {intento + 1}/{self.REINTENTOS - 1} en {pausa:.0f}s")
                time.sleep(pausa)
                espera = min(espera * 2, 30.0)
        self.log(f"  IA: {motivo} tras {self.REINTENTOS} intentos")
        return None

    def juzgar(self, A, B, uso):
        prompt = ("En una red social, una afirmación circula sobre un tema local. Abajo hay comentarios de dos grupos que "
                  "discuten esa misma afirmación: un grupo la sostiene y el otro le contesta para desmentirla, relativizarla "
                  "o darla vuelta.\n\nGrupo A:\n- " + "\n- ".join(A) + "\n\nGrupo B:\n- " + "\n- ".join(B) +
                  "\n\n¿Qué grupo sostiene la afirmación original (la idea que se propaga), y cuál es la respuesta? Pistas: la "
                  "respuesta suele reaccionar a lo que dice el otro grupo (niega un hecho, dice que es exagerado, que es falso, "
                  "que la causa es otra, que lo prometido no se cumplió o que la propuesta no sirve). La afirmación original es "
                  "la idea que tiene sentido por sí sola sin necesitar al otro grupo.\n\n"
                  'Respondé solo JSON: {"lado": "A" o "B" (el que sostiene la afirmación original), '
                  '"titulo": "la afirmación original en una frase corta", "frases_clave": ["3 a 6 términos"]}')
        if self.abortado:
            return None
        body = json.dumps({"model": self.modelo, "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}]}).encode()
        r = self._post(body)
        if r is None:
            with self.lock:
                self.fallos += 1
                self.seguidos += 1
                corta = self.seguidos >= self.CORTE_SEGUIDOS
            if corta:
                self._cortar(f"{self.seguidos} llamadas seguidas sin respuesta")
            return None
        with self.lock:
            self.seguidos = 0
        if not isinstance(r, dict):
            with self.lock:
                self.fallos += 1
            return None
        try:
            u = r.get("usage") or {}
            ti, to = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        except Exception:
            ti = to = 0
        with self.lock:
            self.cortes[str(r.get("stop_reason"))] += 1
            if any(isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") for b in r.get("content") or []):
                self.pensadas += 1
            uso["tokens"] += ti + to; uso["llamadas"] += 1
            uso["usd"] += ti / 1e6 * self.usd_in + to / 1e6 * self.usd_out
        try:
            txt = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
            j = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
            if isinstance(j, dict) and j.get("lado") in ("A", "B"):
                return j
        except Exception:
            pass
        with self.lock:
            self.fallos += 1
        return None
