"""Embeddings locales de textos únicos, con caché en Parquet por hash de texto.

Los comentarios se repiten mucho (plantillas, copias, respuestas cortas), así que
se embebe cada texto normalizado una sola vez. La caché hace que una segunda
corrida, o un stream incremental, solo pague por los textos nuevos.
"""
import hashlib, os, re, time
from pathlib import Path
import numpy as np
import pandas as pd

MODELO = os.environ.get("MURMUR_MODELO", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def normalizar(s) -> str:
    s = s if isinstance(s, str) else ""
    s = re.sub(r"#\w+", "", s)
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch in ",.:;?!¿¡")
    return re.sub(r"\s+", " ", s).strip().lower()


def _h(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=10).hexdigest()


def embeber(textos: pd.Series, cache_dir: Path, log=print):
    """Devuelve (E float32 [n_unicos, d], kid int por texto)."""
    claves = textos.map(normalizar)
    unicos = pd.Index(claves.unique())
    kid = unicos.get_indexer(claves)
    hashes = np.array([_h(u) for u in unicos])

    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / "embeddings.parquet"
    cache = {}
    if cpath.exists():
        cdf = pd.read_parquet(cpath)
        cache = dict(zip(cdf.h, cdf.v))
    falta = [i for i, h in enumerate(hashes) if h not in cache]
    stats = {"textos_unicos": len(unicos), "embebidos_ahora": len(falta), "segundos_embeddings": 0.0}
    if falta:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        t = time.time()
        m = SentenceTransformer(MODELO, device="cpu")
        V = m.encode([unicos[i] for i in falta], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        stats["segundos_embeddings"] = round(time.time() - t, 1)
        for i, v in zip(falta, V.astype(np.float16)):
            cache[hashes[i]] = v
        pd.DataFrame({"h": list(cache.keys()), "v": list(cache.values())}).to_parquet(cpath)
        log(f"  embebidos {len(falta)} textos nuevos en {stats['segundos_embeddings']}s")
    else:
        log("  embeddings: todo en caché")
    E = np.stack([np.asarray(cache[h], dtype=np.float32) for h in hashes])
    return E, kid, unicos, stats
