#!/usr/bin/env python3
"""
deps/alias_service.py
─────────────────────
Scalable Alias & Entity-Alignment with three cascading tiers:

  T0  exact dictionary  (O(1) RAM, sharded TSVs)
  T1  in-document fuzzy (MiniLM cosine on a tiny local index)
  T2  global entity linker (KB ANN + context re-rank - optional)

If a heavy dependency (faiss, sentence-transformers) or KB data is
missing, the corresponding tier is silently skipped.
"""
from __future__ import annotations
import re, hashlib, warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

# ───────────────────────────────────────────────────────────────
# Optional heavy deps
# ───────────────────────────────────────────────────────────────
try:
    import faiss
except ImportError:
    faiss = None                                      # Tier-2 disabled
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None                        # Tier-1 disabled

HERE = Path(__file__).resolve().parent
RES  = HERE.parent.parent / "resources"
RES.mkdir(exist_ok=True)

# ───────────────────────────────────────────────────────────────
# Tier-0  ▸  exact dictionary  (256 shards)
# ───────────────────────────────────────────────────────────────
@lru_cache(maxsize=256)
def _load_shard(prefix: str) -> Dict[str, str]:
    shard = RES / "aliases" / f"{prefix}.tsv"
    if not shard.exists():
        return {}
    return {k.lower(): v for k, v in
            (ln.split("\t", 1) for ln in shard.read_text(encoding="utf8").splitlines())}

def _exact_lookup(token: str) -> Optional[str]:
    prefix = hashlib.sha1(token.encode()).hexdigest()[:2]
    return _load_shard(prefix).get(token.lower())

# ───────────────────────────────────────────────────────────────
# Tier-1  ▸  in-document fuzzy  (MiniLM cosine)
# ───────────────────────────────────────────────────────────────
_MINILM = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") if SentenceTransformer else None
_FUZZY_THRESH = 0.85

def _fuzzy_lookup(token: str, doc) -> Optional[str]:
    if not (_MINILM and doc and doc.vecs.size):
        return None
    q = _MINILM.encode(token, normalize_embeddings=True)
    sims = doc.vecs @ q
    idx  = sims.argmax()
    if sims[idx] > _FUZZY_THRESH:
        return doc.spans[idx]
    return None

class _DocIndex:
    """Tiny per-document in-memory HNSW surrogate."""
    def __init__(self, text: str):
        import numpy as np
        spans = re.findall(r"\b([A-Z][\w\-]{2,}(?:\s+[A-Z][\w\-]{2,}){0,4})", text)
        spans = list(dict.fromkeys(spans))
        self.spans = spans
        self.vecs  = _MINILM.encode(spans, normalize_embeddings=True) if spans else np.empty((0, 384))

# ───────────────────────────────────────────────────────────────
# Tier-2  ▸  global KB entity linker  (optional)
# ───────────────────────────────────────────────────────────────
_KB = _KB_EMB = _KB_IDS = None
def _lazy_load_kb():
    global _KB, _KB_EMB, _KB_IDS
    if _KB or not faiss:
        return
    try:
        import numpy as np
        _KB_EMB = np.load(RES / "kb_emb.npy")          # (N, 384)
        _KB_IDS = (RES / "kb_ids.txt").read_text().splitlines()
        _KB = faiss.IndexFlatIP(_KB_EMB.shape[1])
        _KB.add(_KB_EMB)
    except Exception as e:
        warnings.warn(f"[alias_service] KB load failed: {e}; Tier-2 disabled")

def _link_global(token: str) -> Tuple[Optional[str], Optional[str]]:
    _lazy_load_kb()
    if not (_KB and _MINILM):
        return None, None
    q = _MINILM.encode(token, normalize_embeddings=True).reshape(1, -1)
    sim, idx = _KB.search(q, 1)
    if sim[0, 0] < 0.6:
        return None, None
    return _KB_IDS[idx[0, 0]], token

# ───────────────────────────────────────────────────────────────
# Public resolver
# ───────────────────────────────────────────────────────────────
class AliasResolver:
    """
    Stateless resolver; build once per document/batch.
    Methods
    -------
    resolve(token:str) -> (canonical:str, eid:str|None, tier:int)
    """
    def __init__(self, doc_text: str | None = None):
        self.doc = _DocIndex(doc_text) if (doc_text and _MINILM) else None

    # ------------------------------------------------------------------
    def resolve(self, token: str) -> Tuple[str, Optional[str], int]:
        # T0 exact
        hit = _exact_lookup(token)
        if hit:
            return hit, None, 0
        # T1 fuzzy
        hit = _fuzzy_lookup(token, self.doc)
        if hit:
            return hit, None, 1
        # T2 KB
        eid, canon = _link_global(token)
        if eid:
            return canon, eid, 2
        # fallback
        return token, None, -1
