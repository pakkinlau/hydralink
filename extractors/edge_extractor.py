#!/usr/bin/env python3
"""
Layer-2  —  Sentence-graph extractor with:
  • AliasResolving  (T0/T1/T2)
  • Passive→Active rewrite
  • Predicate→Abstract mapping (Dict-A/Dict-B)   + logging
  • extract_sentence_graph()   – returns list[edge] for ALL tuples
"""
from __future__ import annotations
import re, json, collections
from pathlib import Path
from typing   import Dict, Any, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from deps.alias_service      import AliasResolver
from deps.edge_type_service  import resolve_predicate
from extractors.triple_extractor import Triple, extract_triples

# ───────── user knob + counters
EXPECTED_ABSTRACTS: List[str] = []          # set in notebook
EDGE_COUNTS         = collections.Counter()
ALIAS_TIER_COUNTS   = collections.Counter()

# ───────── paths + resources
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RES  = HERE.parent.parent / "resources";  RES.mkdir(exist_ok=True)

# ───────── alias resolver
_resolver = AliasResolver()

def _alias(text: str, trace: List[str]) -> Tuple[str, dict]:
    meta = {"eid": None, "alias_tier": -1}
    out  = []
    for tok in re.findall(r"\w+|\W+", text):
        canon, eid, tier = _resolver.resolve(tok)
        out.append(canon)
        if eid and not meta["eid"]:
            meta["eid"] = eid
        if tier >= 0:
            meta["alias_tier"] = tier
            ALIAS_TIER_COUNTS[tier] += 1
    trace.append(f"Alias tier={meta['alias_tier']}")
    return "".join(out), meta

# ───────── passive→active helper
_PASSIVE = re.compile(
    r"^(?P<obj>.+?)\s+was\s+(?P<verb>founded|acquired|located)"
    r"\s+by\s+(?P<subj>.+?)\.*$", re.I)

def _to_active(sentence: str, trace: List[str]) -> Tuple[str, bool]:
    m = _PASSIVE.match(sentence.strip())
    if not m:
        return sentence, False
    subj, verb, obj = m["subj"].strip(), m["verb"].lower(), m["obj"].strip()
    trace.append("Passive→active rewrite")
    return f"{subj} {verb} {obj}.", True

# ───────── MiniLM & projection
MINILM = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
proj_path = RES / "A.npy"
if proj_path.exists():
    PROJ = np.load(proj_path)
else:
    rng  = np.random.default_rng(42)
    PROJ = rng.standard_normal((4096,384)).astype(np.float32)
    PROJ /= np.linalg.norm(PROJ,axis=1,keepdims=True)+1e-6
    np.save(proj_path, PROJ)

def _emb(tok: str) -> np.ndarray:
    return np.sign(PROJ @ MINILM.encode(tok, normalize_embeddings=True)).astype(np.int8)

# ───────── HD ops
D = 4096
np.random.seed(42)
RS, RP, RO = [np.random.choice([-1,1],D).astype(np.int8) for _ in range(3)]
_bind   = lambda r,f: r*f
_bundle = lambda *v: np.sum(v,axis=0).astype(np.int8)
_perm   = lambda v,tag: np.roll(v,1) if tag=="S" else np.flip(v)
_surface  = lambda s,p,o: _bundle(_bind(RS,_emb(s)), _bind(RP,_emb(p)), _bind(RO,_emb(o)))
_semantic = lambda s,p,o: _bind(_emb(p), _bundle(_perm(_emb(s),"S"), _perm(_emb(o),"O")))

# ───────── edge-from-triple helper (internal)
def _edge_from_triple(sentence: str,
                      triple: Triple,
                      trace: List[str]) -> Dict[str, Any] | None:
    # predicate is already str after schema change
    subj       = triple.subject
    fine_pred  = triple.predicate.lower()        # normalise
    obj        = triple.object

    abstract, _, source = resolve_predicate(fine_pred, EXPECTED_ABSTRACTS, trace)
    trace.append(f"Abstract ({'Dict-'+source}) = {abstract}")

    if EXPECTED_ABSTRACTS and abstract not in EXPECTED_ABSTRACTS:
        trace.append("-- skipped (abstract not in filter)")
        return None
    EDGE_COUNTS[abstract] += 1

    return {
        "edge_type": abstract,
        "surface":   _surface(subj, fine_pred, obj),
        "semantic":  _semantic(subj, fine_pred, obj),
        "meta":      {"fine_pred": fine_pred, "abstract": abstract}
    }

# ───────── public: ALL edges in one sentence
def extract_sentence_graph(sentence: str,
                           doc_id: str,
                           sent_id: int,
                           *,
                           verbose=False) -> List[Dict[str,Any]]:
    trace: List[str] = []
    sent_act, _ = _to_active(sentence, trace)
    sent_norm, alias_meta = _alias(sent_act, trace)

    triples = extract_triples(sent_norm)
    edges: List[Dict[str,Any]] = []

    for t in triples:
        edge = _edge_from_triple(sentence, t, trace)
        if edge:
            edge["meta"].update(alias_meta | {"doc_id": doc_id, "sent_id": sent_id})
            edges.append(edge)

    if verbose:
        print(f"\nSentence '{sentence}'")
        print("\n".join(trace))
        print(f"-- produced {len(edges)} edge(s)\n")
    return edges
