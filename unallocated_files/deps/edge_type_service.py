#!/usr/bin/env python3
"""
deps/edge_type_service.py
─────────────────────────
Two-pass resolver  (Dict-A ➜ Dict-B/LLM)  **with logging**.

Returns
-------
abstract  : str    # chosen abstract edge type
created   : bool   # True if a NEW mapping was added
source    : str    # 'A' (seed/persistent)  or  'B' (LLM decision)

If you pass a `trace: list[str]`, short log lines are appended.
"""
from __future__ import annotations
import json, threading
from pathlib import Path
from typing import Dict, List, Tuple

from deps.deepinfra_client import client

HERE = Path(__file__).resolve().parent
RES  = HERE.parent.parent / "resources";  RES.mkdir(exist_ok=True)
DICT_PATH = RES / "edge_types.json"
_LOCK = threading.Lock()

# ---------- 0) static seed rules -------------------------------------------
_SEED_MAP = {
    "located":      "located_in",
    "located_in":   "located_in",
    "founded":      "founded_by",
    "founded_by":   "founded_by",
    "acquired":     "acquired_by",
    "acquired_by":  "acquired_by",
}

# ---------- helpers ---------------------------------------------------------
def _load_dict() -> Dict[str, str]:
    return json.loads(DICT_PATH.read_text()) if DICT_PATH.exists() else {}

def _save_dict(d: Dict[str, str]):
    DICT_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False))

_EDGE_DICT: Dict[str, str] = _load_dict()

_PROMPT = (
    "Abstract edge types so far:\n{types}\n\n"
    "Predicate: \"{pred}\"\n"
    "Reply with either an existing type *verbatim*, or\n"
    "NEW: <short_name> if a new abstract type is needed."
)

# ---------- public ----------------------------------------------------------
def resolve_predicate(pred: str,
                      abstract_pool: List[str],
                      trace: List[str] | None = None
                      ) -> Tuple[str, bool, str]:
    """
    Two-pass: Dict-A (seed+persist) else Dict-B (LLM).
    """
    p = pred.lower()
    if trace is not None: trace.append(f"▶ resolving '{pred}'")

    # A-1 seed
    if p in _SEED_MAP:
        if trace is not None: trace.append(f"Dict-A hit (seed) → {_SEED_MAP[p]}")
        return _SEED_MAP[p], False, "A"

    # A-2 persistent
    if p in _EDGE_DICT:
        if trace is not None: trace.append(f"Dict-A hit (persist) → {_EDGE_DICT[p]}")
        return _EDGE_DICT[p], False, "A"

    # B) ask LLM
    try:
        msg = _PROMPT.format(types="\n".join(abstract_pool) or "(none yet)", pred=pred)
        resp = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[{"role": "user", "content": msg}],
            temperature=0.0,
            max_tokens=16,
        )
        ans = resp.choices[0].message.content.strip()
    except Exception:
        ans = "NEW: " + pred        # offline fallback

    if ans.lower().startswith("new:"):
        abstract = ans[4:].strip()
        created  = True
        if trace is not None: trace.append(f"Dict-B NEW → {abstract}")
    else:
        abstract = ans
        created  = False
        if trace is not None: trace.append(f"Dict-B mapped → {abstract}")

    # persist
    with _LOCK:
        _EDGE_DICT[p] = abstract
        _save_dict(_EDGE_DICT)

    return abstract, created, "B"
