#!/usr/bin/env python3
"""
extractors/triple_extractor.py
──────────────────────────────
• extract_triples(sentence) -> list[Triple]
• extract_triple(sentence)  -> first Triple (legacy)
"""

from __future__ import annotations
import json, time
from enum       import Enum
from typing     import List
from pydantic   import BaseModel, constr, ValidationError
from deps.deepinfra_client import client   # DeepInfra token + base_url

# ─── schema ──────────────────────────────────────────────────────
class Predicate(str, Enum):
    located_in  = "located_in"
    founded_by  = "founded_by"
    acquired_by = "acquired_by"

class Triple(BaseModel):
    subject:   constr(strip_whitespace=True, min_length=1)
    predicate: constr(strip_whitespace=True, min_length=1)
    object:    constr(strip_whitespace=True, min_length=1)

# ─── plural-tuple prompt  (all braces escaped) ───────────────────
PROMPT_TMPL = (
    "Extract *all* (subject, predicate, object) tuples from the sentence as "
    "strict JSON **array** with this exact schema:\n"
    "[{{\"subject\": string, \"predicate\": string, \"object\": string}}, ...]\n"
    "Return JSON array only—no commentary.\n\n{sent}"
)

# ─── extractor ───────────────────────────────────────────────────
# ─── robust plural extractor ───────────────────────────────────────────────
def extract_triples(sentence: str, *, max_tries: int = 3) -> List[Triple]:
    """
    Returns list[Triple].  Handles:
      • correct JSON array
      • array-as-string  "[{...}]"
      • single JSON dict  {...}
    Silently skips malformed items.
    """
    for attempt in range(1, max_tries + 1):
        resp = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[{
                "role": "user",
                "content": PROMPT_TMPL.format(sent=sentence)
            }],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )

        raw = resp.choices[0].message.content
        try:
            data = json.loads(raw)
            # Case: the whole array came back as a string
            if isinstance(data, str):
                data = json.loads(data)

            # Normalise to list
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                raise ValueError("LLM returned non-list JSON")

            triples: List[Triple] = []
            for item in data:
                if isinstance(item, dict):
                    try:
                        triples.append(Triple.model_validate(item))
                    except ValidationError:
                        continue   # skip bad item
            return triples

        except Exception as e:
            if attempt == max_tries:
                raise
            time.sleep(1.0)


# legacy single-tuple helper
def extract_triple(sentence: str) -> Triple:
    triples = extract_triples(sentence)
    if not triples:
        raise ValueError("No tuple extracted")
    return triples[0]
