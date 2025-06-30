#!/usr/bin/env python3
"""
event_tuple_extractor.py

Deterministically extract events (Predicate + SPO roles + attributes)
from text, including subordinate clauses (xcomp, ccomp):

1. Heuristic pronoun resolution.
2. For each verb token:
   - Extract predicate lemma, tense, advmod attributes.
   - Extract Subject(s) and Object(s) (nsubj, nsubjpass, dobj, iobj) with amod/compound attrs.
3. Associate attributes with the correct filler via nested structure.

Install:
    pip install spacy
    python -m spacy download en_core_web_sm

Usage:
    python event_tuple_extractor.py \
      "I saw a white dog chase the brown cat quickly in the backyard."
"""

import sys
import spacy

nlp = spacy.load("en_core_web_sm")

# Pronouns for heuristic resolution
PRONOUNS = {"he","she","it","they","him","her","them"}

# Mapping UD deps → roles
SPO_MAP = {
    "nsubj":       "Subject",
    "nsubjpass":   "Subject",
    "dobj":        "Object",
    "iobj":        "IndirectObject"
}
ATTR_DEPS = {"amod", "compound"}    # modifiers on nouns
ADV_DEP    = "advmod"               # adverbial modifiers
TENSE_MAP = {
    "VBD": "past",
    "VBP": "present",
    "VBZ": "present",
    "VBG": "present-participle",
    "VBN": "past-participle",
    "VB":  "base"
}

def resolve_pronouns(text: str) -> str:
    """Replace pronouns with the most recent noun phrase."""
    doc = nlp(text)
    tokens, last_np = [], None
    for tok in doc:
        low = tok.text.lower()
        if low in PRONOUNS and last_np:
            tokens.append(last_np)
        else:
            tokens.append(tok.text)
        # update last_np when encountering a noun phrase head
        if tok.dep_ in {"nsubj","dobj","pobj","iobj","appos"} and tok.pos_ in {"NOUN","PROPN"}:
            span = " ".join(w.text for w in tok.subtree)
            last_np = span
    return " ".join(tokens)

def extract_events(text: str):
    """Extract a list of nested event dicts from resolved text."""
    doc = nlp(text)
    events = []
    for tok in doc:
        if tok.pos_ == "VERB":
            # 1. Predicate entry
            evt = {
                "role": "Predicate",
                "filler": tok.lemma_.lower(),
                "attributes": [c.text.lower() for c in tok.children if c.dep_ == ADV_DEP]
            }
            if tok.tag_ in TENSE_MAP:
                evt["tense"] = TENSE_MAP[tok.tag_]
            events.append(evt)

            # 2. SPO roles for this verb
            for child in tok.children:
                if child.dep_ in SPO_MAP:
                    role = SPO_MAP[child.dep_]
                    # collect noun modifiers
                    attrs = [gc.text.lower() for gc in child.children if gc.dep_ in ATTR_DEPS]
                    events.append({
                        "role": role,
                        "filler": child.text.lower(),
                        "attributes": attrs
                    })
    return events

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python event_tuple_extractor.py \"Your text here.\"")
        sys.exit(1)
    sentence = sys.argv[1]
    resolved = resolve_pronouns(sentence)
    evts = extract_events(resolved)
    print("\nResolved & extracted event tuples:\n")
    for e in evts:
        print(e)
