#!/usr/bin/env python3
"""
enhanced_event_extractor.py

Deterministically extract all event tuples (Predicate + SPO + attributes)
including subordinate clauses (xcomp, ccomp) from input text.

1. Heuristic pronoun resolution.
2. Identify all VERB tokens in the document.
3. For each verb token:
   - Extract predicate lemma, tense, advmod attributes.
   - Find Subject(s) (nsubj, nsubjpass) and Object(s) (dobj, iobj, pobj)
     from its children.
   - Capture amod/compound modifiers on each filler.
   
Install:
    pip install spacy
    python -m spacy download en_core_web_sm

Usage:
    python enhanced_event_extractor.py \
      "I saw a white dog chase the brown cat quickly in the backyard."
"""

import sys
import spacy

nlp = spacy.load("en_core_web_sm")

# Pronoun resolution heuristic
PRONOUNS = {"he","she","it","they","him","her","them"}
# Dependencies mapping
SUBJ_DEPS = {"nsubj", "nsubjpass"}
OBJ_DEPS  = {"dobj", "iobj", "pobj"}
ATTR_DEPS = {"amod", "compound"}
ADV_DEP   = "advmod"
# POS tag → tense
TENSE_MAP = {
    "VBD":"past","VBP":"present","VBZ":"present",
    "VBG":"present-participle","VBN":"past-participle","VB":"base"
}

def resolve_pronouns(text: str) -> str:
    """Replace pronouns with the nearest preceding noun phrase."""
    doc = nlp(text)
    tokens, last_np = [], None
    for tok in doc:
        low = tok.text.lower()
        if low in PRONOUNS and last_np:
            tokens.append(last_np)
        else:
            tokens.append(tok.text)
        # update last_np on NP heads (noun or proper noun)
        if tok.dep_ in SUBJ_DEPS|OBJ_DEPS|{"appos"} and tok.pos_ in {"NOUN","PROPN"}:
            span = " ".join(w.text for w in tok.subtree)
            last_np = span
    return " ".join(tokens)

def extract_events(text: str):
    """Extract events from resolved text."""
    doc = nlp(text)
    events = []
    seen_verbs = set()
    for tok in doc:
        if tok.pos_ == "VERB" and tok.i not in seen_verbs:
            seen_verbs.add(tok.i)
            # 1. Predicate
            attrs = [c.text.lower() for c in tok.children if c.dep_ == ADV_DEP]
            evt = {
                "role": "Predicate",
                "filler": tok.lemma_.lower(),
                "attributes": attrs
            }
            if tok.tag_ in TENSE_MAP:
                evt["tense"] = TENSE_MAP[tok.tag_]
            events.append(evt)
            # 2. Subjects
            for child in tok.children:
                if child.dep_ in SUBJ_DEPS:
                    mods = [gc.text.lower() for gc in child.children if gc.dep_ in ATTR_DEPS]
                    events.append({
                        "role": "Subject",
                        "filler": child.text.lower(),
                        "attributes": mods
                    })
            # 3. Objects
            for child in tok.children:
                if child.dep_ in OBJ_DEPS:
                    mods = [gc.text.lower() for gc in child.children if gc.dep_ in ATTR_DEPS]
                    events.append({
                        "role": "Object",
                        "filler": child.text.lower(),
                        "attributes": mods
                    })
    return events

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python enhanced_event_extractor.py \"Your text here.\"")
        sys.exit(1)
    text = sys.argv[1]
    resolved = resolve_pronouns(text)
    events = extract_events(resolved)
    print("\nResolved & extracted event tuples:\n")
    for e in events:
        print(e)
