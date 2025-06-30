#!/usr/bin/env python3
"""
tuple_extractor_rule_coref.py

Heuristic pronoun resolution + tuple extraction.

Install:
    pip install spacy
    python -m spacy download en_core_web_sm

Usage:
    python tuple_extractor_rule_coref.py "Alice saw her dog. She then walked it home."
"""

import sys
import spacy

nlp = spacy.load("en_core_web_sm")

# Pronouns to resolve (lowercased)
PRONOUNS = {"he","she","it","they","him","her","them"}

# Maps
SPO_MAP = {"nsubj": "Subject", "dobj": "Object", "iobj": "IndirectObject"}
MOD_DEPS = {"amod","compound"}
ADV_DEP = "advmod"
TENSE_MAP = {"VBD":"past","VBP":"present","VBZ":"present",
             "VBG":"present-participle","VBN":"past-participle","VB":"base"}

def resolve_pronouns(text: str):
    doc = nlp(text)
    resolved_tokens = []
    last_np = None
    # Build resolved text token by token
    for token in doc:
        lower = token.text.lower()
        if lower in PRONOUNS and last_np:
            # replace pronoun with last noun-phrase
            resolved_tokens.append(last_np)
        else:
            resolved_tokens.append(token.text)
        # update last_np whenever we see a noun-chunk head
        if token.dep_ in {"nsubj","dobj","pobj","iobj","appos","compound"} and token.pos_ in {"NOUN","PROPN"}:
            # capture the whole chunk
            span = token.text
            # extend to include compound/adjectival modifiers
            for child in token.children:
                if child.dep_ in {"compound","amod"}:
                    span = child.text + " " + span
            last_np = span
    return " ".join(resolved_tokens)

def extract_nested_tuples(text: str):
    doc = nlp(text)
    tuples = []
    # Predicate + tense
    root = next((t for t in doc if t.dep_=="ROOT" and t.pos_=="VERB"), None)
    if root:
        attrs = [c.text for c in root.children if c.dep_==ADV_DEP]
        entry = {"role":"Predicate","filler":root.lemma_,"attributes":attrs}
        if root.tag_ in TENSE_MAP:
            entry["tense"] = TENSE_MAP[root.tag_]
        tuples.append(entry)
    # SPO + modifiers
    for token in doc:
        role = SPO_MAP.get(token.dep_)
        if role:
            attrs = [c.text for c in token.children if c.dep_ in MOD_DEPS]
            tuples.append({"role":role,"filler":token.text,"attributes":attrs})
    return tuples

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python tuple_extractor_rule_coref.py \"Your text here.\"")
        sys.exit(1)
    article = sys.argv[1]
    resolved = resolve_pronouns(article)
    result = extract_nested_tuples(resolved)
    print("\nResolved & extracted tuples:\n")
    for r in result:
        print(r)
