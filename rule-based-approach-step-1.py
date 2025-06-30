#!/usr/bin/env python3
"""
tuple_extractor_spacy.py

A lightweight, deterministic pipeline converting a sentence into
(role, filler) tuples using only spaCy's dependency parse.

Install:
    pip install spacy
    python -m spacy download en_core_web_sm

Usage:
    python tuple_extractor_spacy.py "Alice quickly writes Python code in the morning."
"""

import sys
import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Mapping from UD dependency labels to roles
DEP_ROLE_MAP = {
    "nsubj":       "Subject",
    "dobj":        "Object",
    "iobj":        "IndirectObject",
    "advmod":      "Adverb",
    "amod":        "Attribute",
    "nmod:tmod":   "TimeOfDay",
    "prep":        "LocationPrep",
    "pobj":        "LocationObj",
    "compound":    "Compound",
    "appos":       "Apposition"
}

# POS tag → tense
TENSE_MAP = {
    "VBD": "past",
    "VBP": "present",
    "VBZ": "present",
    "VBG": "present-participle",
    "VBN": "past-participle",
    "VB":  "base"
}

def extract_tuples(sentence: str):
    """
    Runs a deterministic spaCy-based pipeline:
      1. Parse sentence to Doc
      2. Identify main predicate (ROOT or first verb)
      3. Map dependencies around verbs to roles
      4. Infer Tense from POS tags
      5. Normalize spans
    Returns a set of (role, filler) tuples.
    """
    doc = nlp(sentence)
    tuples = set()

    # 1. Identify main verb (predicate)
    root_verb = next((t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)
    if not root_verb:
        root_verb = next((t for t in doc if t.pos_ == "VERB"), None)
    if root_verb:
        tuples.add(("Predicate", root_verb.lemma_))

    # 2. Dependency-based roles
    for token in doc:
        role = DEP_ROLE_MAP.get(token.dep_)
        if role:
            span = " ".join(w.text for w in token.subtree)
            tuples.add((role, span))

    # 3. Tense inference
    if root_verb and root_verb.tag_ in TENSE_MAP:
        tuples.add(("Tense", TENSE_MAP[root_verb.tag_]))

    # 4. Normalize: lowercase, trim whitespace
    clean = set((r, " ".join(s.split()).lower()) for r, s in tuples)
    return clean

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tuple_extractor_spacy.py \"Your sentence here.\"")
        sys.exit(1)

    sentence = sys.argv[1]
    result = extract_tuples(sentence)
    print("\nExtracted (role → filler) tuples:\n")
    for role, span in sorted(result):
        print(f"{role:15} → {span}")
