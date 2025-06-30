#!/usr/bin/env python3
"""
Tiny hierarchical graph store:
   Document ➜ Sentences ➜ Edges     (surface & semantic HVs)
"""
from collections import defaultdict

class GraphStore:
    def __init__(self):
        self.docs = defaultdict(lambda: {"sentences": {}, "edges": []})

    # ----------------------------------------------------------
    def add_sentence(self, doc_id: str, sent_id: int,
                     sentence: str,
                     extract_fn, *, verbose=False):
        edges = extract_fn(sentence, doc_id, sent_id, verbose=verbose)
        self.docs[doc_id]["sentences"][sent_id] = {
            "text": sentence,
            "edges": edges
        }
        self.docs[doc_id]["edges"].extend(edges)

    # ----------------------------------------------------------
    def get_sentence_graph(self, doc_id: str, sent_id: int):
        return self.docs[doc_id]["sentences"][sent_id]["edges"]

    def get_doc_graph(self, doc_id: str):
        return self.docs[doc_id]["edges"]
