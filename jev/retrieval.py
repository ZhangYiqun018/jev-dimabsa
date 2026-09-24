"""BM25 retrieval of same-corpus train reviews with their annotated phrases.

Used to show Jev the dataset's boundary conventions (jev/checks.py). Train
records whose text also occurs in dev or test are never retrieved.
"""
from __future__ import annotations

import math
from collections import Counter

from .data import _annotation_items
from .fewshot import normalise

EXAMPLES = 4


def bigrams(text):
    text = ''.join(text.lower().split())
    return [text[i:i + 2] for i in range(len(text) - 1)] or ([text] if text else [])


class Retriever:
    """BM25 (k1 1.5, b 0.75) over character bigrams of same-corpus train reviews."""

    def __init__(self, rows, banned_texts=()):
        seen = {normalise(t) for t in banned_texts}
        self.rows = []
        for row in rows:
            key = normalise(row['Text'])
            if key not in seen and _annotation_items(row):
                self.rows.append(row)
                seen.add(key)
        self.tf = [Counter(bigrams(r['Text'])) for r in self.rows]
        self.lengths = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.lengths) / max(1, len(self.lengths)) or 1
        self.df = Counter(t for counts in self.tf for t in counts)
        self.index = {}
        for i, counts in enumerate(self.tf):
            for term in counts:
                self.index.setdefault(term, []).append(i)

    def select(self, text, n=EXAMPLES):
        scores = Counter()
        for term in set(bigrams(text)):
            idf = math.log(1 + (len(self.rows) - self.df[term] + .5) / (self.df[term] + .5))
            for i in self.index.get(term, ()):
                freq = self.tf[i][term]
                scores[i] += idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * self.lengths[i] / self.avg))
        ranked = sorted(scores, key=lambda i: (-scores[i], self.rows[i]['ID']))
        return [self.rows[i] for i in ranked[:n]]


def example(row):
    items = _annotation_items(row)
    return {'review': row['Text'],
            'aspects': list(dict.fromkeys(x['Aspect'] for x in items)),
            'opinions': list(dict.fromkeys(x['Opinion'] for x in items))}
