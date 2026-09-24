"""BM25 retrieval variants inspired by Takoyaki; independent of BIO tokenization."""
from collections import Counter
import math

from .fewshot import normalise


def retrieval_terms(text, corpus, variant):
    text = text.lower()
    if variant in ('bigram', 'trigram'):
        text = ''.join(text.split())
        width = 2 if variant == 'bigram' else 3
        return [text[i:i + width] for i in range(len(text) - width + 1)] or ([text] if text else [])
    if variant != 'word':
        raise ValueError(variant)
    if corpus.startswith('zho'):
        import jieba
        return [t for t in jieba.cut(text, cut_all=False, HMM=False) if t.strip()]
    if corpus.startswith('jpn'):
        raise ValueError('Japanese word retrieval is outside this trial experiment')
    return text.split()


class Retriever:
    def __init__(self, rows, corpus, banned_texts=(), variant='bigram'):
        self.corpus, self.variant = corpus, variant
        seen = {normalise(s) for s in banned_texts}
        self.rows = []
        for row in rows:
            key = normalise(row['Text'])
            if key not in seen:
                self.rows.append(row)
                seen.add(key)
        self.tf = [Counter(retrieval_terms(r['Text'], corpus, variant)) for r in self.rows]
        self.lengths = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.lengths) / max(1, len(self.lengths)) or 1
        self.df = Counter(t for counts in self.tf for t in counts)

    def select(self, text, n=50):
        terms = set(retrieval_terms(text, self.corpus, self.variant))
        scores = []
        for i, counts in enumerate(self.tf):
            if normalise(self.rows[i]['Text']) == normalise(text):
                continue
            score = 0.
            for term in sorted(terms & counts.keys()):
                idf = math.log(1 + (len(self.rows) - self.df[term] + .5) / (self.df[term] + .5))
                freq = counts[term]
                score += idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * self.lengths[i] / self.avg))
            scores.append((-score, self.rows[i]['ID'], i))
        return [self.rows[i] for _, _, i in sorted(scores)[:n]]
