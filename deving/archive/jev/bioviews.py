"""Task 2 BIO label views: the BIO r3 label questions with retrieved train examples.

The r3 extraction (jev/extraction.py) labels every token once, with two invented
examples. On dev, gold pairs the reranker misses have much lower BIO-marginal
opinion scores than kept ones, while the example-conditioned checks
separate them less. Here the same per-token B/I/O questions are asked again
with ``EXAMPLES`` BM25-retrieved train reviews and their annotated phrases in
place of the invented examples, once per retrieval view. Only the label
requests are repeated (no pair questions); the per-token probabilities of each
view become reranker features (jev/rerank.py).
"""
from __future__ import annotations

from .extraction import BOUNDARY_GUIDANCE, CHUNK_TOKENS, LABEL_BATCH, RULES, TYPES, label_question, tokenize
from .retrieval import example

EXAMPLES = 4
GUIDANCE = (
    'The annotation examples are reviews from the same dataset with every annotated aspect and '
    'opinion phrase. Label tokens so that the phrases follow those conventions: what counts as '
    'an aspect or opinion, and which words belong inside a phrase at each edge.')


class ExampleLabeler:
    """Per-token B/I/O Choice per role with retrieved examples; trace shaped like BIOExtractor's."""

    def __call__(self, client, record, retriever):
        text = record['Text']  # Deliberately never read annotations of the record itself.
        tokens = tokenize(text)
        examples = [example(r) for r in retriever.select(text, EXAMPLES)]
        trace = []
        for offset in range(0, len(tokens), CHUNK_TOKENS):
            chunk = tokens[offset:offset + CHUNK_TOKENS]
            state = {'review': text, 'rules': RULES,
                     'tokens': '\n'.join(f'{i}|{t.text}' for i, t in enumerate(chunk)),
                     'boundary_guidance': BOUNDARY_GUIDANCE,
                     'guidance': GUIDANCE, 'annotation_examples': examples}
            questions = {f'{kind}_{i}': label_question(kind, i, token)
                         for kind in TYPES for i, token in enumerate(chunk)}
            names = list(questions)
            for start in range(0, len(names), LABEL_BATCH):
                batch = {n: questions[n] for n in names[start:start + LABEL_BATCH]}
                response = client.ask(state, batch)
                trace.append({'state': {'tokens': state['tokens']},
                              'answers': {k: a.raw for k, a in response.answers.items()},
                              'model': response.model, 'usage': response.usage,
                              'attempts': response.attempts})
        return {'ID': record['ID'], 'trace': trace}
