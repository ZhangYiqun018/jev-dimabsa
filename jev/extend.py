"""Task 2 opinion extensions: longer boundary variants of the lattice opinion candidates.

On dev most missed gold opinions are longer than every overlapping lattice
candidate (a leading adverb, negation or clause head the BIO marginals did not
carry). Each opinion candidate is extended by up to ``LEFT`` tokens to the left
and ``RIGHT`` to the right (no punctuation inside, at most ``MAX_TOKENS``). The
example-conditioned span check (jev/checks.py) judges every new variant; those
reaching ``KEEP`` are paired with the lattice aspects, asked the BIO r3 pair
question and the pair check, and join the reranker pool flagged as extensions.
"""
from __future__ import annotations

import re

from .checks import PairChecker, SpanChecker
from .extraction import RULES, pair_question, tokenize
from .lattice import PAIR_BATCH, overlap
from .spans import null_disabled

LEFT, RIGHT, MAX_TOKENS = 3, 2, 12
KEEP = .5
PUNCT = re.compile(r'^[^\w\s]+$')


def extensions(text, opinions):
    """Lower-case surfaces of the new opinion variants, in text order."""
    tokens = tokenize(text)
    text_l = text.lower()
    starts = {t.start: k for k, t in enumerate(tokens)}
    ends = {t.end: k for k, t in enumerate(tokens)}
    known = {o.lower() for o in opinions}
    out = {}
    for surface in opinions:
        s = text_l.find(surface.lower())
        if s < 0 or s not in starts or s + len(surface) not in ends:
            continue
        i, j = starts[s], ends[s + len(surface)]
        for left in range(LEFT + 1):
            for right in range(RIGHT + 1):
                a, b = i - left, j + right
                if (left or right) and a >= 0 and b < len(tokens) and b - a < MAX_TOKENS \
                        and not any(PUNCT.match(t.text) for t in tokens[a:b + 1]):
                    new = text[tokens[a].start:tokens[b].end]
                    if new.lower() not in known:
                        out.setdefault(new.lower(), (tokens[a].start, new))
    return [new for _, new in sorted(out.values())]


def check_extensions(client, record, lattice, retriever):
    """Span-check Noul for each extension of the record's lattice opinion candidates."""
    variants = extensions(record['Text'], lattice['candidates']['opinion'])
    checked = SpanChecker()(client, record, {'aspect': [], 'opinion': variants}, retriever)
    return {'ID': record['ID'], 'opinion': checked['spans']['opinion'], 'trace': checked['trace']}


def extension_pairs(client, record, lattice, checked, corpus, retriever):
    """Pair Noul (BIO r3 question) and pair check for kept extensions x lattice aspects."""
    text = record['Text']
    text_l = text.lower()
    kept = [s for s, p in checked['opinion'].items() if p >= KEEP]
    surfaces = []
    for s in kept:
        start = text_l.find(s)
        surfaces.append(text[start:start + len(s)])
    aspects = [text[text_l.find(a):text_l.find(a) + len(a)] for a in lattice['candidates']['aspect']]
    aspects += [] if null_disabled(corpus) else ['NULL']
    pairs = [(a, o) for a in aspects for o in surfaces if a == 'NULL' or not overlap(a.lower(), o.lower(), text_l)]
    out, trace = [], []
    for offset in range(0, len(pairs), PAIR_BATCH):
        batch = pairs[offset:offset + PAIR_BATCH]
        response = client.ask({'review': text, 'rules': RULES},
                              {f'p{i}': pair_question(a, o) for i, (a, o) in enumerate(batch)})
        trace.append({'n': len(batch), 'model': response.model, 'usage': response.usage,
                      'attempts': response.attempts})
        out.extend({'Aspect': a, 'Opinion': o, 'probability': response.answers[f'p{i}'].noul}
                   for i, (a, o) in enumerate(batch))
    checks = PairChecker()(client, record, out, retriever)
    return {'ID': record['ID'], 'pairs': out, 'paircheck': checks['pairs'], 'trace': trace + checks['trace']}
