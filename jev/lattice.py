"""Task 2 lattice pairing: many boundary candidates per role, Noul per pair, suppression.

Reuses a cached BIO r3 extraction. Candidates per role are the BIO argmax spans,
every span whose BIO-marginal score reaches ``LATTICE_THRESHOLD``, the
train-affix normalisation of each, and train-lexicon matches. Every non-overlapping
aspect x opinion pair gets the BIO r3 Noul pair question (NULL aspects only where
train/README allow them). Decoding keeps accepted pairs in probability order and
suppresses a pair whose aspect and opinion both overlap an already kept pair, so
competing boundary variants resolve to the one Noul prefers.
"""
from __future__ import annotations

from .extraction import RULES, pair_question, tokenize
from .postprocess import normalise_span, null_disabled, occurrences

LATTICE_THRESHOLD = .2
LATTICE_MAX_TOKENS = 12
PAIR_BATCH = 32
# Revision 2: chains of up to MERGE_CHAIN candidates separated by at most MERGE_GAP tokens,
# because many missed gold phrases are adjacent BIO fragments (a hedge word plus an adjective).
MERGE_GAP, MERGE_CHAIN, MERGE_MAX_TOKENS = 1, 3, 16


def marginals(extracted, text):
    """Per-token B/I/O probabilities for each role from the cached label requests."""
    tokens = tokenize(text)
    probs = {'aspect': [None] * len(tokens), 'opinion': [None] * len(tokens)}
    offset, previous = 0, None
    for call in extracted['trace']:
        state = call['state']
        if not isinstance(state, dict) or 'tokens' not in state:
            continue
        if previous is not None and state['tokens'] != previous:
            offset += len(previous.split('\n'))
        previous = state['tokens']
        for name, answer in call['answers'].items():
            role, index = name.rsplit('_', 1)
            probs[role][offset + int(index)] = answer['probabilities']
    return tokens, probs


def lattice_spans(tokens, probs, threshold=LATTICE_THRESHOLD):
    """Character spans whose start x continuation x end probability reaches ``threshold``."""
    spans = {}
    for i in range(len(tokens)):
        run = probs[i].get('B', 0) + probs[i].get('I', 0) * (probs[i - 1].get('O', 0) if i else 1.)
        for j in range(i, min(len(tokens), i + LATTICE_MAX_TOKENS)):
            if j > i:
                run *= probs[j].get('I', 0)
            if run < 1e-4:
                break
            end = 1 - (probs[j + 1].get('I', 0) if j + 1 < len(tokens) else 0)
            if run * end >= threshold:
                spans[(tokens[i].start, tokens[j].end)] = run * end
    return spans


def merge_adjacent(tokens, spans):
    """Spans plus chains of spans separated by at most MERGE_GAP tokens."""
    starts = {t.start: k for k, t in enumerate(tokens)}
    ends = {t.end: k for k, t in enumerate(tokens)}
    units = [(starts[s], ends[e]) for s, e in spans if s in starts and e in ends]
    found, frontier = set(units), set(units)
    for _ in range(MERGE_CHAIN - 1):
        grown = {(i, l) for i, j in frontier for k, l in units
                 if j < k <= j + 1 + MERGE_GAP and l - i < MERGE_MAX_TOKENS} - found
        found |= grown
        frontier = grown
    return set(spans) | {(tokens[i].start, tokens[j].end) for i, j in found}


def candidates(extracted, text, corpus, lexicon=None, merge=False):
    """{role: {lower surface: (surface, first offset)}} ordered by first offset."""
    tokens, probs = marginals(extracted, text)
    patterns = {'aspect': lexicon.aspect_pattern, 'opinion': lexicon.opinion_pattern} if lexicon else {}
    out = {}
    for role in ('aspect', 'opinion'):
        spans = {tuple(s) for s in extracted['offsets'][role]} | set(lattice_spans(tokens, probs[role]))
        spans |= {normalise_span(text, tokens, corpus, role, s, e) for s, e in list(spans)}
        if merge:
            spans = merge_adjacent(tokens, spans)
        if role in patterns:
            spans |= {(m.start(1), m.end(1)) for m in patterns[role].finditer(text)}
        found = {}
        for s, e in sorted(spans):
            surface = text[s:e]
            if surface.strip():
                found.setdefault(surface.lower(), (surface, s))
        out[role] = found
    return out


def _overlap(x, y, text_l):
    """Surfaces overlap: containment, or any two text occurrences intersect."""
    if x == 'null' or y == 'null':
        return x == y
    if x in y or y in x:
        return True
    return any(s1 < e2 and s2 < e1 for s1, e1 in occurrences(text_l, x)
               for s2, e2 in occurrences(text_l, y))


class LatticePairer:
    """Noul for every candidate pair; returns all pairs with probabilities.

    ``known`` maps lower-case pairs to probabilities from an earlier revision;
    those pairs are not asked again.
    """

    def __init__(self, merge=False, known=None):
        self.merge, self.known = merge, known or {}

    def __call__(self, client, record, extracted, corpus, lexicon=None):
        text = record['Text']  # Deliberately never read annotations.
        text_l = text.lower()
        cands = candidates(extracted, text, corpus, lexicon, self.merge)
        aspects = [s for s, _ in cands['aspect'].values()]
        if not null_disabled(corpus):
            aspects.append('NULL')
        opinions = [s for s, _ in cands['opinion'].values()]
        pairs = [(a, o) for a in aspects for o in opinions
                 if a == 'NULL' or not _overlap(a.lower(), o.lower(), text_l)]
        out = [{'Aspect': a, 'Opinion': o, 'probability': self.known[(a.lower(), o.lower())]}
               for a, o in pairs if (a.lower(), o.lower()) in self.known]
        pairs = [(a, o) for a, o in pairs if (a.lower(), o.lower()) not in self.known]
        trace = []
        for offset in range(0, len(pairs), PAIR_BATCH):
            batch = pairs[offset:offset + PAIR_BATCH]
            questions = {f'p{i}': pair_question(a, o) for i, (a, o) in enumerate(batch)}
            response = client.ask({'review': text, 'rules': RULES}, questions)
            trace.append({'n': len(batch), 'model': response.model, 'usage': response.usage,
                          'attempts': response.attempts})
            out.extend({'Aspect': a, 'Opinion': o, 'probability': response.answers[f'p{i}'].noul}
                       for i, (a, o) in enumerate(batch))
        return {'ID': record['ID'], 'candidates': {r: list(c) for r, c in cands.items()},
                'pairs': out, 'trace': trace}


def suppress(pairs, text, threshold):
    """Accepted pairs, highest probability first, without overlapping boundary variants."""
    text_l = text.lower()
    kept = []
    for pair in sorted((p for p in pairs if p['probability'] >= threshold),
                       key=lambda p: -p['probability']):
        a, o = pair['Aspect'].lower(), pair['Opinion'].lower()
        if any(_overlap(a, k['Aspect'].lower(), text_l) and _overlap(o, k['Opinion'].lower(), text_l)
               for k in kept):
            continue
        kept.append(pair)
    return kept
