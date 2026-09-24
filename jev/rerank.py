"""Task 2 candidate reranker: logistic regression over Jev signals of lattice pairs.

Each candidate pair (lattice Noul probability >= ``CANDIDATE_MINIMUM``) is
described by the Jev answers already collected for it and by train statistics:

- lattice Noul, BIO r3 Noul, lexicon-baseline Noul (when the pair was asked);
- example-conditioned span checks (aspect, opinion) and pair check;
- BIO-marginal span scores, BIO argmax / affix-normalised membership;
- train annotation counts, span lengths, competing boundary variants, distance.

One model per language group is fitted with an L2 penalty and applied per record: pairs are kept in
score order, a pair overlapping a kept pair on both roles is suppressed, and the
threshold stops the list. Fitting uses the dev annotations, so dev estimates
must come from cross-validation by record (``cross_validate``).
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from .data import _annotation_items
from .extraction import tokenize
from .lattice import _overlap, lattice_spans, marginals
from .postprocess import affix_counts, normalise_span, occurrences, train_rows
from .task2 import CORPORA

CANDIDATE_MINIMUM = .3
GROUPS = {'eng': ('eng_restaurant', 'eng_laptop'), 'zho': ('zho_restaurant', 'zho_laptop'),
          'jpn': ('jpn_hotel',), 'rtu': ('rus_restaurant', 'tat_restaurant', 'ukr_restaurant')}
PUNCT = re.compile(r'[，。！？!?,.;；、\n]')


def _logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def _log(x):
    return math.log(max(x, 1e-4))


@lru_cache(maxsize=None)
def train_counts(corpus):
    counts = {'aspect': Counter(), 'opinion': Counter()}
    for row in train_rows(corpus):
        for role, key in (('aspect', 'Aspect'), ('opinion', 'Opinion')):
            for span in {x[key].lower() for x in _annotation_items(row) if x[key] != 'NULL'}:
                counts[role][span] += 1
    return counts


def _distance(text_l, a, o):
    if a == 'null':
        return [0., 0., 0., 1.]
    best = None
    for s1, e1 in occurrences(text_l, a):
        for s2, e2 in occurrences(text_l, o):
            gap = max(s2 - e1, s1 - e2, 0)
            if best is None or gap < best[0]:
                best = (gap, s1 < s2, bool(PUNCT.search(text_l[min(e1, e2):max(s1, s2)])))
    if best is None:
        return [0., 0., 0., 0.]
    return [math.log1p(best[0]), float(best[1]), float(best[2]), 0.]


def edge_features(corpus, role, span, text, tokens):
    """Train log-odds that the span's first/last token sits inside a span edge, and that the
    tokens just outside sit outside it (0 when unaligned or unseen)."""
    if span == 'null':
        return [0.] * 4
    counts = affix_counts(corpus, role)
    starts = {t.start: k for k, t in enumerate(tokens)}
    ends = {t.end: k for k, t in enumerate(tokens)}
    text_l = text.lower()
    s = text_l.find(span)
    if s < 0 or s not in starts or s + len(span) not in ends:
        return [0.] * 4
    i, j = starts[s], ends[s + len(span)] + 1
    odds = lambda side, k, inside: (lambda c: math.log((c[0] + 1) / (c[1] + 1)) * (1 if inside else -1))(
        counts.get((side, tokens[k].text.lower()), [0, 0]))
    return [odds('pre', i, True) if j - i > 1 else 0., odds('suf', j - 1, True) if j - i > 1 else 0.,
            odds('pre', i - 1, False) if i > 0 else 0., odds('suf', j, False) if j < len(tokens) else 0.]


def span_priors(extracted, text):
    tokens, probs = marginals(extracted, text)
    text_l = text.lower()
    out = {}
    for role in ('aspect', 'opinion'):
        scores = {}
        for (s, e), v in lattice_spans(tokens, probs[role], 1e-6).items():
            scores[text_l[s:e]] = max(scores.get(text_l[s:e], 0), v)
        out[role] = scores
    return out


def features(corpus, text, lattice, extracted, signals):
    """[(feature vector, (aspect, opinion) lower-case)] for the record's candidate pairs.

    ``signals``: {'lexicon': {pair: p}, 'spancheck': {role: {span: p}}, 'paircheck': {pair: p}}.
    """
    text_l = text.lower()
    tokens = tokenize(text)
    priors = span_priors(extracted, text)
    argmax = {r: {s.lower() for s in extracted['spans'][r]} for r in ('aspect', 'opinion')}
    normalised = {r: {text_l[s:e] for s, e in (normalise_span(text, tokens, corpus, r, *o)
                                                for o in extracted['offsets'][r])}
                  for r in ('aspect', 'opinion')}
    bio = {(p['Aspect'].lower(), p['Opinion'].lower()): p['probability'] for p in extracted['pairs']}
    counts = train_counts(corpus)
    group = next(g for g, members in GROUPS.items() if corpus in members)
    pool = [p for p in lattice['pairs'] if p['probability'] >= CANDIDATE_MINIMUM]
    keys = [(p['Aspect'].lower(), p['Opinion'].lower()) for p in pool]
    rows = []
    for (a, o), p in zip(keys, pool):
        competing = [q['probability'] for (qa, qo), q in zip(keys, pool)
                     if q is not p and _overlap(a, qa, text_l) and _overlap(o, qo, text_l)]
        span_a = 1. if a == 'null' else signals['spancheck']['aspect'].get(a, .5)
        span_o = signals['spancheck']['opinion'].get(o, .5)
        main = [_logit(p['probability']),
                0. if a == 'null' else _log(priors['aspect'].get(a, 0)), _log(priors['opinion'].get(o, 0)),
                _logit(bio.get((a, o), .5)), _logit(span_a), _logit(span_o),
                _logit(signals['paircheck'].get((a, o), .5))]
        f = main + [
            float(a == 'null'), float(a in argmax['aspect']), float(o in argmax['opinion']),
            float(a in normalised['aspect']), float(o in normalised['opinion']),
            math.log1p(counts['aspect'].get(a, 0)), math.log1p(counts['opinion'].get(o, 0)),
            math.log(len(a)) if a != 'null' else 0., math.log(len(o)),
            p['probability'] - max(competing, default=0), math.log1p(len(competing)),
            signals['lexicon'].get((a, o), 0.), float((a, o) in signals['lexicon']),
            float((a, o) in bio), *_distance(text_l, a, o),
            *edge_features(corpus, 'aspect', a, text, tokens), *edge_features(corpus, 'opinion', o, text, tokens)]
        f += [float(corpus == c) for c in CORPORA]
        f += [x * (g == group) for g in GROUPS for x in main]  # Per-language-group slopes.
        rows.append((f, (a, o)))
    return rows


def fit(X, y, l2=5.):
    X = np.hstack([np.asarray(X, dtype=float), np.ones((len(X), 1))])
    y = np.asarray(y, dtype=float)

    def loss(w):
        p = 1 / (1 + np.exp(-(X @ w)))
        value = -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + l2 * np.sum(w[:-1] ** 2)
        return value, X.T @ (p - y) + 2 * l2 * np.r_[w[:-1], 0]

    return minimize(loss, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B').x


def score(w, rows):
    return [(1 / (1 + math.exp(-(float(np.dot(w[:-1], f)) + w[-1]))), pair) for f, pair in rows]


def select(scored, text, threshold):
    """Score order, suppress pairs overlapping a kept pair on both roles, stop at threshold."""
    text_l = text.lower()
    kept = []
    for s, (a, o) in sorted(scored, key=lambda x: -x[0]):
        if s < threshold:
            break
        if not any(_overlap(a, ka, text_l) and _overlap(o, ko, text_l) for ka, ko in kept):
            kept.append((a, o))
    return kept


def fold(corpus, record_id, folds=5):
    return int(hashlib.sha256((corpus + record_id).encode()).hexdigest(), 16) % folds


def group_of(corpus):
    return next(g for g, members in GROUPS.items() if corpus in members)


def fit_groups(items, l2=5.):
    """One model per language group. items: [(corpus, record_id, rows, gold set)]."""
    models = {}
    for g in GROUPS:
        X, y = [], []
        for corpus, _, rows, gold in items:
            if group_of(corpus) == g:
                X += [f for f, _ in rows]
                y += [float(pair in gold) for _, pair in rows]
        models[g] = fit(X, y, l2)
    return models


def cross_validate(items, folds=5, l2=5.):
    """Out-of-fold scores per item (folds by record, one model per language group)."""
    out = [None] * len(items)
    for k in range(folds):
        models = fit_groups([it for it in items if fold(it[0], it[1], folds) != k], l2)
        for i, (corpus, rid, rows, gold) in enumerate(items):
            if fold(corpus, rid, folds) == k:
                out[i] = score(models[group_of(corpus)], rows)
    return out
