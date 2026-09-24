"""Train-split span statistics for Task 2: NULL-aspect policy and edge affixes.

- NULL policy: implicit (NULL) aspects are not proposed where the official
  README says test has none (English) or where train has fewer than 5% of them.
- Edge affixes: for every train span, each 1..n-token affix at its edges is
  counted as inside the span or immediately outside it. Affixes almost always
  left outside (>= 90%, >= 20 times) are stripped from candidate spans and
  those almost always inside are added, producing boundary variants.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from .data import _annotation_items, load_jsonl
from .extraction import tokenize
from .task2 import split_path

CJK = {'zho_restaurant', 'zho_laptop', 'jpn_hotel'}
NULL_RATE_LIMIT = .05
AFFIX_RATIO, AFFIX_SUPPORT = .9, 20


@lru_cache(maxsize=None)
def train_rows(corpus):
    return tuple(load_jsonl(split_path(corpus, 'train')))


def occurrences(text_l, span_l):
    out, i = [], text_l.find(span_l)
    while i >= 0 and span_l:
        out.append((i, i + len(span_l)))
        i = text_l.find(span_l, i + 1)
    return out


@lru_cache(maxsize=None)
def null_disabled(corpus):
    if corpus.startswith('eng_'):
        return True  # Official English README: test excludes implicit (NULL) aspects.
    items = [x for r in train_rows(corpus) for x in _annotation_items(r)]
    return sum(x['Aspect'] == 'NULL' for x in items) / len(items) < NULL_RATE_LIMIT


def _affix(text, tokens, i, j):
    return text[tokens[i].start:tokens[j - 1].end].lower()


def _max_affix(corpus):
    return 4 if corpus in CJK else 2


@lru_cache(maxsize=None)
def affix_rules(corpus, role):
    """(strip, extend), each {'pre': set, 'suf': set} of affix strings.

    Strip when exc/(inc+exc) >= 0.9 with exc >= 20; extend symmetrically
    (counts from ``affix_counts``).
    """
    strip, extend = {'pre': set(), 'suf': set()}, {'pre': set(), 'suf': set()}
    for (side, affix), (inc, exc) in affix_counts(corpus, role).items():
        total = inc + exc
        if not affix.strip() or not total:
            continue
        if exc >= AFFIX_SUPPORT and exc / total >= AFFIX_RATIO:
            strip[side].add(affix)
        if inc >= AFFIX_SUPPORT and inc / total >= AFFIX_RATIO:
            extend[side].add(affix)
    return strip, extend


@lru_cache(maxsize=None)
def affix_counts(corpus, role):
    """{(side, affix): [inc, exc]} over train spans of ``role``.

    For every train span occurrence aligned to tokens, each edge affix of
    1..n tokens counts as inside the span (inc) or immediately outside it (exc).
    """
    key = 'Aspect' if role == 'aspect' else 'Opinion'
    maxn = _max_affix(corpus)
    counts = defaultdict(lambda: [0, 0])
    for row in train_rows(corpus):
        text = row['Text']
        tokens = tokenize(text)
        starts = {t.start: k for k, t in enumerate(tokens)}
        ends = {t.end: k for k, t in enumerate(tokens)}
        seen = set()
        for item in _annotation_items(row):
            span = item[key].lower()
            if span == 'null' or span in seen:
                continue
            seen.add(span)
            for s, e in occurrences(text.lower(), span):
                if s not in starts or e not in ends:
                    continue
                i, j = starts[s], ends[e] + 1
                for n in range(1, maxn + 1):
                    if j - n > i:
                        counts[('suf', _affix(text, tokens, j - n, j))][0] += 1
                        counts[('pre', _affix(text, tokens, i, i + n))][0] += 1
                    if j + n <= len(tokens):
                        counts[('suf', _affix(text, tokens, j, j + n))][1] += 1
                    if i - n >= 0:
                        counts[('pre', _affix(text, tokens, i - n, i))][1] += 1
    return dict(counts)


def normalise_span(text, tokens, corpus, role, s, e, extend=True):
    """Apply train affix rules to one character span; unaligned spans are unchanged."""
    starts = {t.start: k for k, t in enumerate(tokens)}
    ends = {t.end: k for k, t in enumerate(tokens)}
    if s not in starts or e not in ends:
        return s, e
    strip, ext = affix_rules(corpus, role)
    i, j = starts[s], ends[e] + 1
    maxn = _max_affix(corpus)
    for _ in range(4):
        changed = False
        for n in range(maxn, 0, -1):  # Longest affix first.
            if j - n > i and _affix(text, tokens, j - n, j) in strip['suf']:
                j -= n
                changed = True
                break
        for n in range(maxn, 0, -1):
            if i + n < j and _affix(text, tokens, i, i + n) in strip['pre']:
                i += n
                changed = True
                break
        if not changed:
            break
    for _ in range(3 if extend else 0):
        changed = False
        for n in range(maxn, 0, -1):
            if j + n <= len(tokens) and _affix(text, tokens, j, j + n) in ext['suf']:
                j += n
                changed = True
                break
        for n in range(maxn, 0, -1):
            if i - n >= 0 and _affix(text, tokens, i - n, i) in ext['pre']:
                i -= n
                changed = True
                break
        if not changed:
            break
    return tokens[i].start, tokens[j - 1].end
