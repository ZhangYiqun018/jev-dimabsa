"""Shared Task 2 helpers: corpora, request caching, pair VA, official cF1.

Used by the Task 2 pipeline (tools/run_task2.py). The lexicon baseline
(jev/triplets.py via runners/run.py) predates this module and keeps its own
frozen code path.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

from .client import Answer, Response, format_va, score_to_va
from .rubrics import arousal_question, valence_question

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'vendor/DimABSA2026/task-dataset'
CORPORA = ['eng_restaurant', 'eng_laptop', 'zho_restaurant', 'zho_laptop',
           'jpn_hotel', 'rus_restaurant', 'tat_restaurant', 'ukr_restaurant']
VA_BATCH_PAIRS = 16


def split_path(corpus, split):
    folder = DATA / 'track_a/subtask_2' / corpus[:3]
    name = f'{corpus}_train_alltasks.jsonl' if split == 'train' else f'{corpus}_{split}_task2.jsonl'
    return folder / name


def record_key(corpus, record):
    """Record IDs repeat across corpora, so cache identity includes corpus and text."""
    return hashlib.sha256((corpus + record['ID'] + record['Text']).encode()).hexdigest()[:16]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


# ------------------------------------------------------------ requests

class CachedClient:
    """Per-request response cache keyed by model, full state and full questions.

    A response is reused only when the model-visible request is identical, never
    because the AO strings match. Thread-safe; concurrent identical requests
    wait for one call.
    """

    def __init__(self, client, directory, model, stage):
        self.client, self.directory, self.model, self.stage = client, directory, model, stage
        self._lock = threading.Lock()
        self._key_locks = {}

    def with_stage(self, stage):
        other = CachedClient(self.client, self.directory, self.model, stage)
        other._lock, other._key_locks = self._lock, self._key_locks
        return other

    def ask(self, state, questions):
        payload = {'model': self.model, 'state': state, 'questions': questions}
        key = digest(payload)
        path = self.directory / f'{key}.json'
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            if path.exists():
                saved = json.loads(path.read_text())
                return Response(saved['model'], {k: Answer(k, questions[k]['type'], v)
                                for k, v in saved['answers'].items()},
                                saved['usage'], saved['attempts'])
            result = self.client.ask(state, questions, attempts=3)
            save_json(path, {**payload, 'stage': self.stage, 'model': result.model,
                             'usage': result.usage, 'attempts': result.attempts,
                             'answers': {k: a.raw for k, a in result.answers.items()}})
            return result


# ------------------------------------------------------------ VA and scoring

def score_pair_list(client, record, pairs, calibration):
    """V/A for an ordered list of unique (aspect, opinion) pairs, 16 pairs per request."""
    raw, calibrated, trace = [], [], []
    for offset in range(0, len(pairs), VA_BATCH_PAIRS):
        batch = pairs[offset:offset + VA_BATCH_PAIRS]
        questions = {}
        for i, (a, o) in enumerate(batch):
            questions[f'v{i}'] = valence_question(a, o)
            questions[f'a{i}'] = arousal_question(a, o)
        response = client.ask(record['Text'], questions)
        trace.append({'state': record['Text'], 'questions': questions,
                      'answers': {k: a.raw for k, a in response.answers.items()},
                      'model': response.model, 'usage': response.usage,
                      'attempts': response.attempts})
        for i, (a, o) in enumerate(batch):
            va = [score_to_va(response.answers[f'{d}{i}'].score) for d in ('v', 'a')]
            shrunk = [min(9., max(1., s * x + b)) for x, s, b in
                      zip(va, calibration['slope'], calibration['intercept'])]
            raw.append({'Aspect': a, 'Opinion': o, 'VA': format_va(*va)})
            calibrated.append({'Aspect': a, 'Opinion': o, 'VA': format_va(*shrunk)})
    return {'ID': record['ID'], 'raw': raw, 'calibrated': calibrated, 'trace': trace}


def official_score(gold, pred, log):
    """Run the unmodified official Task 2 scorer; stdout/stderr are kept in ``log``."""
    result = subprocess.run([sys.executable, str(ROOT / 'scoring/score.py'), '--task', '2',
                             '--gold', str(gold), '--pred', str(pred)],
                            capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr)
    match = re.search(r'Final Results: (\{.*\})', result.stdout)
    if result.returncode or not match:
        raise RuntimeError(f'Official scorer failed; see {log}')
    metrics = ast.literal_eval(re.sub(r'np\.float64\(([^)]*)\)', r'\1', match.group(1)))
    metrics['categorical_TP'] = float(re.search(r'True Positives \(TP\): ([\d.]+)',
                                                result.stdout).group(1))
    return metrics
