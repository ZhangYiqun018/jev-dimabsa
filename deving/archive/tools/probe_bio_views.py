"""Dev: example-conditioned BIO label views for selected corpora (cached per record)."""
import sys, json, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'tools'); sys.path.insert(0, '.')
import run_task2 as R
from jev.client import JevClient
from jev.bioviews import ExampleLabeler
from jev.retrieval import VIEWS, Retriever
from jev.data import load_jsonl
split, corpora, views = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(',')
STOP = float(sys.argv[4]) * 1e6
client = R.CachedClient(JevClient(timeout=120), R.CACHE / 'calls', R.DEFAULT_MODEL, 'task2')
used, lock, fails = [0], threading.Lock(), [0]
for v in views: (R.CACHE / split / f'bio_{v}').mkdir(parents=True, exist_ok=True)
for c in corpora:
    banned = [r['Text'] for s in ('dev', 'test') for r in load_jsonl(R.split_path(c, s))]
    train = load_jsonl(R.split_path(c, 'train'))
    retr = {v: Retriever(train, banned, VIEWS[v]) for v in views}
    jobs = [(row, v) for row in load_jsonl(R.split_path(c, split)) for v in views]
    def run(job):
        row, v = job
        p = R.CACHE / split / f'bio_{v}' / f'{R.record_key(c, row)}.json'
        if p.exists(): return
        if used[0] > STOP: return
        try:
            out = ExampleLabeler()(client.with_stage(f'bio_{v}'), {'ID': row['ID'], 'Text': row['Text']}, retr[v])
        except Exception as e:
            with lock: fails[0] += 1
            print(c, row['ID'], type(e).__name__, flush=True); return
        R.save_json(p, out)
        with lock: used[0] += sum((t['usage'] or {}).get('input_tokens', 0) for t in out['trace'])
    with ThreadPoolExecutor(max_workers=R.CONCURRENCY) as pool: list(pool.map(run, jobs))
    print(c, 'done; fresh input tokens so far %.2fM' % (used[0] / 1e6), flush=True)
print('failures', fails[0], 'stopped' if used[0] > STOP else '')
