import sys, json, pickle
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'tools'); sys.path.insert(0, '.')
import run_task2 as R
from jev.client import JevClient
from jev.choose import PairChooser
from jev.retrieval import Retriever
from jev.spans import null_disabled
from jev.data import load_jsonl
split, pool_file = sys.argv[1], sys.argv[2]
out, _ = pickle.load(open(pool_file, 'rb'))
pools = {(c, row['ID']): (row, rows) for c, row, rows, _ in out}
client = R.CachedClient(JevClient(timeout=120), R.CACHE / 'calls', R.DEFAULT_MODEL, 'task2').with_stage('choose')
(R.CACHE / split / 'choose').mkdir(parents=True, exist_ok=True)
fails = [0]
for c in R.CORPORA:
    retr = Retriever(load_jsonl(R.split_path(c, 'train')), [r['Text'] for s in ('dev', 'test') for r in load_jsonl(R.split_path(c, s))])
    def run(row):
        _, rows = pools[(c, row['ID'])]; text = row['Text']; tl = text.lower()
        surf = lambda s: text[tl.find(s):tl.find(s) + len(s)]
        aspects = list(dict.fromkeys(surf(a) for _, (a, o) in rows if a != 'null'))
        opinions = list(dict.fromkeys(surf(o) for _, (a, o) in rows))
        try:
            R.cached_stage(R.CACHE / split / 'choose' / f'{R.record_key(c, row)}.json',
                           lambda: PairChooser()(client, {'ID': row['ID'], 'Text': text}, aspects, opinions, retr, not null_disabled(c)))
        except Exception as e:
            fails[0] += 1; print(c, row['ID'], type(e).__name__, flush=True)
    with ThreadPoolExecutor(10) as ex: list(ex.map(run, load_jsonl(R.split_path(c, split))))
    print(c, 'done', flush=True)
print('failures', fails[0])
