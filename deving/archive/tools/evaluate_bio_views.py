"""Dev CV: v2 adopted features + BIO label views (cache only). argv: views (comma) [corpora]"""
import sys, json, pickle, collections
sys.path.insert(0, 'tools'); sys.path.insert(0, '.')
import run_task2 as R
from jev.rerank import features, cross_validate, select
from jev.extend import KEEP
BV = [v for v in sys.argv[1].split(',') if v]
ON = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else {'zho_laptop', 'zho_restaurant'}
split = 'dev'
items = pickle.load(open('/tmp/t2_dev_items.pkl', 'rb'))
lex = {c: R.Lexicon(c, split) for c in R.CORPORA}
by = lambda ps: {(q['Aspect'].lower(), q['Opinion'].lower()): q['probability'] for q in ps}
out = []
for c, row, _, gold in items:
    key = R.record_key(c, row); p = lambda st: json.load(open(R.CACHE / split / st / f'{key}.json'))
    lat, ext, sp, pc, chk, ep = p('lattice'), p('extract'), p('spancheck'), p('paircheck'), p('extcheck'), p('extpairs')
    signals = {'lexicon': lex[c].probabilities(row),
               'spancheck': {'aspect': sp['spans']['aspect'], 'opinion': {**sp['spans']['opinion'], **chk['opinion']}},
               'paircheck': {**by(pc['pairs']), **by(ep['paircheck'])},
               'extension': {s for s, v in chk['opinion'].items() if v >= KEEP},
               'views': [{'spancheck': p(f'spancheck_{v}')['spans'], 'paircheck': by(p(f'paircheck_{v}')['pairs'])} for v in ('trigram', 'word')],
               'relative': True}
    if BV and c in ON: signals['bio_views'] = [p(f'bio_{v}') for v in BV]
    lat = {**lat, 'pairs': lat['pairs'] + ep['pairs']}
    out.append((c, row, features(c, row['Text'], lat, ext, signals), gold))
scored = cross_validate([(c, r['ID'], rows, g) for c, r, rows, g in out])
res = {}
for c in R.CORPORA:
    tp = fp = fn = 0
    for (cc, row, rows, gold), sc in zip(out, scored):
        if cc != c: continue
        ch = set(select(sc, row['Text'], R.THRESHOLD)); tp += len(ch & gold); fp += len(ch - gold); fn += len(gold - ch)
    res[c] = 100 * 2 * tp / (2 * tp + fp + fn)
print(BV, 'macro %.2f' % (sum(res.values()) / 8), {c: round(v, 2) for c, v in res.items()})
