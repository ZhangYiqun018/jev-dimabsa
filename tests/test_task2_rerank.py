import unittest

from jev.client import Answer, Response
from jev.lattice import LatticePairer
from jev.rerank import select
from jev.spancheck import PairChecker, SpanChecker


class FakeClient:
    def __init__(self):
        self.sent = []

    def ask(self, state, questions):
        self.sent.append({'state': state, 'questions': questions})
        return Response('jev-1.13.0', {k: Answer(k, 'noul', {'noul': .8}) for k in questions},
                        {'input_tokens': 1})


class FakeRetriever:
    def select(self, text, n=4):
        return [{'ID': 't', 'Text': 'Soup is tasty.',
                 'Triplet': [{'Aspect': 'Soup', 'Opinion': 'tasty', 'VA': '7#6'}]}]


class RerankTests(unittest.TestCase):
    def test_select_suppresses_boundary_variants_but_keeps_distinct_pairs(self):
        text = 'The screen is very bright and the keyboard is great'
        scored = [(.9, ('screen', 'very bright')), (.8, ('screen', 'bright')),
                  (.7, ('keyboard', 'great')), (.1, ('keyboard', 'is great'))]
        self.assertEqual(select(scored, text, .25), [('screen', 'very bright'), ('keyboard', 'great')])

    def test_candidate_stages_never_send_annotations_and_reuse_known_pairs(self):
        record = {'ID': 'x', 'Text': 'Food great but service slow.',
                  'Triplet': [{'Aspect': 'GOLD_SENTINEL', 'Opinion': 'DO_NOT_READ'}]}
        extracted = {'offsets': {'aspect': [[0, 4], [15, 22]], 'opinion': [[5, 10], [23, 27]]},
                     'spans': {'aspect': ['Food', 'service'], 'opinion': ['great', 'slow']},
                     'trace': [], 'pairs': []}
        # No label trace: marginals are empty, so only argmax spans are candidates.
        extracted['trace'] = [{'state': {'tokens': '\n'.join(['a'] * 6)}, 'answers': {
            f'{r}_{i}': {'probabilities': {'O': 1.}} for r in ('aspect', 'opinion') for i in range(6)}}]
        client = FakeClient()
        result = LatticePairer(known={('food', 'great'): .9})(client, record, extracted, 'eng_restaurant')
        asked = [q['instructions'] for call in client.sent for q in call['questions'].values()]
        self.assertEqual(len(result['pairs']), 4)
        self.assertEqual(len(asked), 3)  # The known pair is not asked again; no NULL in English.
        SpanChecker()(client, record, result['candidates'], FakeRetriever())
        PairChecker()(client, record, result['pairs'], FakeRetriever())
        self.assertNotIn('GOLD_SENTINEL', repr(client.sent))


if __name__ == '__main__':
    unittest.main()
