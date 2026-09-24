import hashlib
import json
import unittest

from jev.client import Answer, Response
from jev.extraction import BIOExtractor, decode_bio, tokenize


class RecordingClient:
    """Alternates B/I labels by token index and accepts every pair."""

    def __init__(self):
        self.sent = []

    def ask(self, state, questions):
        self.sent.append({'state': state, 'questions': questions})
        answers = {}
        for name, question in questions.items():
            raw = ({'choice': 'BI'[int(name.split('_')[-1]) % 2]} if question['type'] == 'choice'
                   else {'noul': .7})
            answers[name] = Answer(name, question['type'], raw)
        return Response('jev-1.13.0', answers, {'input_tokens': 1})


class ExtractionTests(unittest.TestCase):
    def test_offsets_preserve_unicode_spacing_and_bio_boundaries(self):
        text = '餐廳的 customer  service 很好，很好。'
        tokens = tokenize(text)
        labels = ['O'] * len(tokens)
        for i, token in enumerate(tokens):
            self.assertEqual(text[token.start:token.end], token.text)
            if token.text in ('customer', 'service'):
                labels[i] = 'I'  # Orphan I is recovered as a start.
            elif token.text == '很':
                labels[i] = 'B'
            elif token.text == '好':
                labels[i] = 'I'
        spans = decode_bio(tokens, labels)
        self.assertEqual([text[s:e] for s, e in spans],
                         ['customer  service', '很好', '很好'])
        self.assertNotEqual(spans[1], spans[2])

    def test_requests_match_frozen_bio_r3_cache_identity(self):
        # EXTRACTOR_ID names historical caches; any prompt or batching change must
        # fail here rather than silently reuse or orphan them. Digest taken from the
        # archived original extractor on the same input.
        client = RecordingClient()
        record = {'ID': 'x', 'Text': 'Food great, 湯很香 but service slow.',
                  'Triplet': [{'Aspect': 'GOLD_SENTINEL', 'Opinion': 'DO_NOT_READ'}]}
        result = BIOExtractor()(client, record)
        digest = hashlib.sha256(json.dumps(client.sent, ensure_ascii=False,
                                           sort_keys=True).encode()).hexdigest()[:16]
        self.assertEqual(digest, '6178bf3627c6b3d7')
        self.assertNotIn('GOLD_SENTINEL', repr(client.sent))
        # Every aspect plus NULL is crossed with every opinion.
        aspects, opinions = result['spans']['aspect'], result['spans']['opinion']
        self.assertEqual(len(result['pairs']), (len(aspects) + 1) * len(opinions))
        self.assertFalse(result['chunked'])


if __name__ == '__main__':
    unittest.main()
