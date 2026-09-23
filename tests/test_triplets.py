import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev.client import Answer, Response
from jev.data import write_jsonl
from jev.triplets import TripletPredictor


class TripletTests(unittest.TestCase):
    def test_text_only_pair_filtering_calibration_and_batch_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / 'train.jsonl'
            heldout = Path(directory) / 'heldout.jsonl'
            write_jsonl(train, [
                {'ID': 'train', 'Text': 'training example', 'Quadruplet': [
                    {'Aspect': 'food', 'Opinion': 'great', 'VA': '8#7'},
                    {'Aspect': 'rice', 'Opinion': 'great', 'VA': '8#7'},
                    {'Aspect': 'coffee', 'Opinion': 'great', 'VA': '8#7'}]},
                {'ID': 'overlap', 'Text': 'held out text', 'Quadruplet': [
                    {'Aspect': 'secret', 'Opinion': 'hidden', 'VA': '8#7'}]}])
            write_jsonl(heldout, [{'ID': 'test', 'Text': 'held out text'}])
            predictor = TripletPredictor(train, [heldout],
                {'slope': [.5, .25], 'intercept': [3, 5]})
            calls = []

            class Client:
                def ask(self, state, questions):
                    calls.append((state, questions))
                    answers = {}
                    for name in questions:
                        raw = ({'noul': .9 if name == 'p0' else .1} if name[0] == 'p'
                               else {'score': 6 if name[0] == 'v' else 4})
                        answers[name] = Answer(name, questions[name]['type'], raw)
                    return Response('jev-1.13.0', answers, {'input_tokens': 10})

            record = {'ID': 'target', 'Text': 'Food and coffee are great for the price.',
                      'Triplet': [{'Aspect': 'secret', 'Opinion': 'hidden', 'VA': '1#1'}]}
            with patch('jev.triplets.PAIR_BATCH', 1):
                result = predictor(Client(), record)
            self.assertEqual(result['Triplet'], [
                {'Aspect': 'Food', 'Opinion': 'great', 'VA': '6.50#6.25'}])
            self.assertEqual(result['_jev']['usage']['input_tokens'], 20)
            self.assertEqual(len(calls), 2)  # "rice" must not match "price".
            self.assertTrue(all(state == record['Text'] for state, _ in calls))
            self.assertEqual(predictor.candidates('secret hidden'), [])
            self.assertEqual(predictor(Client(), {'ID': 'empty', 'Text': 'Nothing here'})['Triplet'], [])
