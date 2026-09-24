import unittest

from jev.client import Answer, Response
from jev.task2 import score_pair_list


class PairVATest(unittest.TestCase):
    def test_calibration_clipping_and_no_gold(self):
        class Fake:
            def ask(self, state, questions):
                self.state, self.questions = state, questions
                return Response('fake', {k: Answer(k, 'score', {'score': 8 if k.startswith('v') else 0})
                                         for k in questions}, {'input_tokens': 0})
        client = Fake()
        result = score_pair_list(client, {'ID': 'x', 'Text': 'good food', 'Triplet': 'SECRET'},
                                 [('food', 'good')], {'slope': [2, 2], 'intercept': [0, -3]})
        self.assertEqual(len(client.questions), 2)
        self.assertEqual(result['raw'], [{'Aspect': 'food', 'Opinion': 'good', 'VA': '9.00#1.00'}])
        self.assertEqual(result['calibrated'][0]['VA'], '9.00#1.00')
        self.assertEqual(client.state, 'good food')
        self.assertNotIn('SECRET', str(client.questions))


if __name__ == '__main__':
    unittest.main()
