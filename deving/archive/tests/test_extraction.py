import unittest

from jev.client import Answer, Response
from jev.extraction import SpanExtractor, tokenize, decode_bio


class ExtractionTests(unittest.TestCase):
    def test_offsets_preserve_unicode_spacing_and_bio_boundaries(self):
        text = '餐廳的 customer  service 很好，很好。'
        tokens = tokenize(text)
        labels = ['O'] * len(tokens)
        for i, token in enumerate(tokens):
            self.assertEqual(text[token.start:token.end], token.text)
            if token.text == 'customer':
                labels[i] = 'I'  # Orphan I is recovered as a start.
            elif token.text == 'service':
                labels[i] = 'I'
            elif token.text == '很':
                labels[i] = 'B'
            elif token.text == '好':
                labels[i] = 'I'
        spans = decode_bio(text, tokens, labels)
        self.assertEqual([text[s:e] for s,e in spans],
                         ['customer  service', '很好', '很好'])
        self.assertNotEqual(spans[1], spans[2])

    def test_pointer_extracts_multiple_pairs_and_implicit_target_without_gold(self):
        scheduled = iter([
            {'aspect': '0', 'opinion': '1'}, {'aspect': '0', 'opinion': '1'},
            {'aspect': '3', 'opinion': '4'}, {'aspect': '3', 'opinion': '4'},
            {'aspect': 'none', 'opinion': '6'}, {'opinion': '6'},
            {'opinion': 'none'},
        ])
        sent = []

        class Client:
            def ask(self, state, questions):
                sent.append((state, questions))
                answers = {}
                if next(iter(questions.values()))['type'] == 'choice':
                    choices = next(scheduled)
                    for name, value in choices.items():
                        self_test.assertIn(value, questions[name]['criteria'])
                        answers[name] = Answer(name, 'choice', {'choice': value})
                else:
                    for name, q in questions.items():
                        wording = q['instructions']
                        yes = ('"great" directly evaluate "Food"' in wording or
                               '"bad" directly evaluate "coffee"' in wording or
                               ('"lovely"' in wording and 'implicit' in wording))
                        answers[name] = Answer(name, 'noul', {'noul': float(yes)})
                return Response('jev-1.13.0', answers, {'input_tokens': 1})

        self_test = self
        result = SpanExtractor('pointer')(Client(), {
            'ID': 'sample', 'Text': 'Food great; coffee bad; lovely.',
            'Triplet': [{'Aspect': 'GOLD_SENTINEL', 'Opinion': 'DO_NOT_READ'}]})
        selected = [(p['Aspect'], p['Opinion']) for p in result['pairs'] if p['probability'] >= .5]
        self.assertEqual(selected, [('Food', 'great'), ('coffee', 'bad'), ('NULL', 'lovely')])
        self.assertNotIn('GOLD_SENTINEL', repr(sent))
        self.assertEqual(result['span_limit_hit'], [])

    def test_compact_pointer_history_is_snapshot_and_does_not_stop_at_eight(self):
        states, end_questions = [], []
        owner = self

        class Client:
            def ask(self, state, questions):
                states.append(state)
                answers = {}
                for kind, q in questions.items():
                    if kind == 'opinion':
                        value = 'none'
                    else:
                        value = min((k for k in q['criteria'] if k != 'none'), key=int)
                        owner.assertEqual(len(state['already_extracted']['aspect']), int(value))
                        if 'LAST token' in q['instructions']:
                            end_questions.append(q)
                    answers[kind] = Answer(kind, 'choice', {'choice': value})
                return Response('jev-1.13.0', answers, {'input_tokens': 1})

        words = [f'word{i}' for i in range(10)]
        result = SpanExtractor('pointer', revision=3)(Client(), {'ID': 'long', 'Text': ' '.join(words)})
        self.assertEqual(result['spans']['aspect'], words)
        self.assertEqual(states[0]['already_extracted']['aspect'], [])
        self.assertEqual(states[2]['already_extracted']['aspect'][0]['text'], 'word0')
        self.assertTrue(all(len(label.split()) <= 5 for q in end_questions for label in q['criteria'].values()))
        self.assertEqual(result['span_limit_hit'], [])


if __name__ == '__main__':
    unittest.main()
