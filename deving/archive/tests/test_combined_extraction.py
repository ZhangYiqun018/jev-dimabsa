import unittest

from jev.client import Answer, Response
from jev.combined_extraction import CombinedExtractor, Retriever, candidates, demonstrations


class CombinedTests(unittest.TestCase):
    def test_candidate_offsets_preserve_spaces_unicode_and_repeated_occurrences(self):
        text = '湯好 customer  service 好 customer  service'
        spans = candidates(text)
        for span in spans:
            for start, end in span['offsets']:
                self.assertEqual(text[start:end].lower(), span['text'].lower())
        repeated = next(s for s in spans if s['text'] == 'customer  service')
        self.assertEqual(len(repeated['offsets']), 2)
        self.assertIn('湯好', [s['text'] for s in spans])
        self.assertTrue(any(len(s['offsets']) == 2 and s['text'] == '好' for s in spans))
        self.assertEqual(len({s['id'] for s in spans}), len(spans))

    def test_retrieval_excludes_query_and_demo_labels_follow_all_annotations(self):
        query = {'ID': 'query', 'Text': 'Food great', 'Triplet': [{'Aspect': 'Food', 'Opinion': 'great'}]}
        row = {'ID': 'train', 'Text': 'Food great; coffee bad.', 'Triplet': [
            {'Aspect': 'Food', 'Opinion': 'great'}, {'Aspect': 'coffee', 'Opinion': 'bad'}]}
        excluded = {'ID': 'excluded', 'Text': 'Food  great!', 'Triplet': []}
        retriever = Retriever([query, row, excluded], 'eng_restaurant', [' food great! '])
        self.assertEqual([r['ID'] for r in retriever.select('Food great')], ['train'])
        demo = demonstrations(row)
        gold = {k: {a[k.title()].lower() for a in row['Triplet']} for k in ('aspect', 'opinion')}
        pairs = {(a['Aspect'].lower(), a['Opinion'].lower()) for a in row['Triplet']}
        for d in demo['screen']:
            self.assertEqual(d['answer'], d['span'].lower() in gold[d['kind']])
        for d in demo['relations']:
            self.assertEqual(d['answer'], (d['aspect'].lower(), d['opinion'].lower()) in pairs)
        self.assertEqual(len(demo['screen']), 4)
        self.assertEqual(len(demo['relations']), 4)

    def test_pipeline_preserves_many_to_many_and_null_without_query_gold(self):
        owner = self
        seen = []
        expected = {('Food', 'great'), ('Food', 'lovely'), ('coffee', 'bad'), ('NULL', 'lovely')}

        class Client:
            def ask(self, state, questions):
                seen.append(state)
                owner.assertLessEqual(len(questions), 48)
                answers = {}
                for key, question in questions.items():
                    info = question['instructions']
                    if state['stage'] == 'span_screen':
                        text = state['candidates'][info['candidate']]['text']
                        yes = text in ({'Food', 'coffee'} if info['kind'] == 'aspect'
                                       else {'great', 'bad', 'lovely'})
                    else:
                        text = state['opinions'][info['opinion_candidate']]['text']
                        yes = (state['aspect']['text'], text) in expected
                    answers[key] = Answer(key, 'noul', {'noul': .9 if yes else .1})
                return Response('jev-1.13.0', answers, {'input_tokens': 1})

        result = CombinedExtractor(Retriever([], 'eng_restaurant'))(Client(), {
            'ID': 'query', 'Text': 'Food great; coffee bad; lovely.',
            'Triplet': [{'Aspect': 'SECRET_GOLD', 'Opinion': 'NEVER_READ'}]})
        predicted = {(p['Aspect'], p['Opinion']) for p in result['pairs'] if p['probability'] >= .65}
        self.assertEqual(predicted, expected)
        self.assertEqual(len(result['pairs']), 9)
        self.assertNotIn('SECRET_GOLD', repr(seen))
        self.assertEqual(len(result['spans']['aspect']), 2)


if __name__ == '__main__':
    unittest.main()
