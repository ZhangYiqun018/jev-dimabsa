import unittest

from jev.categories import CategoryChooser, Lookup
from jev.client import Answer, Response
from jev.retrieval import Retriever


class FakeClient:
    def __init__(self):
        self.sent = []

    def ask(self, state, questions):
        self.sent.append({'state': state, 'questions': questions})
        return Response('jev-1.13.0', {k: Answer(k, 'choice', {'choice': 'FOOD#QUALITY',
                                                               'probabilities': {'FOOD#QUALITY': 1.}})
                                       for k in questions}, {'input_tokens': 1})


def row(i, text, aspect, category):
    return {'ID': i, 'Text': text,
            'Quadruplet': [{'Aspect': aspect, 'Opinion': 'good', 'Category': category, 'VA': '7#6'}]}


TRAIN = (row('t1', 'The soup was good.', 'soup', 'FOOD#QUALITY'),
         row('t2', 'The soup was good here.', 'soup', 'FOOD#STYLE_OPTIONS'))


class CategoryTests(unittest.TestCase):
    def test_state_holds_no_annotation_of_the_record_nor_its_own_text(self):
        record = row('t1', 'The soup was good.', 'GOLD_SENTINEL', 'DO_NOT_READ')
        client = FakeClient()
        CategoryChooser()(client, record, [('soup', 'good')], Retriever(list(TRAIN)),
                          ['FOOD#QUALITY', 'FOOD#STYLE_OPTIONS'], {}, exclude=record['Text'])
        self.assertNotIn('GOLD_SENTINEL', repr(client.sent))
        examples = client.sent[0]['state']['annotation_examples']
        self.assertEqual([e['review'] for e in examples], ['The soup was good here.'])

    def test_leave_one_out_lookup_drops_the_record_own_counts(self):
        categories = ['FOOD#QUALITY', 'FOOD#STYLE_OPTIONS']
        lookup = Lookup(TRAIN)
        full = lookup.features('Soup', 'good', categories)
        loo = lookup.features('Soup', 'good', categories, without='The soup was good.')
        self.assertAlmostEqual(full[0][0], full[1][0])  # One count each.
        self.assertLess(loo[0][0], loo[1][0])            # Only the other record's category remains.
        self.assertLess(loo[0][3], loo[1][3])            # Same for the opinion lookup.


if __name__ == '__main__':
    unittest.main()
