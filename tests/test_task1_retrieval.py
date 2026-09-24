import unittest

from jev.retrieval import Retriever
from tools.run_task1 import retrieved_examples


def row(i, text, aspect, va='7#6'):
    return {'ID': i, 'Text': text, 'Quadruplet': [{'Aspect': aspect, 'Opinion': 'x', 'Category': 'A#B', 'VA': va}]}


class RetrievedExampleTests(unittest.TestCase):
    def test_examples_skip_the_review_itself_banned_texts_and_null_aspects(self):
        train = [row('t1', 'The soup was good.', 'soup'), row('t2', 'The soup was good here.', 'NULL'),
                 row('t3', 'The soup was very good.', 'soup'), row('t4', 'Soup in the dev split.', 'soup')]
        retriever = Retriever(train, banned_texts=['Soup in the dev split.'])
        record = {'ID': 'd', 'Text': 'The soup was good.', 'Aspect_VA': [{'Aspect': 'soup', 'VA': '0#0'}]}
        examples = retrieved_examples(retriever, record, 'eng_restaurant').examples
        self.assertEqual([e.source_id for e in examples], ['t3'])


if __name__ == '__main__':
    unittest.main()
