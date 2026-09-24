import unittest

from tools.probe_aspect_count import gold_count


class AspectCountTests(unittest.TestCase):
    def test_counts_aspects_not_triplets_mentions_or_implicit_targets(self):
        self.assertEqual(gold_count({'Triplet': [
            {'Aspect': 'battery life', 'Opinion': 'great'},
            {'Aspect': 'Battery Life', 'Opinion': 'long'},
            {'Aspect': 'screen', 'Opinion': 'clear'},
            {'Aspect': 'NULL', 'Opinion': 'satisfied'},
        ]}), 2)
        self.assertEqual(gold_count({'Quadruplet': [
            {'Aspect': 'food', 'Opinion': 'NULL'},
            {'Aspect': 'NULL', 'Opinion': 'NULL'},
        ]}), 1)


if __name__ == '__main__':
    unittest.main()
