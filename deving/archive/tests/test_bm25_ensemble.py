import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ensemble_bio_bm25_50 import agreement


class AgreementTest(unittest.TestCase):
    def test_distinct_arms_threshold_and_pair_identity(self):
        def pair(a, o, score=.65):
            return {'Aspect': a, 'Opinion': o, 'probability': score}
        arms = [
            {'pairs': [pair('Food','Good'), pair('food','good'), pair('NULL','nice'), pair('solo','fine')]},
            {'pairs': [pair('food','good'), pair('null','nice'), pair('solo','fine',.649), pair('room','clean')]},
            {'pairs': [pair('food','good'), pair('room','nice'), pair('food','very good')]},
        ]
        votes = agreement(arms)
        self.assertEqual(votes['food','good'], 3)
        self.assertEqual(votes['solo','fine'], 1)
        self.assertEqual({p for p,n in votes.items() if n >= 2}, {('food','good'),('null','nice')})

if __name__ == '__main__':
    unittest.main()
