"""The ablation must change extraction rules without changing pairing or examples."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from probe_bio_rule_rollback import ExtractionRuleRollback
from jev.span_choice import MatchedExtractor
from jev.client import Answer, Response

class RuleRollbackTest(unittest.TestCase):
    def test_only_extraction_rule_block_changes(self):
        class Fake:
            def __init__(self): self.requests=[]
            def ask(self,state,questions):
                self.requests.append((state,questions))
                answers={k: Answer(k,q['type'],{'choice':'O'} if q['type']=='choice' else {'noul':0})
                         for k,q in questions.items()}
                return Response('fake',answers,{'input_tokens':0})
        examples=[{'Text':'food is good','aligned':{'aspect':[(0,0)],'opinion':[(2,2)]}}]
        record={'ID':'test','Text':'food is good'}
        reference=Fake(); MatchedExtractor('bio',examples)(reference,record)
        actual=Fake(); block={'rules':'OLD_RULES','boundary_guidance':'OLD_GUIDANCE'}
        adapter=ExtractionRuleRollback(actual,block)
        MatchedExtractor('bio',examples)(adapter,record)
        self.assertEqual(len(actual.requests),len(reference.requests))
        for (before,bq),(after,aq),trace in zip(reference.requests,actual.requests,adapter.trace):
            self.assertEqual(bq,aq)
            expected={**before,**block} if before['stage'].startswith('extract_') else before
            self.assertEqual(after,expected)
            self.assertEqual(trace['state'],after)
            self.assertEqual(trace['questions'],aq)
if __name__=='__main__':unittest.main()
