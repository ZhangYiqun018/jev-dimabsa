import unittest
from jev.client import Answer,Response
from jev.span_choice import MatchedExtractor,options,demonstrations
from jev.extraction import tokenize

class SpanChoiceTests(unittest.TestCase):
    def test_anchor_continuation_overlap_null_and_gold_isolation(self):
        class Fake:
            def __init__(self):self.asked=[]
            def ask(self,state,questions):
                self.asked.append((state,questions));answers={}
                for key,q in questions.items():
                    if q['type']=='noul':raw=.9
                    else:
                        info=q['instructions'];role=info['role'];s=info['start_token']
                        if role=='aspect' and s==0:raw='1' if '1' in q['criteria'] else ('0' if '0' in q['criteria'] else 'none')
                        elif role=='aspect' and s==1:raw='1' if not info['already_selected'] else 'none'
                        elif role=='opinion' and s==2:raw='2' if not info['already_selected'] else 'none'
                        else:raw='none'
                    # Match API answer shape.
                    answers[key]=Answer(key,q['type'],{q['type']:raw})
                return Response('fake',answers,{'input_tokens':0})
        fake=Fake()
        pred=MatchedExtractor('choice',[])(fake,{'ID':'x','Text':'food options rule','Triplet':'SECRET_GOLD'})
        self.assertEqual(pred['spans']['aspect'],['food options','options','food'])
        self.assertEqual(pred['spans']['opinion'],['rule'])
        self.assertIn(('food','NULL'),[(p['Aspect'],p['Opinion']) for p in pred['pairs']])
        self.assertNotIn('SECRET_GOLD',str(fake.asked))
        self.assertTrue(any(q['instructions'].get('already_selected')==['food options','food'] for s,qs in fake.asked for q in qs.values() if q['type']=='choice'))

    def test_offsets_and_demo_decision_targets(self):
        text='好吃，好吃';ts=tokenize(text)
        self.assertEqual(options(text,ts,3)['4'],'好吃')
        rows=[{'Text':'food is very good','aligned':{'aspect':[(0,0)],'opinion':[(2,3)]}}]
        bio=demonstrations(rows,'opinion','bio')[0]
        self.assertEqual(bio['labels'],['O','O','B','I'])
        choice=demonstrations(rows,'opinion','choice')[0]['decisions']
        self.assertEqual(choice[0]['choices'][choice[0]['answer']],'very good')
        self.assertEqual(choice[1]['answer'],'none')
if __name__=='__main__':unittest.main()
