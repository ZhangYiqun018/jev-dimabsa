import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from probe_bio_bm25_50 import RetrievedExamples, demonstration
from jev.extraction import SpanExtractor
from jev.client import Answer, Response
from jev.retrieval_variants import Retriever, retrieval_terms


class RetrievalVariantsTest(unittest.TestCase):
    def test_tokenization_ranking_and_exclusions(self):
        self.assertEqual(retrieval_terms('Ab cD', 'eng_restaurant', 'bigram'), ['ab', 'bc', 'cd'])
        self.assertEqual(retrieval_terms('Ab cD', 'eng_restaurant', 'trigram'), ['abc', 'bcd'])
        self.assertEqual(retrieval_terms('Good FOOD!', 'eng_restaurant', 'word'), ['good', 'food!'])
        self.assertEqual(retrieval_terms('服务很好', 'zho_restaurant', 'word'), ['服务', '很', '好'])
        rows = [{'ID':'z','Text':'room nice'}, {'ID':'a','Text':'nice room'},
                {'ID':'b','Text':'room nice'}, {'ID':'c','Text':'excluded'},
                {'ID':'d','Text':'query'}, {'ID':'e','Text':'unrelated'}]
        for variant in ('bigram', 'trigram', 'word'):
            retriever = Retriever(rows, 'eng_restaurant', [' EXCLUDED '], variant)
            result = retriever.select('query', 50)
            self.assertEqual([r['ID'] for r in result], ['a','e','z'])
            self.assertEqual(result, retriever.select('query', 50))
        retriever = Retriever(rows, 'eng_restaurant', (), 'word')
        self.assertEqual([r['ID'] for r in retriever.select('nice', 2)], ['a','z'])

    def test_only_example_package_changes(self):
        train=[{'ID':'a','Text':'good soup','Triplet':[{'Aspect':'soup','Opinion':'good','VA':'SECRET'}]},
               {'ID':'b','Text':'bad soup','Triplet':[{'Aspect':'soup','Opinion':'bad'}]},
               {'ID':'c','Text':'clean room','Triplet':[{'Aspect':'room','Opinion':'clean'}]}]
        selected=Retriever(train,'eng_restaurant',[' GOOD SOUP '], 'word').select('tasty soup',2)
        self.assertEqual([r['ID'] for r in selected],['b','c'])
        demos=[demonstration(r) for r in selected]
        self.assertEqual(set(demos[0]),{'review','tokens','aspects','opinions'})
        class Fake:
            def __init__(self):self.requests=[]
            def ask(self,state,questions):
                self.requests.append((state,questions))
                return Response('fake',{k:Answer(k,q['type'],{'choice':'B'} if q['type']=='choice' else {'noul':0}) for k,q in questions.items()},{'input_tokens':0})
        reference=Fake();actual=Fake();query={'ID':'q','Text':'tasty soup','Triplet':'QUERY_GOLD_SECRET'}
        SpanExtractor('bio',revision=3)(reference,query)
        adapter=RetrievedExamples(actual,demos);SpanExtractor('bio',revision=3)(adapter,query)
        self.assertEqual(len(reference.requests),len(actual.requests))
        for (state,qs),(sent,sqs),trace in zip(reference.requests,actual.requests,adapter.trace):
            expected=dict(state)
            if 'invented_examples' in expected:
                del expected['invented_examples'];expected['examples']=demos
            self.assertEqual(sent,expected);self.assertEqual(qs,sqs);self.assertEqual(trace['state'],sent)
        self.assertNotIn('QUERY_GOLD_SECRET',str(actual.requests))
        self.assertNotIn('VA',str(demos))

if __name__ == '__main__':
    unittest.main()
