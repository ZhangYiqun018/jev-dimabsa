import unittest
from jev.client import Answer, Response
from jev.combined_extraction import Retriever
from jev.conditioned_se import ConditionedSE, labelled_spans, stage_examples


class ConditionedSETests(unittest.TestCase):
    def test_conditioned_targets_reset_cursor_and_keep_many_to_many_and_null(self):
        # opinions can precede a target; the same opinion can evaluate two targets.
        text = 'great Food coffee bad lovely'
        sent = []
        owner = self
        class Client:
            def ask(self, state, questions):
                sent.append(state)
                stage = state['stage']
                if stage == 'aspect_start':
                    starts = [i for i in (1,2) if i >= state['cursor']]
                    answer = str(starts[0]) if starts else 'none'
                elif stage.endswith('_end'):
                    answer = str(state['given_start'])
                else:
                    expected = {'Food':[0,4], 'coffee':[0,3], 'NULL':[4]}
                    starts = [i for i in expected[state['given_aspect']] if i >= state['cursor']]
                    answer = str(starts[0]) if starts else 'none'
                owner.assertIn(answer, questions['position']['criteria'])
                return Response('jev-1.13.0', {'position': Answer('position','choice',{'choice':answer})},
                                {'input_tokens':1})
        result = ConditionedSE(Retriever([], 'eng_restaurant'))(Client(), {
            'ID':'test', 'Text':text, 'Triplet':[{'Aspect':'SECRET_GOLD','Opinion':'DO_NOT_READ'}]})
        self.assertEqual({(p['Aspect'],p['Opinion']) for p in result['pairs']},
                         {('Food','great'),('Food','lovely'),('coffee','great'),('coffee','bad'),('NULL','lovely')})
        self.assertNotIn('SECRET_GOLD', repr(sent))
        start_states = [s for s in sent if s['stage']=='opinion_start' and s['cursor']==0]
        self.assertEqual([s['given_aspect'] for s in start_states], ['Food','coffee','NULL'])
        self.assertEqual(start_states[0]['already_extracted'], [])

    def test_training_alignment_and_stage_specific_start_end_stop_examples(self):
        row = {'ID':'train', 'Text':'湯好 customer  service good good', 'Triplet':[
            {'Aspect':'湯', 'Opinion':'好'}, {'Aspect':'customer  service','Opinion':'good'}]}
        parsed = labelled_spans(row)
        self.assertEqual(parsed['aspects'][1], {'start':2,'end':3,'text':'customer  service'})
        self.assertEqual(parsed['opinions']['customer  service'][0]['start'], 4)
        start = stage_examples([row], 'opinion_start', 'service', 'eng_restaurant')[0]
        self.assertEqual(start['given_aspect'], 'customer  service')
        self.assertEqual([d['answer'] for d in start['labelled_decisions']], ['4','none'])
        end = stage_examples([row], 'opinion_end', 'service', 'eng_restaurant')[0]
        self.assertEqual((end['given_start'],end['answer']), (4,'4'))
        null = stage_examples([row], 'opinion_start', 'NULL', 'eng_restaurant')[0]
        self.assertEqual(null['labelled_decisions'], [{'cursor':0,'already_extracted':[],'answer':'none'}])


if __name__ == '__main__':
    unittest.main()
