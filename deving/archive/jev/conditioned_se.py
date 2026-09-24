"""Direct aspect -> opinion start/end extraction, with no separate pair filter."""
from .combined_extraction import RULES, retrieval_terms
from .data import _annotation_items
from .extraction import tokenize, position_options


CONFIG = {'shots': 3, 'max_tokens': 254, 'context_radius': 2,
          'pair_filter': False, 'null_aspect': True, 'null_opinion': False}


def labelled_spans(row):
    """Train has surface labels, not occurrence offsets: use first aligned occurrence."""
    text, tokens = row['Text'], tokenize(row['Text'])
    ends = {t.end: i for i, t in enumerate(tokens)}

    def locate(surface):
        if surface.upper() == 'NULL':
            return None
        for i, token in enumerate(tokens):
            end = token.start + len(surface)
            if end in ends and text[token.start:end].lower() == surface.lower():
                return {'start': i, 'end': ends[end], 'text': text[token.start:end]}
        return None

    aspects, opinions = {}, {}
    for item in _annotation_items(row):
        a, o = item['Aspect'], item['Opinion']
        aspect = locate(a)
        if aspect:
            aspects.setdefault(a.lower(), aspect)
        opinion = locate(o)
        if opinion and (aspect or a.upper() == 'NULL'):
            opinions.setdefault(a.lower(), {})[o.lower()] = opinion
    return {'review': text, 'tokens': '\n'.join(f'{i}|{t.text}' for i, t in enumerate(tokens)),
            'aspects': sorted(aspects.values(), key=lambda x: (x['start'], x['end'])),
            'opinions': {k: sorted(v.values(), key=lambda x: (x['start'], x['end']))
                         for k, v in opinions.items()}}


def stage_examples(rows, stage, target, corpus):
    examples = []
    for number, row in enumerate(rows):
        parsed = labelled_spans(row)
        spans = parsed['aspects']
        example = {'review': parsed['review'], 'tokens': parsed['tokens']}
        if stage.startswith('opinion'):
            if target == 'NULL':
                aspect = 'NULL'
            elif parsed['aspects']:
                terms = set(retrieval_terms(target, corpus))
                aspect = max(parsed['aspects'], key=lambda s:
                             len(terms & set(retrieval_terms(s['text'], corpus))))['text']
            else:
                continue
            spans = parsed['opinions'].get(aspect.lower(), [])
            example['given_aspect'] = aspect
        if stage.endswith('start'):
            decisions = []
            if spans:
                i = number % len(spans)
                previous = spans[:i]
                cursor = previous[-1]['end'] + 1 if previous else 0
                # A monotonic demonstration cannot represent overlapping targets.
                if spans[i]['start'] >= cursor:
                    decisions.append({'cursor': cursor, 'already_extracted': previous,
                                      'answer': str(spans[i]['start'])})
            decisions.append({'cursor': max((s['end'] for s in spans), default=-1) + 1,
                              'already_extracted': spans, 'answer': 'none'})
            example['labelled_decisions'] = decisions
        else:
            if not spans:
                continue
            span = spans[number % len(spans)]
            example.update(given_start=span['start'], answer=str(span['end']), answer_text=span['text'])
        examples.append(example)
    return examples


class ConditionedSE:
    def __init__(self, retriever):
        self.retriever = retriever

    def __call__(self, client, record):
        text = record['Text']
        tokens = tokenize(text)
        if len(tokens) > CONFIG['max_tokens']:
            raise ValueError('Trial-only extractor supports at most 254 tokens; does not truncate')
        rows = self.retriever.select(text, CONFIG['shots'])
        trace, pairs, aspects, all_opinions = [], [], [], []
        token_table = '\n'.join(f'{i}|{t.text}' for i, t in enumerate(tokens))

        def ask(stage, history, cursor, aspect=None, start=None):
            examples = stage_examples(rows, stage, aspect, self.retriever.corpus)
            state = {'target_review': text, 'tokens': token_table, 'rules': RULES,
                     'stage': stage, 'examples': examples, 'already_extracted': list(history),
                     'cursor': cursor}
            kind = 'opinion' if aspect is not None else 'aspect'
            target_rule = ''
            if aspect is not None:
                state['given_aspect'] = aspect
                target_rule = (' Extract ONLY opinions whose target is implicit, with no explicit '
                               'aspect phrase anywhere in this review.' if aspect == 'NULL' else
                               ' Extract ONLY opinions that evaluate given_aspect; ignore opinions '
                               'about other targets. The opinion can appear before or after the aspect.')
            if start is None:
                instructions = (f'Choose the FIRST token of the leftmost remaining explicit {kind} '
                                'at or after cursor. Choose none if none remains. Do not repeat '
                                'already extracted occurrences. A sentiment expression may span several tokens.'
                                + target_rule)
                criteria = {**position_options(tokens, cursor, CONFIG['context_radius']),
                            'none': 'No remaining matching span'}
            else:
                state['given_start'] = start
                instructions = (f'The {kind} starts at given_start. Choose its LAST token, '
                                'matching the complete phrase boundaries demonstrated in examples.' + target_rule)
                criteria = position_options(tokens, start, CONFIG['context_radius'])
            response = client.ask(state, {'position': {'type': 'choice', 'instructions': instructions,
                                                     'criteria': criteria}})
            trace.append({'stage': stage, 'usage': response.usage, 'model': response.model,
                          'attempts': response.attempts, 'example_count': len(examples)})
            return response.answers['position'].choice

        def next_span(kind, history, cursor, aspect=None):
            start = ask(kind+'_start', history, cursor, aspect)
            if start == 'none':
                return None
            s = int(start)
            e = int(ask(kind+'_end', history, cursor, aspect, s))
            return {'start': s, 'end': e, 'text': text[tokens[s].start:tokens[e].end]}

        def extract_opinions(aspect):
            history, cursor = [], 0  # Reset for EVERY target, including shared opinions.
            while cursor < len(tokens):
                span = next_span('opinion', history, cursor, aspect)
                if span is None:
                    break
                history.append(span)
                all_opinions.append(span)
                pairs.append({'Aspect': aspect, 'Opinion': span['text'], 'probability': 1.})
                cursor = span['end'] + 1

        cursor = 0
        while cursor < len(tokens):
            span = next_span('aspect', aspects, cursor)
            if span is None:
                break
            aspects.append(span)
            extract_opinions(span['text'])
            cursor = span['end'] + 1
        extract_opinions('NULL')
        unique_pairs = {(p['Aspect'].lower(), p['Opinion'].lower()): p for p in pairs}
        return {'ID': record['ID'], 'spans': {
                    'aspect': list(dict.fromkeys(s['text'] for s in aspects)),
                    'opinion': list(dict.fromkeys(s['text'] for s in all_opinions))},
                'pairs': list(unique_pairs.values()), 'trace': trace,
                'extracted_positions': {'aspect': aspects, 'opinion': all_opinions},
                'example_ids': [r['ID'] for r in rows], 'span_limit_hit': [], 'chunked': False}
