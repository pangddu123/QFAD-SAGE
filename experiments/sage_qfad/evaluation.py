"""Frozen answer generation and evidence-only deducibility judging."""
import json
import re

from passage_entity.prompts import make_qa_messages
from .rewards import task_reward


DEDUCIBILITY_PROMPT = '''Decide whether the supplied evidence alone is sufficient to derive
an acceptable reference answer to the question. Do not use outside knowledge or treat the
reference answer as evidence. Return JSON with a boolean "deducible" and a short "reason".'''


class TaskEvaluator:
    def __init__(self, policy, judge, weights, answer_max_tokens=2048):
        self.policy, self.judge, self.weights = policy, judge, weights
        self.answer_max_tokens = answer_max_tokens
        self.cache = {}

    def evaluate(self, query, retrieval):
        key = (query['question'], query.get('answer_question', query['question']),
               tuple(query['answers']), tuple(retrieval['documents']))
        if key not in self.cache:
            response, _ = self.policy.generate('answer', make_qa_messages(
                retrieval['documents'], query.get('answer_question', query['question'])),
                sample=False, seed=0, max_tokens=self.answer_max_tokens)
            if not response['finished']:
                raise RuntimeError('Answer completion reached its output limit')
            text = response['text'].strip()
            markers = list(re.finditer(r'(?:^|\n)\s*(?:final\s+)?answer\s*:\s*', text, re.I))
            prediction = text[markers[-1].end():].strip() if markers else text
            verdict = self.judge.responses(messages=[{'role': 'system', 'content': DEDUCIBILITY_PROMPT},
                {'role': 'user', 'content': json.dumps({'question': query['question'],
                 'reference_answers': query['answers'], 'evidence': retrieval['documents']}, ensure_ascii=False)}],
                max_tokens=4096, reasoning_effort='high')
            decision = json.loads(verdict.content)
            if type(decision['deducible']) is not bool:
                raise ValueError('Deducibility must be a boolean')
            self.cache[key] = {'prediction': prediction, 'response': text,
                               'deducible': decision['deducible'], 'judge': decision,
                               'answer_usage': response.get('usage'), 'judge_usage': verdict.usage}
        answer = self.cache[key]
        return {**answer, 'components': task_reward(retrieval['source_ids'], query['gold_source_ids'],
            answer['deducible'], answer['prediction'], query['answers'], self.weights)}
