from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .data import load_dataset, save_json
from .evaluation import TaskEvaluator
from .loop import AlternatingLoop
from .rewards import RewardWeights
from .worker import PhaseState


class PolicyFixture:
    def __init__(self):
        self.state = PhaseState()
        self.events, self.calls = [], []
        self.counter = 0

    def phase(self, phase):
        self.state.set_phase(phase)
        self.events.append((phase, dict(self.state.steps)))
        return {'steps': dict(self.state.steps)}

    def generate(self, role, messages, sample, seed, max_tokens):
        self.counter += 1
        self.calls.append((role, messages, sample, dict(self.state.steps)))
        action = None
        if sample:
            self.state.require_trainable(role)
            key = str(self.counter)
            self.state.groups[key] = {'role': role, 'step': self.state.steps[role]}
            action = {'group': key, 'row': 0}
        if role == 'writer':
            if '"named_entities"' in messages[-1]['content']:
                value = self.state.steps['writer'] if not sample else (seed // 100000) % 2
                text = json.dumps({'triples': [['patient', 'has', str(value)]]})
            else:
                text = '{"named_entities":["patient"]}'
        else:
            text = str((seed // 100000) % 2 if sample else self.state.steps['reader'])
        return {'text': text, 'finished': True}, action

    def discard(self, traces):
        for trace in traces:
            for action in trace:
                del self.state.groups[action['group']]

    def update(self, role, traces, rewards):
        self.state.trajectories(role, traces)
        self.state.steps[role] += 1
        self.discard(traces)
        return {'steps': dict(self.state.steps)}

    def save(self, name):
        return {'name': name, 'steps': dict(self.state.steps)}


class FactoryFixture:
    def __init__(self, policy):
        self.policy, self.builds, self.reads = policy, [], []

    def build(self, documents, rows, directory):
        self.builds.append((deepcopy(documents), deepcopy(rows), dict(self.policy.state.steps)))
        writer_value = float(rows[documents[0]['id']]['extracted_triples'][0][2])
        factory = self
        class Memory:
            def retrieve(self, question, sample=False, seed=0):
                response, trace = factory.policy.generate('reader', [{'role': 'user', 'content': question}],
                    sample=sample, seed=seed, max_tokens=512)
                factory.reads.append((factory.policy.state.phase, dict(factory.policy.state.steps)))
                value = float(response['text']) if sample else writer_value
                return {'documents': ['source'], 'source_ids': ['a'], 'trace': [trace] if trace else [],
                        'fixture_reward': value}
        return Memory()


class EvaluatorFixture:
    def evaluate(self, query, retrieval):
        return {'components': {'task': retrieval['fixture_reward']}}


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.weights = RewardWeights(1, 1, 1, 1, 0, 0)
        self.dataset = {'documents': [
            {'id': 'a', 'patient': 1, 'stage': 0, 'text': 'Patient source.'},
            {'id': 'future', 'patient': 1, 'stage': 1, 'text': 'Future source.'},
            {'id': 'other', 'patient': 2, 'stage': 0, 'text': 'Other patient source.'}],
            'queries': [{'id': 'q', 'patient': 1, 'stage': 0, 'question': 'Question?',
                         'answers': ['PRIVATE_REFERENCE'], 'gold_source_ids': ['a']}]}

    def tearDown(self):
        self.directory.cleanup()

    def test_two_rounds_freeze_rebuild_and_use_updated_reader(self):
        policy = PolicyFixture()
        factory = FactoryFixture(policy)
        loop = AlternatingLoop(self.dataset, policy, factory, EvaluatorFixture(), self.weights,
                               self.root / 'run', group_size=2)
        rounds = loop.run(2)
        self.assertEqual([phase for phase, _ in policy.events],
                         ['writer', 'rebuild', 'reader', 'evaluate'] * 2)
        self.assertEqual(rounds[0]['after_writer']['steps'], {'writer': 1, 'reader': 0})
        self.assertEqual(rounds[0]['after_reader']['steps'], {'writer': 1, 'reader': 1})
        self.assertEqual(rounds[1]['initial']['steps'], {'writer': 1, 'reader': 1})
        self.assertEqual(rounds[1]['after_reader']['steps'], {'writer': 2, 'reader': 2})
        self.assertTrue(any(phase == 'writer' and steps['reader'] == 1 for phase, steps in factory.reads))
        for docs, rows, steps in factory.builds:
            self.assertEqual([d['id'] for d in docs], ['a'])
        writer_messages = [str(messages) for role, messages, _, _ in policy.calls if role == 'writer']
        self.assertTrue(all('PRIVATE_REFERENCE' not in m and 'Question?' not in m and 'Future source' not in m
                            for m in writer_messages))
        self.assertFalse(policy.state.groups)
        rebuilt = json.loads((self.root / 'run/round0/memory/patient1_extractions.json').read_text())
        self.assertEqual(rebuilt['rows']['a']['extracted_triples'][0][2], '1')

    def test_missing_or_future_gold_stops_before_worker(self):
        for bad in ([], ['future'], ['other']):
            data = deepcopy(self.dataset)
            data['queries'][0]['gold_source_ids'] = bad
            save_json(self.root / 'input.json', data)
            with self.assertRaises(ValueError):
                load_dataset(self.root / 'input.json')

    def test_phase_state_prevents_other_role_updates_and_stale_actions(self):
        state = PhaseState()
        state.set_phase('writer')
        with self.assertRaises(ValueError):
            state.require_trainable('reader')
        with self.assertRaises(ValueError):
            state.require_trainable('answer')
        state.groups['a'] = {'role': 'writer', 'step': 0}
        with self.assertRaises(ValueError):
            state.set_phase('reader')
        state.steps['writer'] = 1
        with self.assertRaises(ValueError):
            state.trajectories('writer', [[{'group': 'a', 'row': 0}]])

    def test_equal_rewards_skip_optimizer(self):
        policy = PolicyFixture()
        policy.phase('writer')
        loop = AlternatingLoop(self.dataset, policy, None, None, self.weights, self.root / 'unused', group_size=2)
        traces = []
        for seed in (1, 2):
            _, action = policy.generate('writer', [{'role': 'user', 'content': 'passage'}],
                                         sample=True, seed=seed, max_tokens=10)
            traces.append([action])
        self.assertIn('skipped', loop.update('writer', traces, [1., 1.]))
        self.assertEqual(policy.state.steps['writer'], 0)
        self.assertFalse(policy.state.groups)

    def test_answer_model_does_not_receive_reference_and_deducibility_is_separate(self):
        calls = []
        class AnswerPolicy:
            def generate(self, role, messages, **kwargs):
                calls.append((role, messages))
                return {'text': 'Answer: PRIVATE_REFERENCE', 'finished': True}, None
        class Judge:
            def responses(self, messages, **kwargs):
                calls.append(('judge', messages))
                return SimpleNamespace(content='{"deducible":false,"reason":"not in evidence"}', usage={})
        evaluator = TaskEvaluator(AnswerPolicy(), Judge(), self.weights)
        result = evaluator.evaluate(self.dataset['queries'][0], {'documents': ['source'], 'source_ids': ['a']})
        self.assertEqual(calls[0][0], 'answer')
        self.assertNotIn('PRIVATE_REFERENCE', str(calls[0][1]))
        self.assertNotIn('prediction', str(calls[1][1]))
        self.assertEqual(result['components']['answer_f1'], 1.)
        self.assertEqual(result['components']['deducibility'], 0.)


if __name__ == '__main__':
    unittest.main()
