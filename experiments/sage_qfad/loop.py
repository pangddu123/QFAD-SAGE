"""Alternate Writer GRPO, graph rebuilding, and QFAD fact-selector GRPO."""
from pathlib import Path
import tempfile

from .data import save_json, visible_documents
from .native import write_documents
from .rewards import group_advantages, trajectory_reward


class AlternatingLoop:
    def __init__(self, dataset, policy, memory_factory, evaluator, weights, out, group_size=4, seed=17):
        if group_size < 2:
            raise ValueError('GRPO needs at least two sampled trajectories')
        self.dataset, self.policy, self.factory = dataset, policy, memory_factory
        self.evaluator, self.weights = evaluator, weights
        self.out, self.group_size, self.seed = Path(out), group_size, seed
        self.counter = 0

    def next_seed(self):
        self.counter += 1
        return self.seed + 100000 * self.counter

    def update(self, role, traces, rewards):
        advantages = group_advantages(rewards)
        if not any(traces):
            return {'skipped': 'No policy action: native dense retrieval fallback', 'advantages': advantages}
        if not all(traces):
            raise ValueError('A GRPO group must have the same policy action opportunity')
        if not any(advantages):
            self.policy.discard(traces)
            return {'skipped': 'Equal rewards', 'advantages': advantages}
        return self.policy.update(role, traces, rewards)

    def train_writer(self, directory):
        """All candidate graphs are scored by the current, frozen Reader."""
        summaries = []
        for index, query in enumerate(self.dataset['queries']):
            documents = visible_documents(self.dataset, query)
            traces, rewards = [], []
            for branch in range(self.group_size):
                written = write_documents(documents, self.policy, sample=True, seed=self.next_seed())
                retrieval, answer = None, None
                task = dict(recall=0., precision=0., deducibility=0., answer_f1=0., task=0.)
                if written.legal:
                    with tempfile.TemporaryDirectory(dir=directory) as temporary:
                        memory = self.factory.build(documents, written.rows, Path(temporary) / 'graph')
                        retrieval = memory.retrieve(query['question'])
                        answer = self.evaluator.evaluate(query, retrieval)
                        task = answer['components']
                feedback = trajectory_reward(task, written.triples, written.formats,
                                             self.weights, written.legal)
                traces.append(written.trace)
                rewards.append(feedback['reward'])
                save_json(directory / f'query{index}_candidate{branch}.json', {
                    'query_id': query['id'], 'extractions': written.rows, 'raw_triples': written.triples,
                    'format_rewards': written.formats, 'error': written.error,
                    'trace': written.trace, 'retrieval': retrieval, 'answer': answer, 'feedback': feedback})
            update = self.update('writer', traces, rewards)
            summaries.append({'query_id': query['id'], 'rewards': rewards, 'update': update})
            save_json(directory / f'query{index}_update.json', summaries[-1])
        return summaries

    def rebuild(self, directory):
        """Regenerate memory with the updated Writer; never select the best sampled candidate."""
        memories = {}
        patients = sorted({q['patient'] for q in self.dataset['queries']})
        for patient in patients:
            queries = [q for q in self.dataset['queries'] if q['patient'] == patient]
            stage = max(q['stage'] for q in queries)
            documents = visible_documents(self.dataset, {'patient': patient, 'stage': stage})
            written = write_documents(documents, self.policy, sample=False, seed=0)
            save_json(directory / f'patient{patient}_extractions.json', {
                'rows': written.rows, 'legal': written.legal, 'error': written.error})
            if not written.legal:
                raise RuntimeError(f'Deterministic memory rebuild failed: {written.error}')
            for checkpoint in sorted({q['stage'] for q in queries}):
                visible = [d for d in documents if d['stage'] <= checkpoint]
                memories[patient, checkpoint] = self.factory.build(
                    visible, {d['id']: written.rows[d['id']] for d in visible},
                    directory / f'patient{patient}_stage{checkpoint}')
        return memories

    def evaluate(self, memories, directory):
        results = []
        for index, query in enumerate(self.dataset['queries']):
            retrieval = memories[query['patient'], query['stage']].retrieve(query['question'])
            result = {'query_id': query['id'], 'retrieval': retrieval,
                      **self.evaluator.evaluate(query, retrieval)}
            save_json(directory / f'query{index}.json', result)
            results.append(result)
        return results

    def train_reader(self, memories, directory):
        """Only the native fact-filter action changes; the rebuilt graph and Writer are frozen."""
        summaries = []
        for index, query in enumerate(self.dataset['queries']):
            memory = memories[query['patient'], query['stage']]
            traces, rewards = [], []
            for branch in range(self.group_size):
                retrieval = memory.retrieve(query['question'], sample=True, seed=self.next_seed())
                answer = self.evaluator.evaluate(query, retrieval)
                traces.append(retrieval['trace'])
                # Graph repetition and writing format are constant in this phase.
                rewards.append(answer['components']['task'])
                save_json(directory / f'query{index}_candidate{branch}.json', {
                    'query_id': query['id'], 'retrieval': retrieval, **answer})
            update = self.update('reader', traces, rewards)
            summaries.append({'query_id': query['id'], 'rewards': rewards, 'update': update})
            save_json(directory / f'query{index}_update.json', summaries[-1])
        return summaries

    def run(self, rounds):
        if rounds < 1:
            raise ValueError('At least one alternating round is required')
        self.out.mkdir(parents=True, exist_ok=False)
        save_json(self.out / 'dataset.json', self.dataset)
        rounds_log = []
        for index in range(rounds):
            directory = self.out / f'round{index}'
            (directory / 'writer').mkdir(parents=True)
            initial = self.policy.phase('writer')
            writer = self.train_writer(directory / 'writer')
            after_writer = self.policy.phase('rebuild')
            memories = self.rebuild(directory / 'memory')
            before_reader = self.evaluate(memories, directory / 'before_reader')
            self.policy.phase('reader')
            (directory / 'reader').mkdir()
            reader = self.train_reader(memories, directory / 'reader')
            after_reader = self.policy.phase('evaluate')
            final = self.evaluate(memories, directory / 'after_reader')
            checkpoint = self.policy.save(f'round{index}')
            rounds_log.append({'round': index, 'initial': initial,
                'after_writer': after_writer, 'after_reader': after_reader,
                'writer_updates': writer, 'reader_updates': reader,
                'before_reader_task': [r['components']['task'] for r in before_reader],
                'after_reader_task': [r['components']['task'] for r in final],
                'checkpoint': checkpoint})
            save_json(self.out / 'rounds.json', rounds_log)
        return rounds_log
