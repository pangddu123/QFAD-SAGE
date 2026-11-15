"""Validate input locally, or explicitly run the alternating QFAD/SAGE loop."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .data import load_dataset, save_json
from .rewards import RewardWeights


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Offline dataset validation; no model or API calls')
    validate.add_argument('--dataset', type=Path, required=True)
    train = commands.add_parser('train')
    train.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'validate':
        dataset = load_dataset(args.dataset)
        print(json.dumps({'documents': len(dataset['documents']), 'queries': len(dataset['queries']),
                          'patients': sorted({q['patient'] for q in dataset['queries']}),
                          'training_started': False}))
        return
    config = json.loads(args.config.read_text())
    dataset = load_dataset(config['dataset'])
    weights = RewardWeights(**config['reward_weights'])
    if config['rounds'] < 1 or config['group_size'] < 2:
        raise ValueError('Positive rounds and group_size >= 2 are required')
    if not isinstance(config['worker_command'], list) or not all(isinstance(s, str) for s in config['worker_command']):
        raise ValueError('worker_command must be an argv list')
    out = Path(config['out'])
    if out.exists() or out.with_name(out.name + '_worker').exists():
        raise ValueError('Use a new output directory')
    from openai import OpenAI
    from passage_entity.config import PassageEntityConfig
    from passage_entity.embedding_store import EmbeddingModelWrapper
    from .evaluation import TaskEvaluator
    from .loop import AlternatingLoop
    from .native import MemoryFactory
    from .policy import WorkerPolicy
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
    from medmemory_scoring import JudgeClient
    judge = JudgeClient(config['judge_model'])
    embedding_api = OpenAI(base_url=config['embedding_url'], api_key='local', timeout=600)
    async def embed(texts):
        response = embedding_api.embeddings.create(model=config['embedding_model'], input=texts)
        return [row.embedding for row in sorted(response.data, key=lambda row: row.index)]
    embedding = EmbeddingModelWrapper(embed)
    native_config = PassageEntityConfig(llm_model='Qwen3.8-27B', embedding_model_key=config['embedding_model'])
    policy = WorkerPolicy(config['worker_command'], out.with_name(out.name + '_worker'))
    try:
        factory = MemoryFactory(native_config, embedding, policy)
        evaluator = TaskEvaluator(policy, judge, weights)
        loop = AlternatingLoop(dataset, policy, factory, evaluator, weights, out,
                               group_size=config['group_size'], seed=config.get('seed', 17))
        # The full, resolved configuration is recorded even if the first group fails.
        save_json(policy.log_dir / 'configuration.json', {
            **config, 'reward_weights': asdict(weights), 'qfad': asdict(native_config), 'worker': policy.ready})
        results = loop.run(config['rounds'])
        save_json(out / 'complete.json', {'rounds': len(results), 'steps': results[-1]['after_reader']['steps']})
    finally:
        policy.close()


if __name__ == '__main__':
    main()
