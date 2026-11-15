"""Export original patient-1/2 passages and previously verified document evidence labels."""
import argparse
import json
from pathlib import Path
import sys

from .data import load_dataset, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root / 'scripts'), str(root / 'MedMemoryBench')]
    import pandas as pd
    import qafd_medmemorybench_mixed as harness
    from utils.templates import get_prompt_manager
    harness.DATA = harness.rawrag.DATA = root / 'MedMemoryBench/data/MedMemoryBench_EN_mixed'
    harness.LANGUAGE = harness.rawrag.LANGUAGE = 'en'
    prompt = get_prompt_manager(dataset='medmemorybench', method='embedding_rag', language='en')
    histories = harness.load_passages([1, 2])
    _, groups = harness.load_units([1, 2])
    documents = []
    for pid in (1, 2):
        frame = pd.read_parquet(root / 'results/qafd_medmemorybench_en_mixed_gpt4omini_20261111' /
            f'persona_{pid:02d}/qafd/gpt-4o-mini_qwen3-embedding-4b/chunk_embeddings/vdb_chunk.parquet')
        ids = dict(zip(frame.content, frame.hash_id))
        seen = set()
        for stage, text in histories[pid]:
            if text in seen:
                continue
            seen.add(text)
            documents.append({'id': f'p{pid}:{ids[text]}', 'patient': pid, 'stage': stage, 'text': text})
    sources = {d['id']: d for d in documents}
    plan_path = root / 'results/medical_sage/extraction_training_plan.json'
    plan = json.loads(plan_path.read_text())
    queries = {}
    for episode in plan['episodes']:
        pid = episode['patient']
        for item in episode['queries']:
            query = next(q for q in groups[pid][item['stage']][1] if q['query_id'] == item['query_id'])
            bundles = item['requirements']['bundles']
            if len(bundles) != 1:
                raise ValueError('Choose a documented gold evidence set before exporting alternative bundles')
            gold = []
            for condition in bundles[0]['conditions']:
                for anchor in condition['anchors']:
                    sid = f'p{pid}:{anchor["source_id"]}'
                    if not anchor['quote'] or anchor['quote'] not in sources[sid]['text']:
                        raise ValueError(f'{sid}: evidence quote no longer matches its original source')
                    gold.append(sid)
            qid = f'p{pid}:{query["query_id"]}'
            row = {'id': qid, 'patient': pid, 'stage': item['stage'], 'question': query['question'],
                   'answer_question': prompt.format_query(query['question'], query['query_type']),
                   'answers': [a['content'] for a in query['answers'] if a.get('is_correct')],
                   'gold_source_ids': list(dict.fromkeys(gold))}
            if qid in queries and queries[qid] != row:
                raise ValueError(f'Conflicting annotations for {qid}')
            queries[qid] = row
    missing = [f'p{pid}:{q["query_id"]}' for pid in (1, 2)
               for _, group in groups[pid] for q in group if f'p{pid}:{q["query_id"]}' not in queries]
    save_json(args.out, {'documents': documents, 'queries': list(queries.values()),
        'evidence_annotation_source': str(plan_path), 'queries_without_evidence_annotations': missing,
        'scope': 'Full patient-1/2 source corpus; only independently source-anchored questions have SAGE rewards'})
    load_dataset(args.out)
    print(json.dumps({'documents': len(documents), 'annotated_queries': len(queries),
                      'queries_without_evidence_annotations': len(missing), 'training_started': False}))


if __name__ == '__main__':
    main()
