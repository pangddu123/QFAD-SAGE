"""Visible source corpora and independently annotated evidence for alternating training."""
import json
from pathlib import Path


def load_dataset(path):
    dataset = json.loads(Path(path).read_text())
    documents, queries = dataset['documents'], dataset['queries']
    by_id = {d['id']: d for d in documents}
    if not documents or len(by_id) != len(documents):
        raise ValueError('The corpus requires unique nonempty source IDs')
    if any(not d['id'] or not d['text'].strip() for d in documents):
        raise ValueError('Sources must have IDs and nonempty original text')
    if len({(d['patient'], d['text']) for d in documents}) != len(documents):
        raise ValueError('Duplicate passages must share a source ID within each patient')
    if not queries or len({q['id'] for q in queries}) != len(queries):
        raise ValueError('Queries must have unique IDs')
    for q in queries:
        if not q['question'].strip() or not q['answers'] or not all(isinstance(a, str) for a in q['answers']):
            raise ValueError(f'{q["id"]}: question and answer aliases are required')
        visible = {d['id'] for d in visible_documents(dataset, q)}
        if not q['gold_source_ids'] or not set(q['gold_source_ids']) <= visible:
            raise ValueError(f'{q["id"]}: supporting sources must be annotated, same-patient and visible')
    return dataset


def visible_documents(dataset, query):
    return [d for d in dataset['documents']
            if d['patient'] == query['patient'] and d['stage'] <= query['stage']]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
