"""Native passage-entity indexing and fact-filter-before-flow retrieval."""
from dataclasses import dataclass, replace
import json
from pathlib import Path

from passage_entity.config import PassageEntityConfig
from passage_entity.kg_builder import KGBuilder
from passage_entity.prompts import make_ner_messages, make_triple_messages
from passage_entity.reranker import FactReranker
from passage_entity.retriever import PassageEntityRetriever
from passage_entity.utils import NerRawOutput, TripleRawOutput, filter_invalid_triples, fix_broken_generated_json


def parse_writer_output(text, field):
    """Repair JSON, then remove malformed individual items without discarding good ones."""
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    data = json.loads(fix_broken_generated_json(text))
    if not isinstance(data, dict) or not isinstance(data.get(field), list):
        raise ValueError(f'Expected an object with a {field} list')
    if field == 'named_entities':
        return list(dict.fromkeys(x for x in data[field] if isinstance(x, str) and x.strip()))
    return [item for item in data[field] if isinstance(item, list) and len(item) == 3
            and all(isinstance(x, str) and x.strip() for x in item)]


@dataclass
class WriteResult:
    rows: dict
    triples: list
    formats: list
    trace: list
    legal: bool = True
    error: str = ''


def write_documents(documents, policy, sample, seed):
    """The actor sees only the native NER/triple prompts for the current passage."""
    result = WriteResult({}, [], [], [])
    for index, doc in enumerate(documents):
        row = {'passage': doc['text']}
        for offset, field in enumerate(('named_entities', 'triples')):
            messages = (make_ner_messages(doc['text']) if field == 'named_entities'
                        else make_triple_messages(doc['text'], row['extracted_entities']))
            response, action = policy.generate('writer', messages, sample=sample,
                                               seed=seed + index * 2 + offset, max_tokens=2048)
            if action is not None:
                result.trace.append(action)
            try:
                items = parse_writer_output(response['text'], field)
            except (ValueError, TypeError) as error:
                result.legal, result.error = False, f'{doc["id"]}: {error}'
                return result
            if field == 'named_entities':
                row['extracted_entities'] = items
            else:
                result.triples.extend(items)
                result.formats.append(1.)
                row['extracted_triples'] = filter_invalid_triples(items)
        result.rows[doc['id']] = row
    return result


class RecordedOpenIE:
    """Supply sampled native OpenIE results to KGBuilder without generating a second graph."""
    def __init__(self, documents, rows):
        self.by_text = {d['text']: rows[d['id']] for d in documents}

    def batch_openie(self, chunks):
        ner, triples = {}, {}
        for key, chunk in chunks.items():
            row = self.by_text[chunk['content']]
            ner[key] = NerRawOutput(key, '', row['extracted_entities'], {})
            triples[key] = TripleRawOutput(key, '', row['extracted_triples'], {})
        return ner, triples


class PolicyFactReranker(FactReranker):
    """Train the LLM call already present in QFAD's fact reranker."""
    def __init__(self, policy):
        super().__init__(None)
        self.policy = policy
        self.sample, self.seed, self.trace, self.failure = False, 0, [], None

    def _call_llm(self, messages):
        try:
            response, action = self.policy.generate('reader', messages, sample=self.sample,
                                                   seed=self.seed, max_tokens=512)
        except Exception as error:
            self.failure = error
            raise
        if action is not None:
            self.trace.append(action)
        return response['text']


class NativeMemory:
    def __init__(self, config, embedding, policy, documents, rows, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.config = replace(config, dataset=None, save_dir=str(directory),
                              force_index_from_scratch=True, force_openie_from_scratch=True)
        self.documents = documents
        self.source_by_text = {d['text']: d['id'] for d in documents}
        if len(self.source_by_text) != len(documents):
            raise ValueError('Documents must use one source ID per distinct native passage')
        builder = KGBuilder(self.config, embedding, RecordedOpenIE(documents, rows))
        builder.index([d['text'] for d in documents])
        self.builder = builder
        self.reranker = PolicyFactReranker(policy)
        self.retriever = PassageEntityRetriever(
            self.config, embedding, self.reranker, builder.graph,
            builder.chunk_embedding_store, builder.entity_embedding_store,
            builder.fact_embedding_store, builder.openie_results_path)
        self.retriever.prepare()
        # Source IDs refer to complete input passages, also when an extractor emits no facts.
        (directory / 'source_index.json').write_text(json.dumps({
            sid: self.source_by_text[row['content']]
            for sid, row in builder.chunk_embedding_store.get_all_id_to_rows().items()}, indent=2))

    def retrieve(self, question, sample=False, seed=0):
        self.reranker.sample, self.reranker.seed = sample, seed
        self.reranker.trace, self.reranker.failure = [], None
        result = self.retriever.retrieve([question], self.config.qa_top_k)[0]
        if self.reranker.failure is not None:
            raise self.reranker.failure
        return {'documents': result.docs,
                'source_ids': [self.source_by_text[text] for text in result.docs],
                'scores': result.doc_scores.tolist(), 'trace': list(self.reranker.trace)}


class MemoryFactory:
    def __init__(self, config, embedding, policy):
        self.config, self.embedding, self.policy = config, embedding, policy

    def build(self, documents, rows, directory):
        return NativeMemory(self.config, self.embedding, self.policy, documents, rows, directory)
