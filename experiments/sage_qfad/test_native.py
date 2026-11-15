from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from passage_entity.config import PassageEntityConfig
from passage_entity.kg_builder import KGBuilder
from passage_entity.openie import OpenIE
from passage_entity.prompts import make_ner_messages
from passage_entity.reranker import FactReranker
from passage_entity.retriever import PassageEntityRetriever
from .native import NativeMemory, RecordedOpenIE, parse_writer_output, write_documents


class EmbeddingFixture:
    def batch_encode(self, texts, instruction=None, norm=True):
        if isinstance(texts, str):
            texts = [texts]
        vectors = np.array([[1. + t.lower().count('alpha'), 1. + t.lower().count('beta'),
                             1. + t.lower().count('gamma')] for t in texts])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class NativePolicyFixture:
    def __init__(self):
        self.calls = []

    def generate(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        if role == 'reader':
            text = '[[ ## fact_after_filter ## ]]\n{"fact":[["alpha","uses","beta"]]}\n[[ ## completed ## ]]'
        elif '"named_entities"' in messages[-1]['content']:
            text = '{"triples":[["alpha","uses","beta"]]}'
        else:
            text = '{"named_entities":["alpha","beta"]}'
        return {'text': text, 'finished': True}, None


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.embedding = EmbeddingFixture()
        self.policy = NativePolicyFixture()
        self.documents = [{'id': 'a', 'text': 'Alpha uses Beta.', 'patient': 1, 'stage': 0},
                          {'id': 'b', 'text': 'Gamma uses Delta.', 'patient': 1, 'stage': 0}]
        self.rows = {'a': {'extracted_entities': ['Alpha', 'Beta'], 'extracted_triples': [['Alpha', 'uses', 'Beta']]},
                     'b': {'extracted_entities': ['Gamma', 'Delta'], 'extracted_triples': [['Gamma', 'uses', 'Delta']]}}
        self.config = PassageEntityConfig(llm_model='fixture', embedding_model_key='fixture')

    def tearDown(self):
        self.directory.cleanup()

    def test_native_builder_and_retriever_parity(self):
        memory = NativeMemory(self.config, self.embedding, self.policy, self.documents, self.rows, self.root / 'actual')
        config = replace(self.config, save_dir=str(self.root / 'expected'))
        builder = KGBuilder(config, self.embedding, RecordedOpenIE(self.documents, self.rows))
        builder.index([d['text'] for d in self.documents])
        async def llm(**kwargs):
            return '[[ ## fact_after_filter ## ]]\n{"fact":[["alpha","uses","beta"]]}\n[[ ## completed ## ]]'
        retriever = PassageEntityRetriever(config, self.embedding, FactReranker(llm), builder.graph,
            builder.chunk_embedding_store, builder.entity_embedding_store, builder.fact_embedding_store,
            builder.openie_results_path)
        actual = memory.retrieve('What does Alpha use?')
        expected = retriever.retrieve(['What does Alpha use?'], config.qa_top_k)[0]
        self.assertEqual(actual['documents'], expected.docs)
        np.testing.assert_allclose(actual['scores'], expected.doc_scores)
        self.assertEqual(memory.builder.graph.vs['name'], builder.graph.vs['name'])
        self.assertEqual(memory.builder.graph.get_edgelist(), builder.graph.get_edgelist())
        self.assertEqual(memory.builder.graph.es['weight'], builder.graph.es['weight'])
        self.assertEqual([c[0] for c in self.policy.calls], ['reader'])
        self.assertTrue(all(text in [d['text'] for d in self.documents] for text in actual['documents']))

    def test_single_fact_and_single_passage_keep_vector_shape(self):
        memory = NativeMemory(self.config, self.embedding, self.policy, self.documents[:1],
                              {'a': self.rows['a']}, self.root / 'single')
        self.assertEqual(memory.retrieve('Alpha?')['source_ids'], ['a'])

    def test_empty_extraction_retains_passages_and_native_fallback(self):
        rows = {d['id']: {'extracted_entities': [], 'extracted_triples': []} for d in self.documents}
        memory = NativeMemory(self.config, self.embedding, self.policy, self.documents, rows, self.root / 'empty')
        result = memory.retrieve('Alpha?')
        self.assertEqual(set(result['source_ids']), {'a', 'b'})
        self.assertFalse(self.policy.calls)

    def test_malformed_item_does_not_remove_good_triples_and_duplicates_are_observable(self):
        raw = '{"triples":[["a","r","b"],["a","r","b"],["bad"],null,["","r","x"]]}'
        self.assertEqual(parse_writer_output(raw, 'triples'), [['a', 'r', 'b'], ['a', 'r', 'b']])

    def test_writer_uses_native_prompts_without_query_or_history(self):
        result = write_documents(self.documents[:1], self.policy, False, 0)
        self.assertTrue(result.legal)
        self.assertEqual(self.policy.calls[0][1], make_ner_messages(self.documents[0]['text']))
        self.assertEqual(len(self.policy.calls), 2)
        self.assertEqual(result.rows['a']['extracted_triples'], [['alpha', 'uses', 'beta']])

    def test_openie_preserves_few_shot_message_order(self):
        seen = []
        async def llm(**kwargs):
            seen.extend([{'role': 'system', 'content': kwargs['system_prompt']},
                         *kwargs['history_messages'], {'role': 'user', 'content': kwargs['prompt']}])
            return '{}'
        messages = make_ner_messages('Alpha uses Beta.')
        OpenIE(llm)._call_llm(messages)
        self.assertEqual(seen, messages)


if __name__ == '__main__':
    unittest.main()
