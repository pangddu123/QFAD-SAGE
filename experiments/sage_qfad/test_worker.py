from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from .worker import AdapterWorker, completion_logps, grpo_token_loss


class TinyPolicy(torch.nn.Module):
    """CPU-only adapter fixture for loss, freezing and checkpoint tests."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 4)
        self.output = torch.nn.Linear(4, 7)
        self.lora_A = torch.nn.ModuleDict({role: torch.nn.Linear(4, 7, bias=False)
                                           for role in ('writer', 'reader')})
        self.enabled = True
        self.set_adapter('writer')

    def set_adapter(self, role):
        self.role = role
        for name, parameter in self.named_parameters():
            parameter.requires_grad_('lora_' in name and f'.{role}.' in name)

    @contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
        hidden = self.embedding(input_ids)
        logits = self.output(hidden)
        if self.enabled:
            logits = logits + self.lora_A[self.role](hidden)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)

    def save_pretrained(self, path, selected_adapters):
        Path(path).mkdir(parents=True, exist_ok=True)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = TinyPolicy()

    def test_completion_logps_shift_and_prompt_mask(self):
        prompt, tokens = [0, 1, 2], [3, 4]
        actual = completion_logps(self.model, prompt, tokens)
        logits = self.model(torch.tensor([prompt + tokens])).logits
        expected = logits[0, 2:4].log_softmax(-1).gather(1, torch.tensor(tokens)[:, None]).flatten()
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(actual), len(tokens))

    def test_advantage_sign_and_clipping(self):
        current = torch.tensor([0.], requires_grad=True)
        loss = grpo_token_loss(current, torch.tensor([0.]), torch.tensor([0.]), 1., .2, 0.).sum()
        loss.backward()
        self.assertLess(current.grad.item(), 0.)
        current = torch.tensor([1.], requires_grad=True)
        loss = grpo_token_loss(current, torch.tensor([0.]), torch.tensor([0.]), 1., .2, 0.).sum()
        loss.backward()
        self.assertEqual(current.grad.item(), 0.)

    def test_role_isolation_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = AdapterWorker(self.model, None, directory, 1e-3, .2, .02, 1, 32)
            worker.request({'operation': 'phase', 'phase': 'writer'})
            before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
            for key, tokens in [('a', [3, 4]), ('b', [5, 6])]:
                worker.state.groups[key] = {'role': 'writer', 'step': 0, 'prompt': [0, 1, 2], 'tokens': tokens}
            result = worker.update({'role': 'writer', 'rewards': [1., 0.],
                                    'trajectories': [[{'group': 'a', 'row': 0}], [{'group': 'b', 'row': 0}]]})
            self.assertGreater(result['adapter_delta_l2'], 0.)
            self.assertEqual(result['steps'], {'writer': 1, 'reader': 0})
            for name, param in self.model.named_parameters():
                if '.writer.' not in name:
                    torch.testing.assert_close(param, before[name], rtol=0, atol=0)
            worker.request({'operation': 'phase', 'phase': 'reader'})
            frozen_writer = {n: p.detach().clone() for n, p in worker.parameters['writer'].items()}
            for key, tokens in [('a', [3, 4]), ('b', [5, 6])]:
                worker.state.groups[key] = {'role': 'reader', 'step': 0, 'prompt': [0, 1, 2], 'tokens': tokens}
            reader = worker.update({'role': 'reader', 'rewards': [0., 1.],
                                    'trajectories': [[{'group': 'a', 'row': 0}], [{'group': 'b', 'row': 0}]]})
            self.assertGreater(reader['adapter_delta_l2'], 0.)
            self.assertEqual(reader['steps'], {'writer': 1, 'reader': 1})
            for name, param in worker.parameters['writer'].items():
                torch.testing.assert_close(param, frozen_writer[name], rtol=0, atol=0)
            restored = AdapterWorker(TinyPolicy(), None, Path(directory) / 'restored',
                1e-3, .2, .02, 1, 32, Path(directory) / 'latest/checkpoint.pt')
            self.assertEqual(restored.state.steps, worker.state.steps)
            for role in ('writer', 'reader'):
                for name, param in restored.parameters[role].items():
                    torch.testing.assert_close(param, worker.parameters[role][name])
            self.assertTrue(restored.optimizers['writer'].state_dict()['state'])


if __name__ == '__main__':
    unittest.main()
