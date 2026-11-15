"""Qwen27B adapter worker. The answer role always uses the frozen base model."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from .rewards import group_advantages


class PhaseState:
    def __init__(self, steps=None):
        self.steps = dict(steps or {'writer': 0, 'reader': 0})
        self.phase = 'evaluate'
        self.groups = {}

    def set_phase(self, phase):
        if phase not in ('writer', 'rebuild', 'reader', 'evaluate'):
            raise ValueError('Unknown alternating phase')
        if self.groups:
            raise ValueError('Consume sampled trajectories before changing phase')
        self.phase = phase

    def require_trainable(self, role):
        if role not in self.steps or self.phase != role:
            raise ValueError(f'{role} is frozen during {self.phase}')

    def trajectories(self, role, paths):
        self.require_trainable(role)
        result = []
        for path in paths:
            if not path:
                raise ValueError('An update trajectory needs at least one sampled action')
            actions = []
            for action in path:
                item = self.groups[action['group']]
                if action['row'] != 0 or item['role'] != role or item['step'] != self.steps[role]:
                    raise ValueError('Trajectory role or on-policy version mismatch')
                actions.append(item)
            result.append(actions)
        return result


def completion_logps(model, prompt, tokens):
    import torch
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + tokens], device=device)
    storage = (torch.autograd.graph.save_on_cpu(pin_memory=True)
               if torch.is_grad_enabled() and device.type == 'cuda' else nullcontext())
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with storage, autocast:
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=len(tokens) + 1).logits[0, :-1].float()
        targets = torch.tensor(tokens, device=device)
        return -torch.nn.functional.cross_entropy(logits, targets, reduction='none')


def grpo_token_loss(current, old, reference, advantage, clip, kl_beta):
    import torch
    ratio = (current - old).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1. - clip, 1. + clip) * advantage)
    log_ratio = reference - current
    kl = log_ratio.exp() - log_ratio - 1.
    return -surrogate + kl_beta * kl


class AdapterWorker:
    def __init__(self, model, tokenizer, out, lr, clip, kl_beta, epochs, max_context,
                 resume=None):
        import torch
        self.model, self.tokenizer = model, tokenizer
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.clip, self.kl_beta, self.epochs, self.max_context = clip, kl_beta, epochs, max_context
        self.parameters = {role: {n: p for n, p in model.named_parameters()
                                 if 'lora_' in n and f'.{role}.' in n}
                           for role in ('writer', 'reader')}
        if not all(self.parameters.values()):
            raise ValueError('Both role adapters must be installed')
        self.optimizers = {role: torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=0.)
                           for role, params in self.parameters.items()}
        self.state = PhaseState()
        if resume:
            checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
            expected = {n for params in self.parameters.values() for n in params}
            if set(checkpoint['parameters']) != expected:
                raise ValueError('Checkpoint must contain exactly the two role adapters')
            model.load_state_dict(checkpoint['parameters'], strict=False)
            for role in self.optimizers:
                self.optimizers[role].load_state_dict(checkpoint['optimizers'][role])
            self.state = PhaseState(checkpoint['steps'])

    def generate(self, request):
        import torch
        role = request['role']
        if role not in ('writer', 'reader', 'answer'):
            raise ValueError('Unknown policy role')
        if request['sample']:
            self.state.require_trainable(role)
        if role != 'answer':
            self.model.set_adapter(role)
        prompt = self.tokenizer.apply_chat_template(request['messages'], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if len(prompt) + request['max_tokens'] > self.max_context:
            raise ValueError('Native prompt and completion exceed model context; no source text was truncated')
        torch.manual_seed(request['seed'])
        self.model.eval()
        device = next(self.model.parameters()).device
        ids = torch.tensor([prompt], device=device)
        generation = ({'do_sample': True, 'temperature': 1., 'top_p': 1., 'top_k': 0}
                      if request['sample'] else {'do_sample': False})
        adapter = self.model.disable_adapter() if role == 'answer' else nullcontext()
        with adapter, torch.no_grad():
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                **generation, max_new_tokens=request['max_tokens'], min_new_tokens=0,
                repetition_penalty=1., no_repeat_ngram_size=0, use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)[0]
        tokens = output[len(prompt):].tolist()
        finished = self.tokenizer.eos_token_id in tokens
        if finished:
            tokens = tokens[:tokens.index(self.tokenizer.eos_token_id) + 1]
        row = {'text': self.tokenizer.decode(tokens, skip_special_tokens=True), 'finished': finished,
               'token_ids': tokens, 'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(tokens)}}
        step = self.state.steps.get(role, 0)
        if request['sample']:
            self.state.groups[request['id']] = {'prompt': prompt, 'tokens': tokens,
                                               'role': role, 'step': step}
        return {'event': 'generated', 'role': role, 'policy_step': step, 'row': row}

    def update(self, request):
        import torch
        role, paths = request['role'], request['trajectories']
        trajectories = self.state.trajectories(role, paths)
        advantages = group_advantages(request['rewards'])
        if len(trajectories) != len(advantages):
            raise ValueError('One return is required per trajectory')
        if not any(advantages):
            raise ValueError('Equal-reward groups must be discarded without an optimizer update')
        self.model.set_adapter(role)
        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        if trainable != set(self.parameters[role]):
            raise ValueError('Only the active role adapter may receive gradients')
        self.model.eval()
        params = list(self.parameters[role].values())
        before = [p.detach().cpu().clone() for p in params]
        with torch.no_grad():
            old = [[completion_logps(self.model, a['prompt'], a['tokens']).detach()
                    for a in path] for path in trajectories]
            with self.model.disable_adapter():
                reference = [[completion_logps(self.model, a['prompt'], a['tokens']).detach()
                              for a in path] for path in trajectories]
        metrics = []
        for epoch in range(self.epochs):
            self.model.zero_grad(set_to_none=True)
            total_loss = 0.
            for path, previous, refs, advantage in zip(trajectories, old, reference, advantages):
                count = sum(len(a['tokens']) for a in path)
                for action, prior, ref in zip(path, previous, refs):
                    current = completion_logps(self.model, action['prompt'], action['tokens'])
                    loss = grpo_token_loss(current, prior, ref, advantage, self.clip, self.kl_beta)
                    loss = loss.sum() / (len(trajectories) * count)
                    loss.backward()
                    total_loss += loss.item()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError('Non-finite policy gradient')
            self.optimizers[role].step()
            metrics.append({'epoch': epoch, 'loss': total_loss, 'gradient_norm': norm.item()})
        self.state.steps[role] += 1
        for group in {a['group'] for path in paths for a in path}:
            del self.state.groups[group]
        delta = sum((p.detach().cpu() - prev).square().sum().item()
                    for p, prev in zip(params, before)) ** .5
        self.save('latest')
        return {'event': 'updated', 'role': role, 'steps': dict(self.state.steps),
                'rewards': request['rewards'], 'advantages': advantages,
                'epochs': metrics, 'adapter_delta_l2': delta}

    def save(self, name):
        import torch
        path = self.out / name
        path.mkdir(exist_ok=True)
        parameters = {n: p.detach().cpu().clone() for role in self.parameters.values() for n, p in role.items()}
        temporary = path / 'checkpoint.pt.tmp'
        torch.save({'parameters': parameters,
                    'optimizers': {r: opt.state_dict() for r, opt in self.optimizers.items()},
                    'steps': dict(self.state.steps)}, temporary)
        temporary.replace(path / 'checkpoint.pt')
        self.model.save_pretrained(path / 'adapters', selected_adapters=['writer', 'reader'])
        return {'event': 'saved', 'path': str(path / 'checkpoint.pt'), 'steps': dict(self.state.steps)}

    def request(self, request):
        operation = request['operation']
        if operation == 'generate':
            return self.generate(request)
        if operation == 'update':
            return self.update(request)
        if operation == 'phase':
            self.state.set_phase(request['phase'])
            return {'event': 'phase', 'phase': self.state.phase, 'steps': dict(self.state.steps)}
        if operation == 'discard':
            for key in set(request['groups']):
                del self.state.groups[key]
            return {'event': 'discarded'}
        if operation == 'save':
            return self.save(request['name'])
        if operation == 'close':
            return {'event': 'closed', 'steps': dict(self.state.steps)}
        raise ValueError(f'Unknown operation: {operation}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=.2)
    parser.add_argument('--kl-beta', type=float, default=.02)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-context', type=int, default=32768)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or not 0 < args.clip < 1 or args.kl_beta < 0:
        parser.error('Invalid GRPO hyperparameters')
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    torch.set_num_threads(4)
    torch.manual_seed(17)
    config = AutoConfig.from_pretrained(args.model).text_config
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='sdpa', key_mapping={r'^model\.language_model\.': 'model.'},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        output_loading_info=True)
    if loading['missing_keys'] or loading.get('mismatched_keys'):
        raise RuntimeError('Base weights do not match the Qwen text architecture')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    lora = dict(task_type='CAUSAL_LM', r=8, lora_alpha=16, target_modules='all-linear', lora_dropout=0., bias='none')
    model = get_peft_model(model, LoraConfig(**lora), adapter_name='writer')
    model.add_adapter('reader', LoraConfig(**lora))
    worker = AdapterWorker(model, AutoTokenizer.from_pretrained(args.model), args.out,
        args.lr, args.clip, args.kl_beta, args.epochs, args.max_context, args.resume)
    def emit(packet):
        print(json.dumps(packet, ensure_ascii=False), flush=True)
    emit({'event': 'ready', 'model': args.model, 'steps': dict(worker.state.steps),
          'answer_role': 'frozen base; adapters disabled', 'quantization': 'NF4'})
    for line in sys.stdin:
        request = json.loads(line)
        try:
            emit(worker.request(request))
        except Exception as error:
            emit({'event': 'error', 'message': f'{type(error).__name__}: {error}'})
            raise
        if request['operation'] == 'close':
            break


if __name__ == '__main__':
    main()
