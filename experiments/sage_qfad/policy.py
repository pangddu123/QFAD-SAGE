"""JSON-lines connection to one Qwen base with independent Writer and Reader adapters."""
import json
from pathlib import Path
import subprocess


class WorkerPolicy:
    def __init__(self, command, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self.errors = (self.log_dir / 'worker.stderr.log').open('w')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.errors, text=True, bufsize=1)
        self.counter = 0
        try:
            self.ready = self.receive()
            if self.ready['event'] != 'ready':
                raise RuntimeError('Worker did not become ready')
        except BaseException:
            self.close()
            raise

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Worker ended; inspect worker.stderr.log')
        packet = json.loads(line)
        if packet.get('event') == 'error':
            raise RuntimeError(packet['message'])
        return packet

    def request(self, packet):
        self.process.stdin.write(json.dumps(packet, ensure_ascii=False) + '\n')
        self.process.stdin.flush()
        return self.receive()

    def generate(self, role, messages, *, sample, seed, max_tokens):
        self.counter += 1
        key = f'action{self.counter}'
        response = self.request({'operation': 'generate', 'id': key, 'role': role,
            'messages': messages, 'sample': sample, 'seed': seed, 'max_tokens': max_tokens})
        (self.log_dir / f'{key}.json').write_text(json.dumps(
            {'messages': messages, **response}, ensure_ascii=False))
        return response['row'], {'group': key, 'row': 0} if sample else None

    def phase(self, name):
        return self.request({'operation': 'phase', 'phase': name})

    def update(self, role, traces, rewards):
        return self.request({'operation': 'update', 'role': role,
                             'trajectories': traces, 'rewards': rewards})

    def discard(self, traces):
        return self.request({'operation': 'discard',
                             'groups': [a['group'] for trace in traces for a in trace]})

    def save(self, name):
        return self.request({'operation': 'save', 'name': name})

    def close(self):
        if self.process.poll() is None:
            try:
                self.request({'operation': 'close'})
            except (BrokenPipeError, RuntimeError):
                pass
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.errors.close()
