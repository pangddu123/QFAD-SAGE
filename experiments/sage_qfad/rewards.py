"""SAGE section 4.1 trajectory return, with observable component scores."""
from collections import Counter
from dataclasses import dataclass, asdict
import math
import re
import string


@dataclass(frozen=True)
class RewardWeights:
    recall: float
    precision: float
    deducibility: float
    answer_f1: float
    repetition: float
    format: float

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('Reward coefficients must be finite and nonnegative')
        if self.task_weight == 0:
            raise ValueError('At least one task reward coefficient must be positive')

    @property
    def task_weight(self):
        return self.recall + self.precision + self.deducibility + self.answer_f1


def answer_tokens(text):
    normalized = re.sub(r'\b(a|an|the)\b', ' ', text.lower())
    normalized = ''.join(c for c in normalized if c not in string.punctuation)
    # Symbol-only answers such as "++" remain distinguishable from an empty answer.
    return normalized.split() or text.strip().split()


def answer_f1(prediction, aliases):
    if not aliases:
        raise ValueError('Answer aliases are required for the answer reward')
    predicted = answer_tokens(prediction)
    best = 0.
    for alias in aliases:
        expected = answer_tokens(alias)
        if not predicted or not expected:
            score = float(predicted == expected)
        else:
            overlap = sum((Counter(predicted) & Counter(expected)).values())
            score = 2. * overlap / (len(predicted) + len(expected))
        best = max(best, score)
    return best


def task_reward(retrieved_ids, gold_ids, deducible, prediction, aliases, weights):
    gold, retrieved = set(gold_ids), set(retrieved_ids)
    if not gold:
        raise ValueError('SAGE retrieval reward requires annotated supporting documents')
    if type(deducible) is not bool:
        raise ValueError('Deducibility judge must return a boolean')
    hits = len(gold & retrieved)
    parts = {'recall': hits / len(gold),
             'precision': hits / len(retrieved) if retrieved else 0.,
             'deducibility': float(deducible), 'answer_f1': answer_f1(prediction, aliases)}
    parts['task'] = sum(getattr(weights, key) * value for key, value in parts.items()) / weights.task_weight
    return parts


def trajectory_reward(task, triples, format_rewards, weights, legal=True):
    """Count duplicate writes before graph union; illegal trajectories return zero (O.1)."""
    triples = [tuple(t) for t in triples]
    repetition = (len(triples) - len(set(triples))) / len(triples) if triples else 0.
    formatting = sum(format_rewards)
    reward = task['task'] - weights.repetition * repetition + weights.format * formatting
    return {**task, 'repetition_rate': repetition, 'format_sum': formatting,
            'legal': legal, 'reward': reward if legal else 0.}


def group_advantages(rewards):
    if len(rewards) < 2 or not all(math.isfinite(x) for x in rewards):
        raise ValueError('GRPO requires a finite reward group of size at least two')
    mean = sum(rewards) / len(rewards)
    std = (sum((x - mean) ** 2 for x in rewards) / len(rewards)) ** .5
    return [(x - mean) / std for x in rewards] if std else [0.] * len(rewards)
