import unittest

from .rewards import RewardWeights, answer_f1, group_advantages, task_reward, trajectory_reward


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.weights = RewardWeights(1., 1., 1., 1., .2, .1)

    def test_components_and_normalized_task(self):
        value = task_reward(['a', 'c', 'a'], ['a', 'b'], True, 'red blue', ['red'], self.weights)
        self.assertEqual(value['recall'], .5)
        self.assertEqual(value['precision'], .5)
        self.assertEqual(value['deducibility'], 1.)
        self.assertAlmostEqual(value['answer_f1'], 2 / 3)
        self.assertAlmostEqual(value['task'], 2 / 3)

    def test_repetition_before_union_and_format_sum(self):
        task = {'task': .6}
        triples = [('a', 'r', 'b'), ('a', 'r', 'b'), ('c', 'r', 'd')]
        result = trajectory_reward(task, triples, [1, 1], self.weights)
        self.assertAlmostEqual(result['reward'], .6 - .2 / 3 + .2)

    def test_illegal_trajectory_has_zero_return(self):
        result = trajectory_reward({'task': 1.}, [], [1, 1], self.weights, legal=False)
        self.assertEqual(result['reward'], 0.)

    def test_empty_retrieval_does_not_get_precision_credit(self):
        result = task_reward([], ['a'], False, 'x', ['y'], self.weights)
        self.assertEqual(result['task'], 0.)

    def test_missing_gold_is_not_silently_scored(self):
        with self.assertRaises(ValueError):
            task_reward(['a'], [], True, 'x', ['x'], self.weights)

    def test_answer_aliases_and_symbol_only_answers(self):
        self.assertEqual(answer_f1('The Alpha', ['beta', 'alpha']), 1.)
        self.assertEqual(answer_f1('++', ['++']), 1.)
        self.assertEqual(answer_f1('++', ['+']), 0.)
        self.assertEqual(answer_f1('++', ['']), 0.)

    def test_group_normalization_and_equal_rewards(self):
        self.assertEqual(group_advantages([0., 2.]), [-1., 1.])
        self.assertEqual(group_advantages([2., 2.]), [0., 0.])

    def test_weight_validation(self):
        with self.assertRaises(ValueError):
            RewardWeights(0, 0, 0, 0, 1, 1)
        with self.assertRaises(ValueError):
            RewardWeights(1, 1, float('nan'), 1, 1, 1)


if __name__ == '__main__':
    unittest.main()
