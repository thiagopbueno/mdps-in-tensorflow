# This file is part of TF-MDP.

# TF-MDP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# TF-MDP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with TF-MDP.  If not, see <http://www.gnu.org/licenses/>.

from tf_mdp.evaluation.montecarlo import MCPolicyEvaluation
from tf_mdp.models.navigation.navigation import StochasticNavigation
from tf_mdp.policy.deterministic import DeterministicPolicyNetwork

import numpy as np
import tensorflow as tf
import unittest


class TestMCPolicyEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = tf.Graph()

        # MDP model
        cls.config = {
            "initial": [2.0,  5.0],
            "grid": {
                "ndim": 2,
                "size":  [0.0, 10.0],
                "goal":  [8.0,  5.0],
                "deceleration": [{
                    "center": [5.0, 5.0],
                    "decay": 2.0
                }]
            },
            "alpha_min": 0.0,
            "alpha_max": 10.0
        }
        cls.mdp = StochasticNavigation(cls.graph, cls.config)

        # Policy Network
        cls.shape = [cls.mdp.state_size + 1, 20, 5, cls.mdp.action_size]
        cls.policy = DeterministicPolicyNetwork(cls.graph, cls.shape)

        # MCPolicyEvaluation estimator
        cls.max_time = 10
        cls.batch_size = 1000
        cls.gamma = 0.9
        cls.mc = MCPolicyEvaluation(cls.mdp, cls.policy,
                                    initial_state=cls.config["initial"],
                                    max_time=cls.max_time,
                                    batch_size=cls.batch_size,
                                    gamma=cls.gamma)


    def setUp(self):
        with self.graph.as_default():
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def tearDown(self):
        self.sess.close()


    # helper functions
    @classmethod
    def get_all_scopes(cls):
        scopes = set()
        for op in cls.graph.get_operations():
            scope = '/'.join(op.name.split('/')[:-1])
            if scope:
                scopes.add(scope)
        return scopes

    def get_and_check_tensor(self, name, shape, dtype=tf.float32):
        tensor = self.graph.get_tensor_by_name(name)
        self.assertEqual(tensor.shape.as_list(), list(shape))
        self.assertEqual(tensor.dtype, dtype)
        return tensor

    def test_mc_estimator_defines_its_scope(self):
        self.assertTrue("policy_evaluation" in self.get_all_scopes())

    def test_mc_estimator_defines_discount_schedule(self):
        discount_schedule = self.get_and_check_tensor("policy_evaluation/discount_schedule:0", (self.batch_size, self.max_time, 1))
        for schedule in self.sess.run(discount_schedule):
            expected_discount_factor = 1.0
            for discount in schedule:
                self.assertAlmostEqual(discount, expected_discount_factor, places=5)
                expected_discount_factor *= self.gamma

    def test_mc_estimator_evaluates_metrics_with_given_shapes(self):
        expected_return, total, rewards = self.mc.eval(self.sess)
        self.assertEqual(expected_return.shape, (1,))
        self.assertEqual(total.shape, (self.batch_size, 1))

    def test_mc_estimator_evaluates_metrics_consistenly(self):
        expected_return, total, rewards = self.mc.eval(self.sess)
        self.assertAlmostEqual(float(expected_return), float(np.mean(total)), places=4)

        discount_schedule = self.sess.run(self.mc.discount_schedule)
        returns = np.sum(rewards * discount_schedule, axis=1)
        for actual_return_per_episode, expected_return_per_episode in zip(total, returns):
            self.assertAlmostEqual(float(actual_return_per_episode), float(expected_return_per_episode), places=4)

    def test_mc_estimator_sample_complete_trajectories(self):
        states, actions, rewards = self.mc.sample(self.sess)
        self.assertEqual(states.shape,  (self.batch_size, self.max_time, self.mdp.state_size))
        self.assertEqual(actions.shape, (self.batch_size, self.max_time, self.mdp.action_size))
        self.assertEqual(rewards.shape, (self.batch_size, self.max_time, 1))
