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

from tf_mdp.eval.montecarlo import MCPolicyEvaluation
from tf_mdp.models.navigation.navigation import Navigation
from tf_mdp.policy.deterministic import DeterministicPolicyNetwork


import tensorflow as tf
import unittest

class TestMCPolicyEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = tf.Graph()

        # MDP model
        cls.grid = {
            'ndim': 2,
            'size': (0.0, 10.0),
            'start': (2.0,  5.0),
            'goal': (8.0,  5.0),
            'deceleration': {
                'center': (5.0, 5.0),
                'decay': 2.0
            }
        }
        cls.mdp = Navigation(cls.graph, cls.grid)

        # Policy Network
        cls.shape = [cls.mdp.state_size + 1, 20, 5, cls.mdp.action_size]
        cls.policy = DeterministicPolicyNetwork(cls.graph, cls.shape)

        # MCPolicyEvaluation estimator
        cls.max_time = 10
        cls.batch_size = 1000
        cls.gamma = 0.9
        cls.mc = MCPolicyEvaluation(cls.mdp, cls.policy,
                                    initial_state=cls.grid['start'],
                                    max_time=cls.max_time,
                                    batch_size=cls.batch_size,
                                    gamma=cls.gamma)

        cls.expected_return, cls.total, cls.rewards, cls.states, cls.actions = cls.mc.eval()

    def setUp(self):
        pass

    def tearDown(self):
        pass


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
        self.assertEqual(tensor.shape, tf.TensorShape(shape))
        self.assertEqual(tensor.dtype, dtype)
        return tensor


    def test_mc_estimator_sample_trajectories_with_given_batch_size_and_horizon(self):
        self.assertEqual(self.total.shape,   (self.batch_size, 1))
        self.assertEqual(self.rewards.shape, (self.batch_size, self.max_time, 1))
        self.assertEqual(self.states.shape,  (self.batch_size, self.max_time, self.mdp.state_size))
        self.assertEqual(self.actions.shape, (self.batch_size, self.max_time, self.mdp.action_size))

    def test_mc_estimator_defines_discount_schedule(self):
        discount_schedule = self.get_and_check_tensor("policy_evaluation/discount_schedule:0", (self.batch_size, self.max_time, 1))
        with tf.Session(graph=self.mdp.graph) as sess:
            sess.run(tf.global_variables_initializer())
            discount_schedule = sess.run(discount_schedule)
            for schedule in discount_schedule:
                expected_discount_factor = 1.0
                for discount in schedule:
                    self.assertAlmostEqual(discount, expected_discount_factor, places=5)
                    expected_discount_factor *= self.gamma

    def test_mc_estimator_evaluates_policy(self):
        pass



