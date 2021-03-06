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
from tf_mdp.train.optimizer import SGDPolicyOptimizer

import tensorflow as tf
import unittest


class TestSGDPolicyOptimizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = tf.Graph()

        # MDP
        cls.config = {
            "grid": {
                "ndim": 2,
                "size":  [0.0, 10.0],
                "start": [2.0,  5.0],
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

        # PolicyNetwork
        cls.shape = [cls.mdp.state_size + 1, 20, 5, cls.mdp.action_size]
        cls.policy = DeterministicPolicyNetwork(cls.graph, cls.shape)

        # PolicyEvaluation
        cls.max_time = 10
        cls.batch_size = 1000
        cls.gamma = 0.9
        cls.mc = MCPolicyEvaluation(cls.mdp, cls.policy,
                                    initial_state=cls.config["grid"]['start'],
                                    max_time=cls.max_time,
                                    batch_size=cls.batch_size,
                                    gamma=cls.gamma)

        # PolicyOptimizer
        cls.learning_rate = 0.01
        metrics = {
            "loss":  cls.mc.expected_return,
            "total": cls.mc.total
        }
        cls.optimizer = SGDPolicyOptimizer(cls.graph, metrics, cls.learning_rate)

    def test_optimizer_uses_given_graph(self):
        self.assertTrue(self.optimizer.graph is self.graph)

    def test_optimizer_uses_given_learning_rate(self):
        self.assertEqual(self.optimizer.learning_rate, self.learning_rate)
