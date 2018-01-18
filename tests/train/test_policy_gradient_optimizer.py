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

from tf_mdp import models
from tf_mdp.evaluation.mrm import MarkovRecurrentModel
from tf_mdp.policy.deterministic import DeterministicPolicyNetwork
from tf_mdp.train.optimizer import PolicyGradientOptimizer

import tensorflow as tf
import unittest

class TestPolicyGradientOptimizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = tf.Graph()

        # Hyperparameters
        cls.batch_size = 1000
        cls.timesteps = 10
        cls.learning_rate = 0.001
        cls.discount = 0.9

        #MDP
        model, config = models.make("noisy-navigation-small")
        cls.mdp = model(graph, config)
        cls.start = config["initial"]

        # PolicyNetwork
        shape = [cls.mdp.state_size + 1, 20, 5, cls.mdp.action_size]
        cls.policy = DeterministicPolicyNetwork(cls.mdp.graph, shape)

        # MarkovRecurrentModel
        cls.trajectory = MarkovRecurrentModel(cls.mdp, cls.policy).unroll(cls.start, cls.batch_size, cls.timesteps)

        # PolicyOptimizer
        cls.optimizer = PolicyGradientOptimizer(cls.graph, cls.policy, cls.trajectory, cls.learning_rate, cls.discount)
