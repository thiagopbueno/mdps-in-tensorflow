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

from tf_mdp import utils
from tf_mdp.eval.mdp_rnn import MDP_RNNCell, MDP_RNN
from tf_mdp.models.navigation.navigation import Navigation
from tf_mdp.policy.deterministic import DeterministicPolicyNetwork

import tensorflow as tf
import unittest

class TestMDP_RNN(unittest.TestCase):

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

        # RNN
        cls.batch_size = 1000
        cls.max_time = 10
        cls.timesteps = utils.timesteps(cls.batch_size, cls.max_time)
        cls.initial_state = utils.initial_state(cls.grid['start'], cls.batch_size)
        cls.rnn = MDP_RNN(cls.mdp, cls.policy)
        cls.rewards, cls.states, cls.actions, cls.final_state = cls.rnn.unroll(cls.initial_state, cls.timesteps)

    def setUp(self):
        with self.graph.as_default():
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def tearDown(self):
        self.sess.close()

    def test_mdp_rnn_uses_given_graph(self):
        self.assertTrue(self.rnn.graph is self.graph)
        # self.assertTrue(self.rnn.cell.graph is self.graph)
