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

import tf_mdp.evaluation.utils as utils
from tf_mdp.evaluation.rnn import StochasticMarkovCell, MarkovRecurrentModel
from tf_mdp.models.navigation.navigation import StochasticNavigation
from tf_mdp.policy.deterministic import DeterministicPolicyNetwork

import tensorflow as tf
import unittest

class TestRecurrentModel(unittest.TestCase):

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
        cls.policy_shape = [cls.mdp.state_size + 1, 20, 5, cls.mdp.action_size]
        cls.policy = DeterministicPolicyNetwork(cls.graph, cls.policy_shape)

        # RNN
        cell = StochasticMarkovCell(cls.mdp, cls.policy)
        cls.rnn = MarkovRecurrentModel(cell)
        cls.batch_size = 1000
        cls.max_time = 10
        with cls.graph.as_default():
            cls.timesteps = tf.constant(utils.timesteps(cls.batch_size, cls.max_time), name="timesteps")
        cls.initial_state = utils.initial_state(cls.config["initial"], cls.batch_size)
        cls.rewards, cls.states, cls.actions = cls.rnn.unroll(cls.initial_state, cls.timesteps)

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

    @classmethod
    def graph_has_scope(cls, scope, fullname=False):
        if fullname:
            return scope in cls.get_all_scopes()

        for sc in cls.get_all_scopes():
            if sc.startswith(scope):
                return True
        return False

    def get_and_check_tensor(self, name, shape, dtype=tf.float32):
        tensor = self.graph.get_tensor_by_name(name)
        self.assertEqual(tensor.shape.as_list(), list(shape))
        self.assertEqual(tensor.dtype, dtype)
        return tensor


    def test_mdp_rnn_uses_given_graph(self):
        self.assertTrue(self.rnn.cell.graph is self.graph)
        self.assertTrue(self.rnn.cell.mdp.graph is self.graph)

    def test_mdp_rnn_defines_its_scope(self):
        self.assertTrue(self.graph_has_scope("mdp_rnn"))

    def test_mdp_rnn_defines_outputs_scope(self):
        self.assertTrue(self.graph_has_scope("mdp_rnn/outputs"))

    def test_mdp_rnn_cell_defines_output_scope(self):
        self.assertTrue(self.graph_has_scope("mdp_rnn/while/output"))

    def test_mdp_rnn_cell_encapsulates_policy(self):
        self.assertTrue(self.graph_has_scope("mdp_rnn/while/policy_cell/policy"))

    def test_mdp_rnn_cell_encapsulates_transition(self):
        self.assertTrue(self.graph_has_scope("mdp_rnn/while/transition_cell/transition"))

    def test_mdp_rnn_cell_encapsulates_reward(self):
        self.assertTrue(self.graph_has_scope("mdp_rnn/while/reward_cell/reward"))

    def test_mdp_rnn_cell_defines_its_output_shape_and_size(self):
        self.assertEqual(self.rnn.cell.output_size, self.mdp.state_size + self.mdp.action_size + 1)

    def test_mdp_rnn_cell_defines_its_hidden_state_size(self):
        self.assertEqual(self.rnn.cell.state_size, self.mdp.state_size)

    def test_mdp_rnn_cell_augment_state_representation_with_timestep(self):
        self.sess.run([self.rewards, self.states, self.actions])
        self.get_and_check_tensor("mdp_rnn/while/policy_cell/state_t:0", (self.batch_size, 3))

    def test_mdp_rnn_reuse_variables_in_policy_network(self):
        trainable_variable_scopes = set()
        trainable_variables = self.rnn.cell.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in trainable_variables:
            trainable_variable_scopes.add('/'.join(var.name.split('/')[:-1]))
        self.assertEqual(len(trainable_variable_scopes), len(self.policy_shape) - 1)
        self.assertEqual(len(trainable_variables), 2 * len(trainable_variable_scopes))

    def test_mdp_rnn_generates_trajectories_with_given_batch_size_and_horizon(self):
        rewards, states, actions = self.sess.run([self.rewards, self.states, self.actions])
        self.assertEqual(rewards.shape, (self.batch_size, self.max_time, 1))
        self.assertEqual(states.shape,  (self.batch_size, self.max_time, self.mdp.state_size))
        self.assertEqual(actions.shape, (self.batch_size, self.max_time, self.mdp.action_size))

