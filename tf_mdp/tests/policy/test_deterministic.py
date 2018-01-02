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

from tf_mdp.policy.deterministic import DeterministicPolicyNetwork

import re
import numpy as np
import tensorflow as tf
import unittest

class TestDeterministicPolicyNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = tf.Graph()

        cls.batch_size = 10000
        cls.max_time = 10

        with cls.graph.as_default():
            cls.state = tf.random_uniform((cls.batch_size, 2), minval=0.0, maxval=10.0, name="state")
            cls.timestep = tf.cast(tf.random_uniform((cls.batch_size, 1), minval=0, maxval=cls.max_time, dtype=tf.int32, name="timestep"), tf.float32)
            cls.inputs = tf.concat([cls.state, cls.timestep], axis=1, name="inputs")

        cls.shape = (3, 20, 10, 5, 2)
        cls.bounds = 2.0
        cls.policy = DeterministicPolicyNetwork(cls.graph, cls.shape, cls.bounds)
        cls.action = cls.policy(cls.inputs)

        cls.scopes = cls.get_all_scopes()

    def setUp(self):
        with self.policy.graph.as_default():
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
        tensor = self.policy.graph.get_tensor_by_name(name)
        self.assertEqual(tensor.shape, tf.TensorShape(shape))
        self.assertEqual(tensor.dtype, dtype)
        return tensor


    def test_policy_uses_given_graph(self):
        self.assertTrue(self.policy.graph is self.graph)

    def test_policy_defines_its_namescope(self):
        self.assertTrue("policy" in self.scopes)

    def test_policy_net_has_given_number_of_layers(self):
        number_of_layers = 0
        for scope in self.scopes:
            if re.search(r"^policy/layer\d+$", scope):
                number_of_layers += 1
        self.assertEqual(number_of_layers, len(self.shape) - 1)

    def test_policy_net_has_given_layers_units_and_activation(self):
        for i, n_h in enumerate(self.shape[1:-1]):
            layer = self.get_and_check_tensor("policy/layer{}/Relu:0".format(str(i + 1)), (self.batch_size, n_h))
        n_h = self.shape[-1]
        layer = self.get_and_check_tensor("policy/layer{}/Tanh:0".format(str(len(self.shape) - 1)), (self.batch_size, n_h))

    def test_policy_has_action_as_its_output(self):
        self.assertTrue(self.action is self.get_and_check_tensor("policy/action:0", (self.batch_size, self.shape[-1])))

        last_layer = self.policy.graph.get_tensor_by_name("policy/layer{}/Tanh:0".format(len(self.shape) - 1))
        bounds = self.get_and_check_tensor("policy/bounds:0", ())
        self.assertTrue(set(self.action.op.inputs), set([last_layer, bounds]))

        action_ = self.sess.run(self.action)
        for a in action_:
            for a_i in a:
                self.assertTrue(np.abs(a_i) <= self.bounds)
