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

import unittest
import numpy as np
import tensorflow as tf
from tf_mdp.models import mdp  # noqa
from tf_mdp.models.reservoir import reservoir  # noqa


class ReservoirTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = tf.Graph()

        cls.system_dict = {'n_reservoirs': 3,
                           'upper_bounds': [200.0, 200.0, 200.0],
                           'lower_bounds': [10.0, 10.0, 10],
                           'initial_states': [105.0, 105.0, 105.0],
                           'adjacency_matrix': [[0, 1, 0],
                                                [0, 0, 1],
                                                [0, 0, 0]],
                           'reward_param': 0.1,
                           'above_penalty': - 100,
                           'below_penalty': - 5}

        cls.env_dict = {'rain_mean': 5.0,
                        'rain_std': 1.,
                        'evaporation_std': 5.}

        cls.reservoir = reservoir.ReservoirNonLinear(cls.graph,
                                                     cls.system_dict,
                                                     cls.env_dict)

        cls.batch_size = 1000
        cls.min_size = 50.0
        cls.max_size = cls.system_dict['upper_bounds'][0]
        cls.s_a_shape = (cls.batch_size, cls.reservoir.state_size)

        with cls.graph.as_default():
            # MDP inputs
            state_batch = tf.random_uniform(cls.s_a_shape,
                                            minval=cls.min_size,
                                            maxval=cls.max_size,
                                            name="state")
            cls.state = tf.placeholder_with_default(state_batch,
                                                    shape=cls.s_a_shape,
                                                    name="state")
            cls.action = tf.random_uniform(cls.s_a_shape,
                                           minval=0.0,
                                           maxval=50.0,
                                           name="action")
            # MDP transition
            cls.next_state = cls.reservoir.transition(cls.state, cls.action)
            # MDP reward
            cls.reward = cls.reservoir.reward(cls.state, cls.action)

        cls.scopes = cls.get_all_scopes()

    @classmethod
    def get_all_scopes(cls):
        scopes = set()
        for op in cls.graph.get_operations():
            scope = '/'.join(op.name.split('/')[:-1])
            if scope:
                scopes.add(scope)
        return scopes

    def setUp(self):
        self.sess = tf.Session(graph=self.graph)

    def tearDown(self):
        self.sess.close()

    def get_and_check_tensor(self, graph, name, shape, dtype=tf.float32):
        tensor = graph.get_tensor_by_name(name)
        self.assertEqual(tensor.shape, tf.TensorShape(shape))
        self.assertEqual(tensor.dtype, dtype)
        return tensor

    def test_model_is_subclass_of_tf_mdp(self):
        self.assertTrue(issubclass(self.reservoir.__class__, mdp.MDP))
        self.assertTrue(isinstance(self.reservoir, mdp.MDP))

    def test_model_is_subclass_of_Reservoir(self):
        self.assertTrue(issubclass(self.reservoir.__class__,
                                   reservoir.Reservoir))
        self.assertTrue(isinstance(self.reservoir, reservoir.Reservoir))

    def test_model_uses_given_graph(self):
        self.assertEqual(self.reservoir.graph, self.graph)

    def test_model_defines_name_scope(self):
        self.assertIn("transition/random_variables/rain_noise", self.scopes)
        self.assertIn("transition/random_variables/rain_noise_1/r_t",
                      self.scopes)
        self.assertIn("transition/random_variables/eva_noise", self.scopes)
        self.assertIn("transition/random_variables/eva_noise_1/e_t",
                      self.scopes)
        self.assertIn("transition/next_position/downstream", self.scopes)
        self.assertIn("reward/final_reward", self.scopes)

    def test_model_has_correct_action_size(self):
        self.assertEqual(self.reservoir.action_size,
                         self.system_dict['n_reservoirs'])

    def test_model_has_correct_state_size(self):
        self.assertEqual(self.reservoir.state_size,
                         self.system_dict['n_reservoirs'])

    def test_transition_computes_next_state_with_same_shape_of_state(self):
        self.assertEqual(self.next_state.shape, self.state.shape)

    def test_transition_computes_next_state_with_same_type_of_state(self):
        self.assertEqual(self.next_state.dtype, self.state.dtype)

    def test_transition_computes_next_state_with_correct_shape(self):
        self.assertEqual(self.next_state.shape,
                         tf.TensorShape([self.batch_size,
                                         self.reservoir.state_size]))

    def test_reward_has_correct_shape(self):
        self.assertEqual(self.reward.shape,
                         tf.TensorShape([self.batch_size, 1]))

    def test_transition_computes_next_position_with_only_positive_values(self):
        next_state_name = "transition/next_position/next_state:0"
        tf_next_state = self.get_and_check_tensor(self.graph,
                                                  next_state_name,
                                                  self.s_a_shape)
        next_state = self.sess.run(tf_next_state)
        self.assertTrue(np.all(next_state >= 0))

    def test_downstream_respects_adjacency_matrix(self):
        downstream_name = "transition/next_position/downstream_final:0"
        adjacency_matrix_name = "system/adjacency_matrix:0"
        adjacency_matrix_shape = [self.system_dict['n_reservoirs'],
                                  self.system_dict['n_reservoirs']]
        control_adjacency_matrix = self.system_dict['adjacency_matrix']
        tf_downstream = self.get_and_check_tensor(self.graph,
                                                  downstream_name,
                                                  self.s_a_shape)

        tf_state = self.get_and_check_tensor(self.graph,
                                             "state:0",
                                             self.s_a_shape)
        tf_action = self.get_and_check_tensor(self.graph,
                                              "action:0",
                                              self.s_a_shape)
        tf_adj_matrix = self.get_and_check_tensor(self.graph,
                                                  adjacency_matrix_name,
                                                  adjacency_matrix_shape)
        downstream, state, action, adj_matrix = self.sess.run([tf_downstream,
                                                               tf_state,
                                                               tf_action,
                                                               tf_adj_matrix])
        control_downstream = np.matmul(action, adj_matrix)
        control_adjacency_matrix = np.array(control_adjacency_matrix)
        test_ajacency = np.all(adj_matrix == control_adjacency_matrix)
        test_downstream = np.all(downstream == control_downstream)
        self.assertTrue(test_ajacency)
        self.assertTrue(test_downstream)

    def test_deterministic_transition(self):
        zeros = np.zeros(self.s_a_shape)
        e_t_name = "transition/random_variables/eva_noise_1/e_t/Reshape:0"
        r_t_name = "transition/random_variables/rain_noise_1/r_t/Reshape:0"
        tf_e_t = self.get_and_check_tensor(self.graph,
                                           e_t_name,
                                           self.s_a_shape)
        tf_r_t = self.get_and_check_tensor(self.graph,
                                           r_t_name,
                                           self.s_a_shape)
        feed_dict = {tf_r_t: zeros, tf_e_t: zeros}
        tensor_list = [self.state, self.action, self.next_state]
        state, action, new_state = self.sess.run(tensor_list,
                                                 feed_dict=feed_dict)
        all_water_in_state = np.sum(state, axis=1)
        condbelow = all_water_in_state <= sum(self.system_dict['upper_bounds'])
        condabove = all_water_in_state >= sum(self.system_dict['lower_bounds'])
        basic_condition = np.all(condbelow) and np.all(condabove)
        all_water_in_new_state = np.sum(new_state, axis=1)
        dumped_water = all_water_in_state - all_water_in_new_state
        action_of_the_last_reservoir = action[:, 2]
        tolerance = 1e-04
        main_comparison = np.all(np.isclose(dumped_water,
                                            action_of_the_last_reservoir,
                                            atol=tolerance))
        basic_condition_msg = "initial_states are not between bounds"
        main_comparison_msg = """the difference between
        action_of_the_last_reservoir and dumped_water are bigger
        than the difference expected with tolerance {}""".format(tolerance)
        self.assertTrue(basic_condition, msg=basic_condition_msg)
        self.assertTrue(main_comparison, msg=main_comparison_msg)

    def test_safe_range_is_respected(self):
        rewards_range_name = "reward/rewards_safe_range:0"
        states_in_range = np.full(self.s_a_shape, 105.0)
        states_above_range = np.full(self.s_a_shape, 205.0)
        states_below_range = np.full(self.s_a_shape, 5.0)

        feed_in_range = {self.state: states_in_range}
        feed_above_range = {self.state: states_above_range}
        feed_below_range = {self.state: states_below_range}

        tf_rewards_safe_range = self.get_and_check_tensor(self.graph,
                                                          rewards_range_name,
                                                          self.s_a_shape)

        rewards_safe_range = self.sess.run(tf_rewards_safe_range,
                                           feed_dict=feed_in_range)
        rewards_above_range = self.sess.run(tf_rewards_safe_range,
                                            feed_dict=feed_above_range)
        rewards_below_range = self.sess.run(tf_rewards_safe_range,
                                            feed_dict=feed_below_range)

        self.assertAlmostEqual(np.mean(rewards_safe_range),
                               0.00,
                               places=2)
        self.assertAlmostEqual(np.mean(rewards_above_range),
                               -500.00,
                               places=2)
        self.assertAlmostEqual(np.mean(rewards_below_range),
                               -25.00,
                               places=2)

    def test_reward_is_zero_when_state_is_middle_point(self):
        states_in_middle = np.full(self.s_a_shape, 105.0)
        feed_middle = {self.state: states_in_middle}
        result = np.mean(self.sess.run(self.reward, feed_dict=feed_middle))
        self.assertAlmostEqual(result,
                               0.00,
                               places=2,
                               msg="result = {}".format(result))

    def test_evaporation_and_rain_are_positiv(self):
        e_t_name = "transition/random_variables/e_t_protected:0"
        r_t_name = "transition/random_variables/r_t_protected:0"
        tf_e_t = self.get_and_check_tensor(self.graph,
                                           e_t_name,
                                           self.s_a_shape)
        tf_r_t = self.get_and_check_tensor(self.graph,
                                           r_t_name,
                                           self.s_a_shape)
        evaporation, rain = self.sess.run([tf_e_t, tf_r_t])
        condition_evaporation = np.all(evaporation >= 0)
        condition_rain = np.all(rain >= 0)
        condition_evaporation_msg = "some negative values in evaporation"
        condition_rain_msg = "some negative values in rain"
        self.assertTrue(condition_evaporation, condition_evaporation_msg)
        self.assertTrue(condition_rain, condition_rain_msg)
