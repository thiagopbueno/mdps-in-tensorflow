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

from tf_mdp.models import mdp
from tf_mdp.models.navigation import navigation
import numpy as np
import tensorflow as tf
import unittest

class TestNavigation(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.graph = tf.Graph()

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
		cls.model = navigation.Navigation(cls.graph, cls.grid)

		with cls.graph.as_default():
			# MDP inputs
			cls.batch_size = 1000
			cls.min_size, cls.max_size = cls.grid["size"]
			cls.state  = tf.random_uniform((cls.batch_size, cls.model.state_size),  minval=cls.min_size, maxval=cls.max_size, name="state")
			cls.action = tf.random_uniform((cls.batch_size, cls.model.action_size), minval=-1.0, maxval=1.0, name="action")

			# MDP transition
			cls.next_state = cls.model.transition(cls.state, cls.action)

			# MDP reward
			cls.reward = cls.model.reward(cls.state, cls.action)

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
		self.sess = tf.Session(graph=self.model.graph)

	def tearDown(self):
		self.sess.close()

	def get_and_check_tensor(self, name, shape, dtype=tf.float32):
		tensor = self.model.graph.get_tensor_by_name(name)
		self.assertEqual(tensor.shape, tf.TensorShape(shape))
		self.assertEqual(tensor.dtype, dtype)
		return tensor


	def test_model_is_subclass_of_tf_mdp(self):
		self.assertTrue(issubclass(self.model.__class__, mdp.TF_MDP))
		self.assertTrue(isinstance(self.model, mdp.TF_MDP))

	def test_model_uses_given_graph(self):
		self.assertTrue(self.model.graph is self.graph)

	def test_model_defines_constants_name_scope(self):
		self.assertTrue("constants/grid" in self.scopes)
		self.assertTrue("constants/deceleration" in self.scopes)
		self.assertTrue("constants/distribution" in self.scopes)

	def test_model_defines_transition_name_scope(self):
		self.assertTrue("transition/deviation"     in self.scopes)
		self.assertTrue("transition/direction"     in self.scopes)
		self.assertTrue("transition/deceleration"  in self.scopes)
		self.assertTrue("transition/next_position" in self.scopes)

	def test_model_defines_reward_name_scope(self):
		self.assertTrue("reward" in self.scopes)

	def test_model_has_correct_action_size(self):
		self.assertEqual(self.model.action_size, self.grid["ndim"])

	def test_model_has_correct_state_size(self):
		self.assertEqual(self.model.state_size, self.grid["ndim"])

	def test_transition_computes_next_state_with_same_shape_of_state(self):
		self.assertEqual(self.next_state.shape, self.state.shape)

	def test_transition_computes_next_state_with_same_type_of_state(self):
		self.assertEqual(self.next_state.dtype, self.state.dtype)

	def test_transition_computes_next_state_with_correct_shape(self):
		self.assertEqual(self.next_state.shape, tf.TensorShape([self.batch_size, self.model.state_size]))

	def test_reward_has_correct_shape(self):
		self.assertEqual(self.reward.shape, tf.TensorShape([self.batch_size, 1]))

	def test_transition_computes_velocity_as_the_norm_of_action_when_sampling_deviations(self):
		velocity_ = self.get_and_check_tensor("transition/deviation/velocity/Sqrt:0", [self.batch_size, 1])
		action, velocity_ = self.sess.run([self.action, velocity_])
		velocity = np.linalg.norm(action, axis=1)
		for i in range(len(velocity)):
			self.assertAlmostEqual(float(velocity_[i]), float(velocity[i]), places=5)

	def test_transition_defines_distribution_location_as_zero_mean_when_sampling_deviations(self):
		loc_ = self.get_and_check_tensor("transition/deviation/loc:0", ())
		loc_ = self.sess.run(loc_)
		self.assertAlmostEqual(loc_, 0.0)

	def test_transition_defines_distribution_scale_range_when_sampling_deviations(self):
		scale_min_ = self.get_and_check_tensor("constants/distribution/scale_min:0", ())
		scale_max_ = self.get_and_check_tensor("constants/distribution/scale_max:0", ())
		scale_ = self.get_and_check_tensor("transition/deviation/scale:0", [self.batch_size, 1])
		scale_min_, scale_max_, scale_ = self.sess.run([scale_min_, scale_max_, scale_])
		self.assertAlmostEqual(scale_min_, np.pi / 180 * self.model.alpha_min, places=5)
		self.assertAlmostEqual(scale_max_, np.pi / 180 * self.model.alpha_max, places=5)
		self.assertTrue(np.all(scale_ >= scale_min_))
		self.assertTrue(np.all(scale_ <= scale_max_))

	def test_transition_computes_distribution_scale_as_linear_function_of_velocity_when_sampling_deviations(self):
		velocity_ = self.get_and_check_tensor("transition/deviation/velocity/Sqrt:0", [self.batch_size, 1])
		scale_ = self.get_and_check_tensor("transition/deviation/scale:0", [self.batch_size, 1])
		scale_min_ = self.get_and_check_tensor("constants/distribution/scale_min:0", ())
		scale_max_ = self.get_and_check_tensor("constants/distribution/scale_max:0", ())
		velocity_, scale_, scale_min_, scale_max_ = self.sess.run([velocity_, scale_, scale_min_, scale_max_])
		for i in range(len(scale_)):
			if scale_[i] > scale_min_:
				self.assertAlmostEqual(float(scale_[i] / velocity_[i]), float(scale_max_ / np.sqrt(2)), places=5)

	def test_transition_samples_deviation_angle_with_correct_shape_and_type(self):
		alpha_ = self.get_and_check_tensor("transition/deviation/noise_1/alpha/Reshape:0", [self.batch_size, 1])

	def test_transition_applies_rotation_matrix(self):
		alpha_ = self.get_and_check_tensor("transition/deviation/noise_1/alpha/Reshape:0", [self.batch_size, 1])
		cos_alpha_ = self.get_and_check_tensor("transition/direction/cos_alpha:0", [self.batch_size, 1])

		alpha_, cos_alpha_ = self.sess.run([alpha_, cos_alpha_])
		for i in range(len(alpha_)):
			self.assertAlmostEqual(float(cos_alpha_[i]), float(np.cos(alpha_[i])), places=5)	

		action_ = self.get_and_check_tensor("action:0", [self.batch_size, self.grid["ndim"]])
		noisy_action_ = self.get_and_check_tensor("transition/direction/noisy_action:0", [self.batch_size, self.grid["ndim"]])

		cos_alpha_ = self.get_and_check_tensor("transition/direction/cos_alpha:0", [self.batch_size, 1])
		action_, noisy_action_, cos_alpha_ = self.sess.run([action_, noisy_action_, cos_alpha_])
		for i in range(len(action_)):
			actual_cos = float(np.dot(action_[i], noisy_action_[i]) / (np.linalg.norm(action_[i]) * np.linalg.norm(noisy_action_[i])))
			expected_cos = float(cos_alpha_[i])
			self.assertAlmostEqual(actual_cos, expected_cos, places=5)

	def test_transition_decelerates_action(self):
		d_ = self.get_and_check_tensor("transition/deceleration/d:0", [self.batch_size, 1])
		deceleration_ = self.get_and_check_tensor("transition/deceleration/sub_1:0", [self.batch_size, 1])
		decelerated_noisy_direction_ = self.get_and_check_tensor("transition/deceleration/decelerated_noisy_direction:0", [self.batch_size, self.grid["ndim"]])
		noisy_action_ = self.get_and_check_tensor("transition/direction/noisy_action:0", [self.batch_size, self.grid["ndim"]])
		d_, deceleration_, decelerated_noisy_direction_, noisy_action_ = self.sess.run([d_, deceleration_, decelerated_noisy_direction_, noisy_action_])
		for i in range(len(noisy_action_)):
			actual_velocity = float(np.linalg.norm(decelerated_noisy_direction_[i]))
			expected_velocity = float(deceleration_[i] * np.linalg.norm(noisy_action_[i]))
			self.assertAlmostEqual(actual_velocity, expected_velocity, places=5)

	def test_transition_computes_next_position_in_grid(self):
		p_ = self.get_and_check_tensor("transition/next_position/p:0", [self.batch_size, self.grid["ndim"]])
		state_ = self.get_and_check_tensor("state:0", [self.batch_size, self.grid["ndim"]])
		next_state_ = self.get_and_check_tensor("transition/next_position/next_state:0", [self.batch_size, self.grid["ndim"]])
		decelerated_noisy_direction_ = self.get_and_check_tensor("transition/deceleration/decelerated_noisy_direction:0", [self.batch_size, self.grid["ndim"]])
		p_, state_, next_state_, decelerated_noisy_direction_ = self.sess.run([p_, state_, next_state_, decelerated_noisy_direction_])

		for i in range(len(p_)):
			actual_next_p_x, actual_next_p_y = float(p_[i][0]), float(p_[i][1])
			expected_next_p = state_[i] + decelerated_noisy_direction_[i]
			expected_next_p_x, expected_next_p_y = float(expected_next_p[0]), float(expected_next_p[1])
			self.assertAlmostEqual(actual_next_p_x, expected_next_p_x, places=5)
			self.assertAlmostEqual(actual_next_p_y, expected_next_p_y, places=5)

			actual_next_state_x = float(next_state_[i][0])
			if actual_next_p_x > self.min_size and actual_next_p_x < self.max_size:
				self.assertEqual(expected_next_p_x, actual_next_state_x)
			elif actual_next_p_x <= self.min_size:
				self.assertAlmostEqual(actual_next_state_x, self.min_size)
			elif actual_next_p_x >= self.max_size:
				self.assertAlmostEqual(actual_next_state_x, self.max_size)

			actual_next_state_y = float(next_state_[i][1])
			if actual_next_p_y > self.min_size and actual_next_p_y < self.max_size:
				self.assertEqual(expected_next_p_y, actual_next_state_y)
			elif actual_next_p_y <= self.min_size:
				self.assertAlmostEqual(actual_next_state_y, self.min_size)
			elif actual_next_p_y >= self.max_size:
				self.assertAlmostEqual(actual_next_state_y, self.max_size)

	def test_reward_proportional_to_distance_to_goal_position(self):
		r_ = self.get_and_check_tensor("reward/Neg:0", (self.batch_size, 1))
		state_ = self.get_and_check_tensor("state:0", [self.batch_size, self.grid["ndim"]])
		goal = self.grid["goal"]
		state_, r_ = self.sess.run([state_, r_])
		for i in range(len(state_)):
			actual_reward = float(r_[i])
			expected_reward = float(-np.linalg.norm(state_[i] - goal))
			self.assertAlmostEqual(actual_reward, expected_reward, places=5)
