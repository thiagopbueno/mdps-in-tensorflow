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

from ..mdp import MDP

import numpy as np
import tensorflow as tf


class Navigation(MDP):
    """
    Navigation: base class for the 2D navigation domain,
    in which an agent is supposed to get to a goal position
    from a start position as fast as possible.

    :param graph: computation graph
    :type graph: tf.Graph
    :param config: problem-dependent configuration
    :type config: dict
    """

    def __init__(self, graph, config):
        self.graph = graph

        self.grid = config["grid"]
        self.ndim = config["grid"]["ndim"]

        with self.graph.as_default():

            with tf.name_scope("constants/grid"):
                min_size, max_size = config["grid"]["size"]
                self._grid_lower_bound = tf.constant(min_size, dtype=tf.float32, name="grid_lower_bound")
                self._grid_upper_bound = tf.constant(max_size, dtype=tf.float32, name="grid_upper_bound")
                self._goal = tf.constant(config["grid"]["goal"], dtype=tf.float32, name="goal")

            with tf.name_scope("constants/deceleration"):
                deceleration = config["grid"]["deceleration"][0]
                self._center = tf.constant(deceleration["center"], dtype=tf.float32, name="center")
                self._decay  = tf.constant(deceleration["decay"],  dtype=tf.float32, name="decay")
                self._1_00 = tf.constant(1.00, dtype=tf.float32)
                self._2_00 = tf.constant(2.00, dtype=tf.float32)

    @property
    def action_size(self):
        return self.ndim

    @property
    def state_size(self):
        return self.ndim

    def reward(self, state, action):
        with self.graph.as_default():
            with tf.name_scope("reward"):
                r = -tf.sqrt(tf.reduce_sum(tf.square(state - self._goal), axis=1, keep_dims=True), name="r")
        return r


class DeterministicNavigation(Navigation):
    """
    DeterministicNavigation: an agent is supposed to get to a goal position
    from a start position, subject to deceleration zones.

    :param graph: computation graph
    :type graph: tf.Graph
    :param config: problem-dependent configuration
    :type config: dict
    """

    def __init__(self, graph, config):
        super().__init__(graph, config)

    def transition(self, state, action):

        with self.graph.as_default():

            with tf.name_scope("transition"):

                with tf.name_scope("deceleration"):
                    d = tf.sqrt(tf.reduce_sum(tf.square(state - self._center), 1, keep_dims=True), name="d")
                    deceleration = self._2_00 / (self._1_00 + tf.exp(-self._decay * d)) - self._1_00
                    decelerated_action = tf.multiply(deceleration, action, name="decelerated_action")

                with tf.name_scope("next_position"):
                    p = tf.add(state, decelerated_action, name="p")
                    next_state = tf.clip_by_value(p, self._grid_lower_bound, self._grid_upper_bound, name="next_state")

        return next_state


class StochasticNavigation(Navigation):
    """
    Navigation 2D domain: an agent is supposed to get to a goal position
    from a start position, subject to noisy directions and deceleration zones.

    :param graph: computation graph
    :type graph: tf.Graph
    :param config: problem-dependent configuration
    :type config: dict
    """

    def __init__(self, graph, config):
        super().__init__(graph, config)

        with self.graph.as_default():
            with tf.name_scope("constants/distribution"):
                self.alpha_min, self.alpha_max = config["alpha_min"], config["alpha_max"]
                self._scale_min = tf.constant(2 * np.pi / 360 * self.alpha_min, dtype=tf.float32, name="scale_min")
                self._scale_max = tf.constant(2 * np.pi / 360 * self.alpha_max, dtype=tf.float32, name="scale_max")

    def transition(self, state, action):

        with self.graph.as_default():

            with tf.name_scope("transition"):

                with tf.name_scope("deviation"):
                    velocity = tf.norm(action, axis=1, keep_dims=True, name="velocity")
                    max_velocity = tf.sqrt(2.0, name="max_velocity")
                    scale = tf.maximum(self._scale_min, self._scale_max / max_velocity * velocity, name="scale")
                    loc = tf.constant(0.0, name="loc")
                    noise = tf.distributions.Normal(loc=loc, scale=scale, name="noise")
                    alpha = noise.sample(name="alpha")

                with tf.name_scope("direction"):
                    cos, sin = tf.cos(alpha, name="cos_alpha"), tf.sin(alpha, name="sin_alpha")
                    rotation_matrix = tf.stack([cos, -sin, sin, cos], axis=1)
                    rotation_matrix = tf.reshape(rotation_matrix, [-1, 2, 2], name="rotation_matrix")
                    noisy_action = tf.matmul(rotation_matrix, tf.reshape(action, [-1, 2, 1]))
                    noisy_action = tf.reshape(noisy_action, [-1, 2], name="noisy_action")

                with tf.name_scope("deceleration"):
                    d = tf.sqrt(tf.reduce_sum(tf.square(state - self._center), 1, keep_dims=True), name="d")
                    deceleration = self._2_00 / (self._1_00 + tf.exp(-self._decay * d)) - self._1_00
                    decelerated_noisy_direction = tf.multiply(deceleration, noisy_action, name="decelerated_noisy_direction")

                with tf.name_scope("next_position"):
                    p = tf.add(state, decelerated_noisy_direction, name="p")
                    next_state = tf.clip_by_value(p, self._grid_lower_bound, self._grid_upper_bound, name="next_state")

        return next_state, None


class NoisyNavigation(Navigation):

    def __init__(self, graph, config):
        super().__init__(graph, config)

        with self.graph.as_default():
            with tf.name_scope("constants/distribution"):
                self._scale_max = tf.constant(0.1, dtype=tf.float32, name="scale_max")
                self._max_velocity = tf.sqrt(2.0, name="max_velocity")

    def transition(self, state, action):

        with self.graph.as_default():

            with tf.name_scope("transition/deceleration"):
                d = tf.sqrt(tf.reduce_sum(tf.square(state - self._center), 1, keep_dims=True), name="d")
                deceleration = self._2_00 / (self._1_00 + tf.exp(-self._decay * d)) - self._1_00
                decelerated_action = tf.multiply(deceleration, action, name="decelerated_action")

            with tf.name_scope("transition/next_position"):
                p = tf.add(state, decelerated_action, name="p")
                velocity = tf.norm(action, axis=1, keep_dims=True, name="velocity")
                scale = tf.multiply(self._scale_max / self._max_velocity, velocity, name="scale")
                next_state_dist = tf.distributions.Normal(loc=p, scale=scale, name="next_state_dist")
                next_state_sampled = next_state_dist.sample(name="next_state_sampled")
                next_state = tf.clip_by_value(
                                next_state_sampled,
                                self._grid_lower_bound, self._grid_upper_bound,
                                name="next_state")
                next_state_log_prob = next_state_dist.log_prob(next_state_sampled, name="next_state_log_prob")

        return next_state, next_state_log_prob
