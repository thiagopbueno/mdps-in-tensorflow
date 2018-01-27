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

import tensorflow as tf


class Navigation(MDP):
    """
    Navigation: an agent is supposed to get to a goal position
    from a start position as fast as possible while avoiding
    deceleration zones.

    :param graph: computation graph
    :type graph: tf.Graph
    :param config: problem-dependent configuration
    :type config: dict
    """

    def __init__(self, graph, config):
        self.graph = graph
        self.config = config

        self.grid = config["grid"]
        self.ndim = config["grid"]["ndim"]
        self.deceleration = config["grid"]["deceleration"]

        with self.graph.as_default():

            with tf.name_scope("constants/grid"):
                min_size, max_size = config["grid"]["size"]
                self._grid_lower_bound = tf.constant(min_size, dtype=tf.float32, name="grid_lower_bound")
                self._grid_upper_bound = tf.constant(max_size, dtype=tf.float32, name="grid_upper_bound")
                self._goal = tf.constant(config["grid"]["goal"], dtype=tf.float32, name="goal")

            with tf.name_scope("constants/deceleration"):
                self._1_00 = tf.constant(1.00, dtype=tf.float32)
                self._2_00 = tf.constant(2.00, dtype=tf.float32)
                self._center = []
                self._decay = []
                for i, deceleration in enumerate(self.deceleration):
                    self._center.append(tf.constant(deceleration["center"], dtype=tf.float32, name="center_{}".format(i)))
                    self._decay.append(tf.constant(deceleration["decay"],  dtype=tf.float32, name="decay_{}".format(i)))

            with tf.name_scope("constants/next_position"):
                self._scale_max = tf.constant(0.1, dtype=tf.float32, name="scale_max")
                self._max_velocity = tf.sqrt(2.0, name="max_velocity")

    @property
    def action_size(self):
        return self.ndim

    @property
    def state_size(self):
        return self.ndim

    def transition(self, state, action):

        with self.graph.as_default():

            with tf.name_scope("transition/deceleration"):
                deceleration = self._1_00
                for i in range(len(self.deceleration)):
                    d = tf.sqrt(tf.reduce_sum(tf.square(state - self._center[i]), 1, keep_dims=True), name="d_{}".format(i))
                    deceleration *= self._2_00 / (self._1_00 + tf.exp(-self._decay[i] * d)) - self._1_00
                decelerated_action = tf.multiply(deceleration, action, name="decelerated_action")

            with tf.name_scope("transition/next_position"):
                p = tf.add(state, decelerated_action, name="p")
                loc = tf.clip_by_value(p, self._grid_lower_bound, self._grid_upper_bound, name="loc")

                v = tf.norm(action, axis=1, keep_dims=True, name="v")
                scale = tf.multiply(self._scale_max / self._max_velocity, v, name="scale")

                next_state_dist = tf.distributions.Normal(loc=loc, scale=scale, name="next_state_dist")

        return next_state_dist

    def reward(self, state, action):
        with self.graph.as_default():
            with tf.name_scope("reward"):
                r = -tf.sqrt(tf.reduce_sum(tf.square(state - self._goal), axis=1, keep_dims=True), name="r")
        return r
