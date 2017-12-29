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

import tf_mdp.models.mdp as mdp

import numpy as np
import tensorflow as tf

class Navigation(mdp.TF_MDP):
    """
    Navigation 2D domain: an agent is supposed to get to a goal position
    from a start position, subject to noisy directions and deceleration zones,
    while maximizing its total reward.

    :param graph: computation graph
    :type graph: tf.Graph
    :param grid: spatial parameters of the problem
    :type grid: dict
    :param alpha_min: minimum angular deviation in degrees
    :type alpha_min: float
    :param alpha_max: maximum angular deviation in degrees
    :type alpha_max: float
    """

    def __init__(self, graph, grid, alpha_min=0.0, alpha_max=10.0):
        self.graph = graph

        self.ndim = grid["ndim"]
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        with self.graph.as_default():

            with tf.name_scope("constants/grid"):
                min_size, max_size = grid["size"]
                self.__grid_lower_bound = tf.constant(min_size, dtype=tf.float32, name="grid_lower_bound")
                self.__grid_upper_bound = tf.constant(max_size, dtype=tf.float32, name="grid_upper_bound")
                self.__goal = tf.constant(grid["goal"], dtype=tf.float32, name="goal")

            with tf.name_scope("constants/deceleration"):
                deceleration = grid["deceleration"]
                self.__center = tf.constant(deceleration["center"], dtype=tf.float32, name="center")
                self.__decay  = tf.constant(deceleration["decay"],  dtype=tf.float32, name="decay")
                self.__1_00 = tf.constant(1.00, dtype=tf.float32)
                self.__2_00 = tf.constant(2.00, dtype=tf.float32)

            with tf.name_scope("constants/distribution"):
                self.__scale_min = tf.constant(2 * np.pi / 360 * self.alpha_min, dtype=tf.float32, name="scale_min")
                self.__scale_max = tf.constant(2 * np.pi / 360 * self.alpha_max, dtype=tf.float32, name="scale_max")

    @property
    def action_size(self):
        return self.ndim
    
    @property
    def state_size(self):
        return self.ndim
        
    def transition(self, state, action):

        with self.graph.as_default():

            with tf.name_scope("transition"):

                # sample angular deviation
                with tf.name_scope("deviation"):
                    velocity = tf.norm(action, axis=1, keep_dims=True, name="velocity")
                    max_velocity = tf.sqrt(2.0, name="max_velocity")
                    scale = tf.maximum(self.__scale_min, self.__scale_max / max_velocity * velocity, name="scale")
                    loc = tf.constant(0.0, name="loc")
                    noise = tf.distributions.Normal(loc=loc, scale=scale, name="noise")
                    alpha = noise.sample(name="alpha")
                
                # apply angular deviation to generate noisy action
                with tf.name_scope("direction"):
                    cos, sin = tf.cos(alpha, name="cos_alpha"), tf.sin(alpha, name="sin_alpha")
                    rotation_matrix = tf.stack([cos, -sin, sin, cos], axis=1)
                    rotation_matrix = tf.reshape(rotation_matrix, [-1, 2, 2], name="rotation_matrix")
                    noisy_action = tf.matmul(rotation_matrix, tf.reshape(action, [-1, 2, 1]))
                    noisy_action = tf.reshape(noisy_action, [-1, 2], name="noisy_action")

                with tf.name_scope("deceleration"):
                    # distance to center of deceleration zone
                    d = tf.sqrt(tf.reduce_sum(tf.square(state - self.__center), 1, keep_dims=True), name="d")
                    # deceleration_factor
                    deceleration = self.__2_00 / (self.__1_00 + tf.exp(-self.__decay * d)) - self.__1_00
                    # decelerated noisy direction
                    decelerated_noisy_direction = tf.multiply(deceleration, noisy_action, name="decelerated_noisy_direction")
                
                # compute next state
                with tf.name_scope("next_position"):
                    # next position
                    p = tf.add(state, decelerated_noisy_direction, name="p")
                    # avoid getting out of map
                    next_state = tf.clip_by_value(p, self.__grid_lower_bound, self.__grid_upper_bound, name="next_state")

        return next_state

    def reward(self, state, action):

        with self.graph.as_default():

            with tf.name_scope("reward"):
                # norm L-2 (euclidean distance)
                r = -tf.sqrt(tf.reduce_sum(tf.square(state - self.__goal), axis=1, keep_dims=True), name="r")
                print(r)

        return r
