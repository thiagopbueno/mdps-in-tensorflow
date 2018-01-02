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
import sys

class Reservoir_non_linear(mdp.TF_MDP):
    """
    Class that encodes a mnp reservoir scenario

    :param graph: computation graph
    :type graph: tf.Graph
    :param reserv_dict: specific parameters of the problem
    :type reserv_dict: dict
    """
    def __init__(self,
                 graph,
                 reserv_dict):
        self.graph = graph
        self.n_reservoirs = reserv_dict["n_reservoirs"]
        self.rain_mean = reserv_dict["rain_mean"]
        self.rain_std = reserv_dict["rain_std"]
        self.e_t_std = reserv_dict["evaporation_std"]
        self.lower = np.array(reserv_dict['lower_bounds'], dtype="float32")
        self.upper = np.array(reserv_dict['upper_bounds'], dtype="float32")
        self.halfs = (self.upper + self.lower) / 2.0
        self.max_capacity_largest = np.max(self.upper - self.lower) 
        self.max_capacity = float(sys.maxsize)


        with self.graph.as_default():
            pass

    @property
    def action_size(self):
        return self.n_reservoirs
    
    @property
    def state_size(self):
        return self.n_reservoirs
        
    def transition(self, state, action):
        """
        Takes one step on the MDP.

        :param state: MDP state
        :type state: tf.Tensor
                     shape=(batch_size,
                            self.state_size)
                     dtype=float32
                     
        :param action: MDP action
        :type state: tf.Tensor
                     shape=(batch_size,
                            self.action_size)
                     dtype=float32

        :rtype: tf.Tensor
                shape=(batch_size, self.state_size)
                dtype=float32
        """
        state_shape = state.get_shape()
        with self.graph.as_default():
            with tf.name_scope("random_variables"):
                sin_e_t = tf.sin(state / self.max_capacity_largest)
                loc_e_t = tf.multiply(0.5 * state, sin_e_t)  
                rain_noise = tf.distributions.Normal(loc=self.rain_mean,
                                                     scale=self.rain_std)
                water_loss_noise = tf.distributions.Normal(loc=loc_e_t,
                                                           scale= self.e_t_std)
                r_t = rain_noise.sample(state_shape, seed=1)
                e_t = water_loss_noise.sample(seed=1)

            with tf.name_scope("transition"):
                new_state = state + r_t - e_t - action
                new_state = tf.clip_by_value(new_state,
                                             clip_value_min=0.0,
                                             clip_value_max=self.max_capacity)

        return new_state   
        

    def reward(self, state, action):
        """
        calculates the reward.

        :param state: MDP state
        :type state: tf.Tensor
                     shape=(batch_size,
                            self.state_size)
                     dtype=float32
                     
        :param action: MDP action
        :type action: None

        :rtype: tf.Tensor
                shape=(batch_size, 1)
                dtype=float32
        """
        with self.graph.as_default():

            
            with tf.name_scope("reward"):
                __zeros = tf.zeros(state.get_shape(), dtype=tf.float32)
                lower_comparison = tf.greater_equal(state,
                                                    self.lower)
                upper_comparison = tf.greater_equal(self.upper,
                                                    state)
                in_bounds = tf.logical_and(lower_comparison,
                                           upper_comparison)
                below_bounds = tf.less(state, self.lower) 
                rewards = tf.where(in_bounds,
                                   __zeros,
                                  tf.where(below_bounds,
                                           - 5 * (self.lower - state),
                                           - 100 * (state - self.upper)))
                rewards += tf.abs(self.halfs - state) * (-0.1)
                rewards = tf.reduce_sum(rewards, 1, keep_dims=True)

        return rewards
