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

class Reservoir_non_linear(mdp.TF_MDP):
    """
    Class that encodes a mnp reservoir scenario

    :type graph: tf.Graph
    :type reserv_dict: dict
    """
    def __init__(self,
                 graph,
                 reserv_dict):
        
        self.graph = graph

        self.n_reservoirs = reserv_dict["n_reservoirs"]
        
        # reservoir constants
        self.lower = np.array(reserv_dict['lower_bounds'], dtype="float32")
        self.upper = np.array(reserv_dict['upper_bounds'], dtype="float32")
        self.halfs = (self.upper + self.lower) / 2.0


        with self.graph.as_default():
            pass

            # reservoir constants


            # deceleration constants


            # numerical constants


    @property
    def action_size(self):
        return self.n_reservoirs
    
    @property
    def state_size(self):
        return self.n_reservoirs
        
    def transition(self, state, action, noise):
        pass

#        with self.graph.as_default():


#        return next_state

    def reward(self, state, action):
        """
        calculates the reward.

        :type stape: tf.Tensor
                     shape=(batch_size,
                            self.ndim)
                     dtype=float32

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

                in_bounds = tf.logical_and(lower_comparison, upper_comparison)

                below_bounds = tf.less(state, self.lower) 

                rewards = tf.where(in_bounds,
                                   __zeros,
                                  tf.where(below_bounds,
                                           - 5 * (self.lower - state),
                                           - 100 * (state - self.upper)))


                rewards += tf.abs(self.halfs - state) * (-0.1)
                rewards = tf.reduce_sum(rewards, 1, keep_dims=True)

        return rewards
