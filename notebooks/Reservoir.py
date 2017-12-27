import numpy as np
import tensorflow as tf
from MDP import MDP


class Reservoir(MDP):
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

                lower_comparison = tf.greater_equal(states,
                                        lower)
                upper_comparison = tf.greater_equal(upper,
                                        states)

                in_bounds = tf.logical_and(lower_comparison, upper_comparison)

                below_bounds = tf.less(states, lower) 

                rewards = tf.where(in_bounds,
                                   __zeros,
                                  tf.where(below_bounds,
                                           - 5 * (self.lower - states),
                                           - 100 * (states - self.upper)))


                rewards += tf.abs(self.halfs - states) * (-0.1)
                rewards = tf.reduce_sum(rewards, 1, keep_dims=True)

        return rewards
