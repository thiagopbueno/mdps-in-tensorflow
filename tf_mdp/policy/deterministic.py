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

import tensorflow as tf

class DeterministicPolicyNetwork(object):
    """
    Deterministic Policy Network: implements a non-linear parametric approximator
    for the deterministic function a_t = pi(s_t) using a feedforward neural net with
    given architecture. Hidden layers use ReLUs and output layer uses tanh activation
    functions.

    :param graph: computation graph
    :type graph: tf.Graph
    :param shape: number of units in each layer
    :type shape: list of ints
    :param bounds: maximum absolute action value
    :type bounds: float
    """

    def __init__(self, graph, shape, bounds=1.0):
        self.graph = graph
        self.shape = shape
        self.bounds = bounds
    
    def __call__(self, state):
        """
        Return the prescribed action by the parametric policy for the given state.

        :param state: MDP state
        :type state: tf.Tensor(shape=(batch_size, state_size))
        :rtype: tf.Tensor(shape=(batch_size, action_size))
        """
        with self.graph.as_default():

            with tf.variable_scope('policy'):

                # input layer
                layer = state

                # hidden layers
                for i, n_h in enumerate(self.shape[1:-1]):
                    layer = tf.layers.dense(layer,
                                            units=n_h,
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.glorot_normal_initializer(),
                                            name="layer" + str(i + 1))

                # output layer
                layer = tf.layers.dense(layer,
                        units=self.shape[-1],
                        activation=tf.nn.tanh,
                        kernel_initializer=tf.glorot_normal_initializer(),
                        name="layer" + str(len(self.shape) - 1))

                # add action bounds over last tanh layer
                bounds = tf.constant(self.bounds, name="bounds")
                action = tf.multiply(bounds, layer, name="action")

        return action
