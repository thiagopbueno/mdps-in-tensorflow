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

import abc

class PolicyNetwork(metaclass=abc.ABCMeta):
    """
    Policy Network: interface for implementations of parametric policy approximator.

    :param graph: computation graph
    :type graph: tf.Graph
    :param shape: number of units in each layer
    :type shape: list of ints
    """

    def __init__(self, graph, shape):
        self.graph = graph
        self.shape = shape
        self.layers = []

    def count_params(self):
        """
        Returns the total number of network parameters.

        :rtype: float
        """
        return sum(layer.count_params() for layer in self.layers)

    @abc.abstractmethod
    def __call__(self, state):
        """ Returns prescribed action or action distribution for given state. """
        raise NotImplementedError
