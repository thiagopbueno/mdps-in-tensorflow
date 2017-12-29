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


class TF_MDP(metaclass=abc.ABCMeta):
    """
    Interface to define models in TF-MDP.

    A well-formed model must specify the MDP's dynamics and reward
    functions, as well as the dimensions of its actions and states.
    """

    @abc.abstractproperty
    def action_size(self):
        """
        Returns the dimensionality of action space.

        :rtype: float
        """
        return

    @abc.abstractproperty
    def state_size(self):
        """
        Returns the dimensionality of state space.

        :rtype: float
        """
        return

    @abc.abstractmethod
    def transition(self, state, action):
        """
        Executes action in current state and returns next state.

        :param state: MDP state
        :type state: tf.Tensor(shape=(batch_size, state_size))
        :param action: MDP action
        :type action: tf.Tensor(shape=(batch_size, action_size))
        :rtype: tf.Tensor(shape=(batch_size, state_size))
        """
        return

    @abc.abstractmethod
    def reward(self, state, action):
        """
        Returns reward as a function of current state and action.

        :param state: MDP state
        :type state: tf.Tensor(shape=(batch_size, state_size))
        :param action: MDP action
        :type action: tf.Tensor(shape=(batch_size, action_size))
        :rtype: tf.Tensor(shape=(batch_size, 1))
        """
        return
