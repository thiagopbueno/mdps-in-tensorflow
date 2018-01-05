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

import sys
import abc
import numpy as np
import tensorflow as tf
import tf_mdp.models.mdp as mdp

class Reservoir(mdp.TF_MDP, metaclass=abc.ABCMeta):
    """
    Reservoir Control: the agent control multiple connected
    reservoirs. Each state is a sequence of water levels in each
    reservoir. An action is a flow from the each reservoir to the next
    downstream reservoir. The objective is to maintain the water level
    of each reservoir in a safe range and as close to half of its
    capacity as possible.

    :param graph: computation graph
    :type graph: tf.Graph
    :param system: parameters of reservoir system
    :type system: dict
    :param environment: parameters of the environment as rain and evaporation
    :type environment: dict
    """
    def __init__(self,
                 graph,
                 system,
                 environment):

        self.graph = graph

        self.rain_mean = environment["rain_mean"]
        self.rain_std = environment["rain_std"]
        self.e_t_std = environment["evaporation_std"]

        self.n_reservoirs = system["n_reservoirs"]
        self.lower = np.array(system['lower_bounds'], dtype="float32")
        self.upper = np.array(system['upper_bounds'], dtype="float32")
        self.halfs = (self.upper + self.lower) / 2.0
        self.max_capacity_largest = np.max(self.upper - self.lower)
        self.max_cap = float(sys.maxsize)

        with self.graph.as_default():
            with tf.name_scope("system"):
                self.adjacency_matrix = tf.constant(system['adjacency_matrix'],
                                                    dtype="float32",
                                                    name="adjacency_matrix")

    @property
    def action_size(self):
        return self.n_reservoirs

    @property
    def state_size(self):
        return self.n_reservoirs

    @abc.abstractmethod
    def evaporation(self, state):
        """
        Returns evaporation levels as a function of current state.

        :param state: MDP state
        :type state: tf.Tensor(shape=(batch_size, state_size))
        :rtype: tf.Tensor(shape=(batch_size, state_size))
        """
        return

    def transition(self, state, action):
        """
        Takes one step on the MDP as defined in the paper:

        T(s_t, a_t) = s_t + r_t - e_t - a_t + downstream

        :param state: MDP state
        :type state: tf.Tensor(shape=(batch_size,
                                      self.state_size),
                               dtype=float32)
        :param action: MDP action
        :type state: tf.Tensor(shape=(batch_size,
                                      self.action_size),
                               dtype=float32)
        :rtype: tf.Tensor(shape=(batch_size,
                                 self.state_size),
                          dtype=float32)
        """
        state_shape = state.get_shape()
        with self.graph.as_default():

            with tf.name_scope("transition"):

                # sample rain and evaporation values
                with tf.name_scope("random_variables"):

                    loc_e_t = self.evaporation(state)
                    rain_noise = tf.distributions.Normal(loc=self.rain_mean,
                                                         scale=self.rain_std,
                                                         name="rain_noise")
                    eva_noise = tf.distributions.Normal(loc=loc_e_t,
                                                        scale=self.e_t_std,
                                                        name="eva_noise")
                    r_t = rain_noise.sample(state_shape, name="r_t")
                    e_t = eva_noise.sample(name="e_t")

                # compute next state
                with tf.name_scope("next_position"):
                    # calculate the quantity flowing downstream
                    downstream = tf.tensordot(action,
                                              self.adjacency_matrix,
                                              axes=1,
                                              name="downstream")

                    downstream = tf.reshape(downstream, state_shape, name="downstream_final")

                    # next position
                    new_state = state + r_t - e_t - action + downstream

                    # avoid getting negative values
                    new_state = tf.clip_by_value(new_state,
                                                 clip_value_min=0.0,
                                                 clip_value_max=self.max_cap,
                                                 name="next_state")

        return new_state

    def reward(self, state, action):
        """
        Calculates the reward.

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

                # calculating the safe range
                __zeros = tf.zeros(state.get_shape(),
                                   dtype=tf.float32,
                                   name="zeros")
                lower_comparison = tf.greater_equal(state,
                                                    self.lower,
                                                    name="leq_than_expected")
                upper_comparison = tf.greater_equal(self.upper,
                                                    state,
                                                    name="geq_than_expected")
                in_bounds = tf.logical_and(lower_comparison,
                                           upper_comparison,
                                           name="in_bounds")
                below_bounds = tf.less(state,
                                       self.lower,
                                       name="l_than_expected")
                rewards = tf.where(in_bounds,
                                   __zeros,
                                   tf.where(below_bounds,
                                            - 5 * (self.lower - state),
                                            - 100 * (state - self.upper)),
                                   name="rewards_safe_range")

                rewards += tf.abs(self.halfs - state) * (- 0.1)
                rewards = tf.reduce_sum(rewards,
                                        1,
                                        keep_dims=True,
                                        name="final_reward")

        return rewards


class ReservoirNonLinear(Reservoir):
    """
    An reservoir wiht

    :param graph: computation graph
    :type graph: tf.Graph
    :param reserv_dict: specific parameters of the problem
    :type reserv_dict: dict
    """

    def __init__(self, graph, system, environment):
        super().__init__(graph, system, environment)

    def evaporation(self, state):
        sin_e_t = tf.sin(state / self.max_capacity_largest,
                         name="non_linear")
        loc_e_t = tf.multiply(0.5 * state, sin_e_t,
                              name="evaporation_loc")
        return loc_e_t


class ReservoirLinear(Reservoir):
    """
    Class that encodes a mnp reservoir scenario

    :param graph: computation graph
    :type graph: tf.Graph
    :param reserv_dict: specific parameters of the problem
    :type reserv_dict: dict
    """
    def __init__(self, graph, system, environment):
        super().__init__(graph, system, environment)

    def evaporation(self, state):
        loc_e_t = 0.1 * state
        return loc_e_t
