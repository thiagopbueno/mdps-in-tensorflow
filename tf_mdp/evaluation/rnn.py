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

import tensorflow as tf


class MarkovCell(tf.nn.rnn_cell.RNNCell, metaclass=abc.ABCMeta):

    def __init__(self, mdp):
        with mdp.graph.as_default():
            super().__init__()

        self.mdp = mdp

    @abc.abstractproperty
    def input_size(self):
        raise NotImplementedError

    @property
    def state_size(self):
        return self.mdp.state_size

    @property
    def output_size(self):
        return self.mdp.state_size + self.mdp.action_size + 1

    @abc.abstractmethod
    def __call__(self, inputs, state, scope=None):
        raise NotImplementedError


class DeterministicMarkovCell(MarkovCell):

    def __init__(self, mdp):
        super().__init__(mdp)

    def input_size(self):
        return self.mdp.action_size

    def __call__(self, inputs, state, scope=None):
        with self.mdp.graph.as_default():

            with tf.name_scope("transition_cell"):
                action = inputs
                next_state = self.mdp.transition(state, action)

            with tf.name_scope("reward_cell"):
                reward = self.mdp.reward(next_state, action)

            with tf.name_scope("output"):
                outputs = tf.concat([reward, next_state, action], axis=1, name="outputs")

        return outputs, next_state


class StochasticMarkovCell(MarkovCell):
    """
    MDP_RNNCell: memory cell that encapsulates the components of MDP's
    dynamics and reward functions, and integrates the policy network
    into a single timestep transition.

    :param mdp: MDP model
    :type mdp: tf_mdp.models.mdp.MDP
    :param policy: policy approximator
    :type policy: tf_mdp.policy.PolicyNetwork
    """

    def __init__(self, mdp, policy):
        with tf.name_scope("mdp_cell"):
            super().__init__(mdp)

        self.policy = policy

    @property
    def input_size(self):
        return 1

    def __call__(self, inputs, state, scope=None):

        with self.mdp.graph.as_default():

            with tf.name_scope("policy_cell"):
                timestep = inputs
                state_t = tf.concat([state, timestep], axis=1, name="state_t")
                action = self.policy(state_t)

            with tf.name_scope("transition_cell"):
                next_state, _ = self.mdp.transition(state, action)

            with tf.name_scope("reward_cell"):
                reward = self.mdp.reward(next_state, action)

            with tf.name_scope("output"):
                outputs = tf.concat([reward, next_state, action], axis=1)

        return outputs, next_state


class MarkovRecurrentModel(object):
    """
    MarkovRecurrentModel: recurrent model that implements
    batch trajectory sampling using RNN truncated dynamic unrolling
    from initial state.

    :param cell: MDP cell to build recurrent stochastic graph
    :type cell: tf_mdp.evaluation.rnn.MarkovCell
    """

    def __init__(self, cell):
        self.cell = cell

    def unroll(self, initial_state, inputs):
        self.inputs = inputs

        max_time = inputs.shape[1]
        input_size = self.cell.input_size
        state_size = self.cell.mdp.state_size
        action_size = self.cell.mdp.action_size

        with self.cell.mdp.graph.as_default():

            initial_state_initializer = tf.constant(initial_state, name="initial_state_initializer")
            self.initial_state = tf.placeholder_with_default(
                                    initial_state_initializer,
                                    shape=(None, state_size),
                                    name="initial_state")

            outputs, final_state = tf.nn.dynamic_rnn(
                                    self.cell,
                                    inputs,
                                    initial_state=self.initial_state,
                                    dtype=tf.float32,
                                    scope="mdp_rnn")

            with tf.name_scope("mdp_rnn/outputs"):
                outputs = tf.unstack(outputs, axis=2)
                rewards = tf.reshape(outputs[0], [-1, max_time, 1])
                states  = tf.stack(outputs[1 : 1 + state_size], axis=2)
                actions = tf.stack(outputs[1 + state_size : 1 + state_size + action_size], axis=2)

        return rewards, states, actions
