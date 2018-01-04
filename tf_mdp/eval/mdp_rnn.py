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

class MDP_RNNCell(tf.nn.rnn_cell.RNNCell):
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
        with mdp.graph.as_default():
            super().__init__()

        self.mdp = mdp
        self.policy = policy

    @property
    def state_size(self):
        return self.mdp.state_size

    @property
    def output_size(self):
        return self.mdp.state_size + self.mdp.action_size + 1

    def __call__(self, inputs, state, scope=None):

        with self.mdp.graph.as_default():

            # timestep
            timestep = inputs

            # add policy network with augmented state as input
            with tf.name_scope("policy_cell"):

                # augment state by adding timestep to state vector
                state_t = tf.concat([state, timestep], axis=1, name="state_t")

                # select action from policy
                action = self.policy(state_t)

            # add MDP components to the RNN cell output
            with tf.name_scope("transition_cell"):
                next_state = self.mdp.transition(state, action)

            with tf.name_scope("reward_cell"):
                reward = self.mdp.reward(next_state, action)

            # concatenate outputs
            with tf.name_scope("output"):
                outputs = tf.concat([reward, next_state, action], axis=1)

        return outputs, next_state


class MDP_RNN(object):
    """
    MDP_RNN: recurrent model that implements batch trajectory sampling
    using truncated dynamic unrolling from initial state given a fixed
    number of timesteps.

    :param mdp: MDP object to construct stochastic transition graph
    :type mdp: tf_mdp.models.mdp.MDP object
    :param policy: policy approximator
    :type policy: tf_mdp.policy.PolicyNetwork
    """

    def __init__(self, mdp, policy):
        self.graph = mdp.graph
        self.mdp = mdp

        with tf.name_scope("mdp_cell"):
            self.cell = MDP_RNNCell(mdp, policy)

    def unroll(self, initial_state, timesteps):

        inputs = timesteps

        max_time = inputs.shape[1]
        state_size = self.mdp.state_size
        action_size = self.mdp.action_size

        with self.graph.as_default():

            # timesteps
            inputs_initializer = tf.constant(timesteps, name='inputs_initializer')
            self.inputs = tf.placeholder_with_default(
                            inputs_initializer,
                            shape=(None, max_time, 1),
                            name='inputs')

            # initial cell state
            initial_state_initializer = tf.constant(initial_state, name="initial_state_initializer")
            self.initial_state = tf.placeholder_with_default(
                                    initial_state_initializer,
                                    shape=(None, state_size),
                                    name='initial_state')

            # dynamic time unrolling
            outputs, final_state = tf.nn.dynamic_rnn(
                                    self.cell,
                                    self.inputs,
                                    initial_state=self.initial_state,
                                    dtype=tf.float32,
                                    scope="mdp_rnn")

            # gather reward, state and action series
            with tf.name_scope("mdp_rnn/outputs"):
                outputs = tf.unstack(outputs, axis=2)
                rewards = tf.reshape(outputs[0], [-1, max_time, 1])
                states  = tf.stack(outputs[1 : 1 + state_size], axis=2)
                actions = tf.stack(outputs[1 + state_size : 1 + state_size + action_size], axis=2)

        return rewards, states, actions, final_state
