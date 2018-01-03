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
        self.mdp = mdp
        self.policy = policy

    @property
    def action_size(self):
        return self.mdp.action_size

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

            # augment state by adding timestep to state vector
            state_t = tf.concat([state, timestep], axis=1)

            # add policy network with augmented state as input
            action = self.policy(state_t)

            # add MDP components to the RNN cell output
            next_state = self.mdp.transition(state, action)
            reward = self.mdp.reward(next_state, action)

            # concatenate outputs
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
        self.cell = MDP_RNNCell(mdp, policy)

    def unroll(self, initial_state, timesteps):

        inputs = timesteps

        max_time = int(inputs.shape[1])
        state_size = self.cell.state_size
        action_size = self.cell.action_size

        with self.graph.as_default():

            # timesteps
            self.inputs = tf.placeholder_with_default(tf.constant(timesteps, name='timesteps'),
                                                 shape=(None, max_time, 1),
                                                 name='inputs')
            # initial cell state
            self.initial_state = tf.placeholder_with_default(tf.constant(initial_state),
                                                        shape=(None, self.cell.state_size),
                                                        name='initial_state')

            # dynamic time unrolling
            outputs, final_state = tf.nn.dynamic_rnn(
                self.cell,
                inputs,
                initial_state=initial_state,
                dtype=tf.float32)

            # gather reward, state and action series
            outputs = tf.unstack(outputs, axis=2)
            rewards = tf.reshape(outputs[0], [-1, max_time, 1])
            states  = tf.stack(outputs[1:1+state_size], axis=2)
            actions = tf.stack(outputs[1+state_size:1+state_size+action_size],  axis=2)

        return rewards, states, actions, final_state
