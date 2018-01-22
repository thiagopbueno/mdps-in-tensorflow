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

import collections
import numpy as np
import tensorflow as tf


Trajectory = collections.namedtuple("Trajectory", "timesteps states actions rewards next_states")


class MarkovCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    @property
    def input_size(self):
        return 1

    @property
    def state_size(self):
        return self.mdp.state_size

    @property
    def output_size(self):
        return 2 * self.mdp.state_size + self.mdp.action_size + 2

    def __call__(self, inputs, state, scope=None):

        with self.mdp.graph.as_default():

            with tf.name_scope("policy_cell"):
                timestep = inputs
                state_t = tf.concat([state, timestep], axis=1, name="state_t")
                action = self.policy(state_t)

            with tf.name_scope("transition_cell"):
                next_state_dist = self.mdp.transition(state, action)
                next_state = next_state_dist.sample(name="next_state")

            with tf.name_scope("reward_cell"):
                reward = self.mdp.reward(state, action)

            with tf.name_scope("output"):
                outputs = tf.concat([timestep, state, action, reward, next_state], axis=1)

        return outputs, next_state


class MarkovRecurrentModel(object):

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy
        self._cell = MarkovCell(mdp, policy)

    def unroll(self, initial_state, inputs):

        max_time = inputs.shape[1]

        input_size = self._cell.input_size
        state_size = self._cell.mdp.state_size
        action_size = self._cell.mdp.action_size

        with self.mdp.graph.as_default():

            self.initial_state = tf.constant(initial_state, name="initial_state")
            self.inputs = tf.constant(inputs, name="inputs")

            outputs, final_state = tf.nn.dynamic_rnn(
                                    self._cell,
                                    self.inputs,
                                    initial_state=self.initial_state,
                                    dtype=tf.float32,
                                    scope="recurrent")

            with tf.name_scope("recurrent/trajectory"):
                timestep_idx = 0
                state_idx = timestep_idx + 1
                action_idx = state_idx + state_size
                reward_idx = action_idx + action_size
                next_state_idx = reward_idx + 1

                outputs = tf.unstack(outputs, axis=2)

                timesteps = tf.reshape(outputs[timestep_idx], [-1, max_time, 1], name="timesteps")
                states = tf.stack(outputs[state_idx : action_idx], axis=2, name="states")
                actions = tf.stack(outputs[action_idx: reward_idx], axis=2, name="actions")
                rewards = tf.reshape(outputs[reward_idx], [-1, max_time, 1], name="rewards")
                next_states = tf.stack(outputs[next_state_idx : ], axis=2, name="next_states")

        return Trajectory(timesteps=timesteps, states=states, actions=actions, rewards=rewards, next_states=next_states)
