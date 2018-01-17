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

Trajectory = collections.namedtuple("Trajectory", "states actions rewards next_states log_probs")

class MarkovRecurrentModel(object):

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    def __call__(self, state, steps_to_go, max_time):
        timestep = max_time - steps_to_go

        with self.mdp.graph.as_default():

            with tf.name_scope("t{}/".format(timestep)):

                with tf.name_scope("policy_cell"):
                    batch_size = state.shape[0].value
                    steps_to_go = tf.fill([batch_size, 1], float(steps_to_go), name="steps_to_go")
                    state_t = tf.concat([state, steps_to_go], axis=1, name="state_t")
                    action = self.policy(state_t)

                with tf.name_scope("transition_cell"):
                    next_state, log_prob = self.mdp.transition(state, action)

                with tf.name_scope("reward_cell"):
                    reward = self.mdp.reward(next_state, action)

        return action, reward, next_state, log_prob

    def unroll(self, start, batch_size, max_time):

        states = []
        actions = []
        rewards = []
        next_states = []
        log_probs = []

        with self.mdp.graph.as_default():

            initial_state = np.repeat([start], batch_size, axis=0).astype(np.float32)
            initial_state = tf.constant(initial_state, name="initial_state")

            state = initial_state
            for steps_to_go in range(max_time, 0, -1):
                action, reward, next_state, log_prob = self(state, steps_to_go, max_time)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                log_probs.append(log_prob)
                state = next_state

        return Trajectory(states=states, actions=actions, rewards=rewards, next_states=next_states, log_probs=log_probs)
