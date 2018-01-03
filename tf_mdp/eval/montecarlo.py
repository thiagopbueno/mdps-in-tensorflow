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

from tf_mdp import utils
from tf_mdp.eval.mdp_rnn import MDP_RNN

import numpy as np
import tensorflow as tf


class MCPolicyEvaluation(object):
    """
    MCPolicyEvaluation: implements a monte-carlo approximation of the
    value of policy for MDP's initial state.

    :param mdp: MDP model
    :type mdp: tf_mdp.models.mdp.MDP
    :param policy: policy approximator
    :type policy: tf_mdp.policy.PolicyNetwork
    :param initial_state: MDP initial state
    :type initial_state: np.array(shape=mdp.state_size)
    :param batch_size: number of trajectories in a batch
    :type batch_size: float
    """

    def __init__(self, mdp, policy, initial_state, max_time=100, batch_size=1000, gamma=0.99):
        self.mdp = mdp
        self.policy = policy

        self.initial_state = initial_state
        self.max_time = max_time
        self.batch_size = batch_size
        self.gamma = gamma

        self.__unroll_trajectories()
        self.__compute_expected_return()

    def __unroll_trajectories(self):
        self.__initial_state = utils.initial_state(self.initial_state, self.batch_size)
        self.__timesteps = utils.timesteps(self.batch_size, self.max_time)
        self.__rnn = MDP_RNN(self.mdp, self.policy)
        self.rewards, self.states, self.actions, _ = self.__rnn.unroll(self.__initial_state, self.__timesteps)

    def __compute_expected_return(self):
        with self.mdp.graph.as_default():
            with tf.name_scope("policy_evaluation"):
                discount_schedule = np.geomspace(1, self.gamma ** (self.max_time - 1), self.max_time, dtype=np.float32)
                discount_schedule = np.repeat([discount_schedule], self.batch_size, axis=0)
                discount_schedule = np.reshape(discount_schedule, (self.batch_size, self.max_time, 1))
                self.__discount_schedule = tf.constant(discount_schedule, dtype=tf.float32, name="discount_schedule")
                self.total = tf.reduce_sum(self.rewards * self.__discount_schedule, axis=1, name="total_discount_reward")
                self.expected_return = tf.reduce_mean(self.total, axis=0, name="expected_return")

                print(self.__discount_schedule)
                print(self.total)
                print(self.expected_return)

    def eval(self):
        with tf.Session(graph=self.mdp.graph) as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run([self.expected_return, self.total, self.rewards, self.states, self.actions])
