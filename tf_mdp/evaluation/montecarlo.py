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

from . import utils
from .mdp_rnn import MDP_RNN

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

        with self.mdp.graph.as_default():
            with tf.name_scope("policy_evaluation"):
                self._build_trajectory_ops()
                self._build_evaluation_ops()

    def _build_trajectory_ops(self):
        self.__initial_state = utils.initial_state(self.initial_state, self.batch_size)
        self.__timesteps = utils.timesteps(self.batch_size, self.max_time)
        self.__rnn = MDP_RNN(self.mdp, self.policy)
        self.rewards, self.states, self.actions, _ = self.__rnn.unroll(self.__initial_state, self.__timesteps)

    def _build_evaluation_ops(self):
        discount_schedule = np.geomspace(1, self.gamma ** (self.max_time - 1), self.max_time, dtype=np.float32)
        discount_schedule = np.repeat([discount_schedule], self.batch_size, axis=0)
        discount_schedule = np.reshape(discount_schedule, (self.batch_size, self.max_time, 1))

        self.discount_schedule = tf.constant(discount_schedule, dtype=tf.float32, name="discount_schedule")
        self.total = tf.reduce_sum(self.rewards * self.discount_schedule, axis=1, name="total_discount_reward")
        self.expected_return = tf.reduce_mean(self.total, axis=0, name="expected_return")

    def _run(self, tensors, sess=None):
        if sess is None:
            with tf.Session(graph=self.mdp.graph) as sess:
                sess.run(tf.global_variables_initializer())
                return sess.run(tensors)
        else:
            return sess.run(tensors)

    def eval(self, sess=None):
        """
        Returns a list of arrays containing:
            1. an estimate of the value of initial state;
            2. the total discounted reward of each episode in batch, and
            3. rewards series for given horizon.

        :param sess: current session, if none, starts a new session
        :type sess: tf.Session
        :rtype: list(np.array)
        """
        return self._run([self.expected_return, self.total, self.rewards], sess)

    def sample(self, sess=None):
        """
        Returns a list of arrays containing samples of:
            1. state series for given horizon;
            2. action series for given hofizon; and
            3. rewards series for given horizon.

        :param sess: current session, if none, starts a new session
        :type sess: tf.Session
        :rtype: list(np.array)
        """
        return self._run([self.states, self.actions, self.rewards], sess)
