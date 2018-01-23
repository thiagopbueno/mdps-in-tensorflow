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
import time

from collections import defaultdict

import numpy as np
import tensorflow as tf


class PolicyGradientCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    @property
    def input_size(self):
        return 2 * self.mdp.state_size + 2

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.mdp.state_size + 2

    def __call__(self, inputs, h, scope=None):

        discount_idx = 0
        timestep_idx = 1
        state_idx = timestep_idx + 1
        next_state_idx = state_idx + self.mdp.state_size
        Q_idx = next_state_idx + self.mdp.state_size

        with self.mdp.graph.as_default():

            with tf.name_scope("pg_celll/inputs"):
                inputs = tf.unstack(inputs, axis=1, name="inputs")
                discount = tf.reshape(inputs[discount_idx], [-1, 1], name="discount")
                timestep = tf.reshape(inputs[timestep_idx], [-1, 1], name="timestep")
                state = tf.stack(inputs[state_idx : next_state_idx], axis=1, name="state")
                next_state = tf.stack(inputs[next_state_idx : Q_idx], axis=1, name="next_state")
                Q = tf.reshape(inputs[Q_idx : ], [-1, 1], name="Q")

            with tf.name_scope("pg_cell/policy_cell"):
                state_t = tf.concat([state, timestep], axis=1, name="state_t")
                action = self.policy(state_t)

            with tf.name_scope("pg_cell/transition_cell"):
                next_state_dist = self.mdp.transition(state, action)
                log_prob = next_state_dist.log_prob(next_state, name="log_prob")

            with tf.name_scope("pg_cell/reward_cell"):
                reward = tf.multiply(discount, self.mdp.reward(state, action), name="discounted_reward")

            with tf.name_scope("pg_cell/outputs"):
                outputs = tf.concat([reward, log_prob, Q], axis=1, name="outputs")

            with tf.name_scope("pg_cell/next_h"):
                state_log_prob = tf.reduce_sum(log_prob, axis=1, keep_dims=True, name="state_log_prob")
                weighted_log_prob = tf.multiply(Q, state_log_prob, name="weighted_log_prob")
                next_h = tf.add(h, reward + weighted_log_prob * Q, name="next_h")

        return outputs, next_h


class PolicyGradientOptimizer():

    def __init__(self, mdp, policy, trajectory, learning_rate, gamma):
        self.mdp = mdp
        self.policy = policy
        self.trajectory = trajectory
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.batch_size = self.trajectory.states.shape[0].value
        self.max_time = self.trajectory.states.shape[1].value

        with self.mdp.graph.as_default():
            self._discount_schedule()
            self._total_discounted_reward()
            self._reward_to_go()
            self._loss()
            self._surrogate_loss()
            self._train_op()

    def _discount_schedule(self):
        discount_schedule = np.geomspace(1, self.gamma ** (self.max_time - 1), self.max_time, dtype=np.float32)
        discount_schedule = np.repeat([discount_schedule], self.batch_size, axis=0)
        discount_schedule = np.reshape(discount_schedule, (self.batch_size, self.max_time, 1))
        self.discount_schedule = tf.constant(discount_schedule, dtype=tf.float32, name="discount_schedule")

    def _total_discounted_reward(self):
        self.total_discounted_reward = tf.multiply(self.trajectory.rewards, self.discount_schedule, name="total_discount_reward")

    def _reward_to_go(self):
        self.Q = tf.cumsum(self.total_discounted_reward, axis=1, exclusive=True, reverse=True, name="Q")
        self.baseline = tf.reduce_mean(self.Q, axis=0, name="baseline")

    def _loss(self):
        self.total = tf.reduce_sum(self.total_discounted_reward, axis=1, name="total")
        self.loss = tf.reduce_mean(self.total, axis=0, name="loss")

    def _surrogate_loss(self):
        self._cell = PolicyGradientCell(self.mdp, self.policy)

        self._initial_state = tf.zeros([self.batch_size, 1], name="initial_state")

        timesteps_shape = [self.batch_size, self.max_time, 1]
        states_shape = [self.batch_size, self.max_time, self.mdp.state_size]
        Q_shape = [self.batch_size, self.max_time, 1]
        self._timesteps = tf.placeholder(tf.float32, shape=timesteps_shape, name="timesteps")
        self._states = tf.placeholder(tf.float32, shape=states_shape, name="states")
        self._next_states = tf.placeholder(tf.float32, shape=states_shape, name="next_states")
        self._Q = tf.placeholder(tf.float32, shape=Q_shape, name="Q")
        self._inputs = tf.concat([self.discount_schedule, self._timesteps, self._states, self._next_states, self._Q], axis=2, name="inputs")

        outputs, final_h = tf.nn.dynamic_rnn(
                                self._cell,
                                self._inputs,
                                initial_state=self._initial_state,
                                dtype=tf.float32,
                                scope="recurrent")

        self.surrogate_loss = tf.reduce_mean(final_h, name="surrogate_loss")

    def _train_op(self):
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.surrogate_loss)

    def minimize(self, epochs, show_progress=True):
        losses = []

        start = time.time()
        with tf.Session(graph=self.mdp.graph) as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(epochs):

                # Sample trajectories
                timesteps, states, next_states, Q, baseline = sess.run([
                                                                self.trajectory.timesteps,
                                                                self.trajectory.states,
                                                                self.trajectory.next_states,
                                                                self.Q,
                                                                self.baseline])

                # Apply gradients and evaluate loss
                feed_dict = {
                    self._timesteps: timesteps,
                    self._states: states,
                    self._next_states: next_states,
                    self._Q: Q
                }
                loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                losses.append(loss[0])

                # Show progress
                if show_progress and step % 10 == 0:
                    print('Epoch {0:5}: loss = {1:.6f}\r'.format(step, loss[0]), end="")

        end = time.time()
        uptime = end - start
        print("\nDone in {0:.6f} sec.\n".format(uptime))

        return losses


class PolicyOptimizer(metaclass=abc.ABCMeta):
    """
    PolicyOptimizer: interface for implementations of policy network optimizers.

    :param graph: computation graph
    :type graph: tf.Graph
    :param loss: loss function to be optimized
    :type loss: tf.Tensor(shape(1,))
    :param learning_rate: optimization hyperparameter
    :type learning_rate: float
    """

    def __init__(self, graph, metrics, learning_rate):
        self.graph = graph

        # performance metrics
        self.loss = metrics["loss"]
        self.total = metrics["total"]

        # hyperparameters
        self.hyperparameters = {}
        self.hyperparameters["learning_rate"] = learning_rate

        with self.graph.as_default():
            self._build_optimization_ops()

    @abc.abstractmethod
    def _build_optimization_ops(self):
        raise NotImplementedError

    @abc.abstractproperty
    def learning_rate(self):
        raise NotImplementedError

    def minimize(self, epoch, show_progress=True):

        start = time.time()
        with tf.Session(graph=self.graph) as sess:

            sess.run(tf.global_variables_initializer())

            losses = []
            for epoch_idx in range(epoch):

                # backprop and update weights
                _, loss, total = sess.run([self.train_step, self.loss, self.total])

                # store results
                losses.append(loss[0])

                # show information
                if show_progress and epoch_idx % 10 == 0:
                    print('Epoch {0:5}: loss = {1:.6f}\r'.format(epoch_idx, loss[0]), end='')

        end = time.time()
        uptime = end - start
        print("\nDone in {0:.6f} sec.\n".format(uptime))

        return losses


class SGDPolicyOptimizer(PolicyOptimizer):
    """
    SGDPolicyOptimizer: policy network optimizer based on Stochastic Gradient Descent.

    :param graph: computation graph
    :type graph: tf.Graph
    :param loss: loss function to be optimized
    :type loss: tf.Tensor(shape(1,))
    :param learning_rate: optimization hyperparameter
    :type learning_rate: float
    """

    def __init__(self, graph, metrics, learning_rate):
        super().__init__(graph, metrics, learning_rate)

    def _build_optimization_ops(self):
        learning_rate = self.hyperparameters["learning_rate"]
        with tf.name_scope("SGDPolicyOptimizer"):
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.train_step = self._optimizer.minimize(-self.loss)

    @property
    def learning_rate(self):
        """
        Returns hyperparameter learning rate as used by SGD optimizer.

        :rtype: float
        """
        return self._optimizer._learning_rate


class ActionOptimizer(object):
    
    def __init__(self, graph, metrics, learning_rate, limits=None):
        self.graph = graph

        # performance metrics
        self.loss = metrics["loss"]
        self.total = metrics["total"]

        # trajectory metrics
        self.states = metrics["states"]
        self.actions = metrics["actions"]
        self.rewards = metrics["rewards"]

        # hyperparameters
        self.hyperparameters = {}
        self.hyperparameters["learning_rate"] = learning_rate
        self.hyperparameters["limits"] = limits

        with self.graph.as_default():
            self._build_optimization_ops()

    @property
    def learning_rate(self):
        """
        Returns hyperparameter learning rate.

        :rtype: float
        """
        return self._optimizer._learning_rate

    def _build_optimization_ops(self):
        limits = self.hyperparameters["limits"]
        self.enforce_action_limits = None
        if limits is not None:
            self.enforce_action_limits = tf.assign(
                                            self.actions,
                                            tf.clip_by_value(self.actions,
                                                            limits[0],
                                                            limits[1]), name="action_limits")

        learning_rate = self.hyperparameters["learning_rate"]
        with tf.name_scope("ActionOptimizer"):
            self._optimizer = tf.train.RMSPropOptimizer(learning_rate)
            self.train_step = self._optimizer.minimize(self.loss)

    def minimize(self, epoch, show_progress=True):
        start = time.time()
        with tf.Session(graph=self.graph) as sess:

            sess.run(tf.global_variables_initializer())

            losses = []
            for epoch_idx in range(epoch):
                # backprop and update weights
                sess.run(self.train_step)

                # maintain action constraints if any
                if self.enforce_action_limits is not None:
                    sess.run(self.enforce_action_limits)

                # store and show loss information
                loss = sess.run(self.loss)
                losses.append(loss)
                if show_progress and epoch_idx % 10 == 0:
                    print('Epoch {0:5}: loss = {1}\r'.format(epoch_idx, loss), end='')

            # index of best solution among all planners
            with tf.name_scope("best_batch"):
                best_batch_idx = tf.argmax(self.total, axis=0, name="best_batch_index")
                best_batch_idx = sess.run(best_batch_idx)
                best_batch = {
                    "total":   np.squeeze(sess.run(self.total)[best_batch_idx]),
                    "actions": np.squeeze(sess.run(self.actions)[best_batch_idx]),
                    "states":  np.squeeze(sess.run(self.states)[best_batch_idx]),
                    "rewards": np.squeeze(sess.run(self.rewards)[best_batch_idx])
                }

        end = time.time()
        uptime = end - start
        print("\nDone in {0:.6f} sec.\n".format(uptime))

        # return losses, best_batch, uptime
        return losses
