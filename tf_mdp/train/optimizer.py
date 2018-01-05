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


class PolicyGradientOptimizer():

    def __init__(self, graph, policy, trajectory, learning_rate, gamma):
        self.graph = graph
        self.policy = policy
        self.trajectory = trajectory
        self.learning_rate = learning_rate
        self.gamma = gamma

        with self.graph.as_default():
            self.max_time = len(self.trajectory.states)
            self.batch_size = self.trajectory.states[0].shape[0].value
            self._compute_loss()
            self._compute_gradients()
            self._apply_gradients()

    def _compute_loss(self):
        discount_schedule = np.geomspace(1, self.gamma ** (self.max_time - 1), self.max_time, dtype=np.float32)
        discount_schedule = np.repeat([discount_schedule], self.batch_size, axis=0)
        discount_schedule = np.reshape(discount_schedule, (self.batch_size, self.max_time, 1))
        self.discount_schedule = tf.constant(discount_schedule, dtype=tf.float32, name="discount_schedule")
        self.total = tf.reduce_sum(tf.stack(self.trajectory.rewards, axis=1) * self.discount_schedule, axis=1, name="total_discount_reward")
        self.loss = tf.reduce_mean(self.total, axis=0, name="loss")

    def _compute_gradients(self):
        Q = tf.cumsum(self.trajectory.rewards, exclusive=True, reverse=True, name="Q")
        baseline = tf.reduce_mean(Q, axis=1, name="baseline")
        print(Q)
        print(baseline)

        self.grads_rewards = defaultdict(list)
        self.grads_future_rewards = defaultdict(list)
        self.grads_and_vars = []

        params = self.policy.params

        discount = 1.0
        for t in range(self.max_time):

            with tf.name_scope("t{}/".format(t)):

                state = self.trajectory.states[t]
                reward = self.trajectory.rewards[t]
                next_state = self.trajectory.next_states[t]
                log_prob = self.trajectory.log_probs[t]

                average_reward = tf.multiply(discount, tf.reduce_mean(reward), name="average_reward")
                grads_reward = tf.gradients(ys=average_reward, xs=params, stop_gradients=[state, next_state], name="grads_reward")

                log_prob = tf.reduce_sum(log_prob, axis=1)
                average_log_prob_future_reward = tf.reduce_mean(log_prob * (Q[t] - baseline[t]), name="average_log_prob_future_reward")
                grads_future_reward = tf.gradients(ys=average_log_prob_future_reward, xs=params, stop_gradients=[state, next_state, Q], name="grads_future_reward")

                discount *= self.gamma

                for j, param in enumerate(params):
                    if grads_reward[j] is not None:
                        self.grads_rewards[param].append(grads_reward[j])
                    self.grads_future_rewards[param].append(grads_future_reward[j])

        with tf.name_scope("grads"):
            for j, param in enumerate(params):
                grad_1 = self.grads_rewards[param]
                grad_2 = self.grads_future_rewards[param]
                grad = tf.reduce_sum(tf.stack(grad_1 + grad_2), axis=0, name="grad")
                self.grads_and_vars.append((grad, param))
                assert(grad.shape == param.shape)

    def _apply_gradients(self):
        with tf.name_scope("update"):
            updates = []
            for grad, var in self.grads_and_vars:
                updates.append(var.assign(var + self.learning_rate * grad))
            self.train_op = tf.group(*updates, name="train_op")

    def minimize(self, epochs, show_progress=True):
        losses = []
        start = time.time()
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(epochs):
                loss, _ = sess.run([self.loss, self.train_op])
                losses.append(loss[0])
                if show_progress and step % 10 == 0:
                    print('Epoch {0:5}: loss = {1:.6f}\r'.format(step, loss[0]), end='')
        end = time.time()
        uptime = end - start
        print("\nDone in {0:.6f} sec.\n".format(uptime))
        return losses, uptime


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

            losses, totals = [], []
            for epoch_idx in range(epoch):

                # backprop and update weights
                _, loss, total = sess.run([self.train_step, self.loss, self.total])

                # store results
                losses.append(loss[0])
                totals.append(total)

                # show information
                if show_progress and epoch_idx % 10 == 0:
                    print('Epoch {0:5}: loss = {1:.6f}\r'.format(epoch_idx, loss[0]), end='')

        end = time.time()
        uptime = end - start
        print("\nDone in {0:.6f} sec.\n".format(uptime))

        return losses, totals, uptime


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

        return losses, best_batch, uptime
