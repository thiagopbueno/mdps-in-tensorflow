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

import numpy as np
import tensorflow as tf


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
                losses.append(loss)
                totals.append(total)

                # show information
                if show_progress and epoch_idx % 10 == 0:
                    print('Epoch {0:5}: loss = {1}\r'.format(epoch_idx, loss, total), end='')

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
            self.train_step = self._optimizer.minimize(self.loss)

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
                if show_progress:
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
