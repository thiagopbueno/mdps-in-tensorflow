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

    def __init__(self, graph, loss, total, learning_rate):
        self.graph = graph

        # performance metrics
        self.loss = loss
        self.total = total

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
                if show_progress:
                    print('Epoch {0:5}: loss = {1}\r'.format(epoch_idx, loss, total), end='')

        end = time.time()
        uptime = end - start
        print("\nDone in {0:.6f} sec.\n".format(uptime))

        return losses, uptime


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

    def __init__(self, graph, loss, total, learning_rate):
        super().__init__(graph, loss, total, learning_rate)

    def _build_optimization_ops(self):
        learning_rate = self.hyperparameters["learning_rate"]
        with tf.name_scope("SGDPolicyOptimizer"):
            self.__sgd = tf.train.GradientDescentOptimizer(learning_rate)
            self.train_step = self.__sgd.minimize(self.loss)

    @property
    def learning_rate(self):
        """
        Returns hyperparameter learning rate as used by SGD optimizer.

        :rtype: float
        """
        return self.__sgd._learning_rate
