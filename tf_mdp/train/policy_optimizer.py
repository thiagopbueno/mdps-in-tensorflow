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

import numpy as np
import tensorflow as tf
import time


class PolicyOptimizer(object):
    """
    PolicyOptimizer: interface for implementations of policy network optimizers.

    :param graph: computation graph
    :type graph: tf.Graph
    :param loss: loss function to be optimized
    :type loss: tf.Tensor(shape(1,))
    :param learning_rate: optimization hyperparameter
    :type learning_rate: float
    """

    def __init__(self, graph, trajectory, learning_rate, gamma):
        self.graph = graph
        self.trajectory = trajectory
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.batch_size = self.trajectory.states.shape[0].value
        self.max_time = self.trajectory.states.shape[1].value

        with self.graph.as_default():
            with tf.name_scope("policy_optimizer"):
                self._discount_schedule()
                self._total_discounted_reward()
                self._loss()
                self._train_op()

    def _discount_schedule(self):
        discount_schedule = np.geomspace(1, self.gamma ** (self.max_time - 1), self.max_time, dtype=np.float32)
        discount_schedule = np.repeat([discount_schedule], self.batch_size, axis=0)
        discount_schedule = np.reshape(discount_schedule, (self.batch_size, self.max_time, 1))
        self.discount_schedule = tf.constant(discount_schedule, dtype=tf.float32, name="discount_schedule")

    def _total_discounted_reward(self):
        self.total_discounted_reward = tf.reduce_sum(self.trajectory.rewards * self.discount_schedule, axis=1, name="total_discount_reward")

    def _loss(self):
        self.loss = tf.reduce_mean(self.total_discounted_reward, axis=0, name="loss")

    def _train_op(self):
        self._optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = self._optimizer.minimize(-self.loss)

    def minimize(self, epoch, show_progress=True):

        start = time.time()
        with tf.Session(graph=self.graph) as sess:

            sess.run(tf.global_variables_initializer())

            losses = []
            for epoch_idx in range(epoch):

                # backprop and update weights
                _, loss = sess.run([self.train_op, self.loss])

                # store results
                losses.append(loss[0])

                # show information
                if show_progress and epoch_idx % 10 == 0:
                    print('Epoch {0:5}: loss = {1:.6f}\r'.format(epoch_idx, loss[0]), end='')

        end = time.time()
        uptime = end - start
        print("\nDone in {0:.6f} sec.\n".format(uptime))

        return {
            "losses": losses,
            "uptime": uptime
        }
