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
                    print('Epoch {0:5}: loss = {1}\r'.format(epoch_idx, loss), end="")

            # index of best solution among all planners
            with tf.name_scope("best_batch"):
                best_batch_idx = tf.argmax(self.total, axis=0, name="best_batch_index")
                best_batch_idx = sess.run(best_batch_idx)
                best_batch = {
                    "total":   np.squeeze(sess.run(self.total)[best_batch_idx]).tolist(),
                    "actions": np.squeeze(sess.run(self.actions)[best_batch_idx]).tolist(),
                    "states":  np.squeeze(sess.run(self.states)[best_batch_idx]).tolist(),
                    "rewards": np.squeeze(sess.run(self.rewards)[best_batch_idx]).tolist()
                }
                print("\nBest batch total = {0:.4f}".format(best_batch["total"]))

        end = time.time()
        uptime = end - start
        print("Done in {0:.4f} sec.\n".format(uptime))

        return {
            "losses": losses,
            "solution": best_batch,
            "uptime": uptime
        }
