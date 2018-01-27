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


class ActionCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, mdp):
        self.mdp = mdp

    def input_size(self):
        return self.mdp.action_size

    @property
    def state_size(self):
        return self.mdp.state_size

    @property
    def output_size(self):
        return self.mdp.state_size + 1

    def __call__(self, inputs, state, scope=None):
        with self.mdp.graph.as_default():

            with tf.name_scope("transition_cell"):
                action = inputs
                next_state_dist = self.mdp.transition(state, action)
                next_state = next_state_dist.loc

            with tf.name_scope("reward_cell"):
                reward = self.mdp.reward(state, action)

            with tf.name_scope("output"):
                outputs = tf.concat([reward, next_state], axis=1, name="outputs")

        return outputs, next_state


class ActionOptimizer(object):

    def __init__(self, mdp, start, plan, learning_rate, limits=None):
        self.mdp = mdp
        self.plan = plan
        self.start = start

        self.cell = ActionCell(mdp)

        self.limits = limits
        self.learning_rate = learning_rate

        with self.mdp.graph.as_default():
            self._initial_state()
            self._limits()
            self._trajectory()
            self._loss()
            self._train_op()

    def _initial_state(self):
        batch_size = self.plan.shape[0].value
        initial_state = np.repeat([self.start], batch_size, axis=0).astype(np.float32)
        self.initial_state = tf.constant(initial_state, name="initial_state")

    def _limits(self):
        with tf.name_scope("action_optimizer/limits"):
            self.enforce_action_limits = None
            if self.limits is not None:
                self.enforce_action_limits = tf.assign(
                                                self.plan,
                                                tf.clip_by_value(self.plan,
                                                                 self.limits[0],
                                                                 self.limits[1]),
                                                name="action_limits")

    def _trajectory(self):
        max_time = self.plan.shape[1]

        input_size = self.cell.input_size
        state_size = self.mdp.state_size
        action_size = self.mdp.action_size

        outputs, self.final_state = tf.nn.dynamic_rnn(
                                        self.cell,
                                        self.plan,
                                        initial_state=self.initial_state,
                                        dtype=tf.float32,
                                        scope="action_optimizer/rnn")

        with tf.name_scope("action_optimizer/rnn/outputs"):
            outputs = tf.unstack(outputs, axis=2)
            self.rewards = tf.reshape(outputs[0], [-1, max_time, 1])
            self.states  = tf.stack(outputs[1 : ], axis=2)

    def _loss(self):
        with tf.name_scope("action_optimizer/loss"):
            self.total = tf.reduce_sum(self.rewards, axis=1, name="total")
            self.loss = tf.reduce_mean(tf.square(self.total), name="mse") # Mean-Squared Error (MSE)

    def _train_op(self):
        with tf.name_scope("action_optimizer"):
            self._optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.train_op = self._optimizer.minimize(self.loss)

    def minimize(self, epoch, show_progress=True):
        start = time.time()
        with tf.Session(graph=self.mdp.graph) as sess:

            sess.run(tf.global_variables_initializer())

            losses = []
            for epoch_idx in range(epoch):

                sess.run(self.train_op)

                if self.enforce_action_limits is not None:
                    sess.run(self.enforce_action_limits)

                loss = sess.run(self.loss)
                losses.append(loss)
                if show_progress and epoch_idx % 10 == 0:
                    print('Epoch {0:5}: loss = {1}\r'.format(epoch_idx, loss), end="")

            with tf.name_scope("best_batch"):
                best_batch_idx = tf.argmax(self.total, axis=0, name="best_batch_index")
                best_batch_idx = sess.run(best_batch_idx)
                best_batch = {
                    "total":   np.squeeze(sess.run(self.total)[best_batch_idx]).tolist(),
                    "actions": np.squeeze(sess.run(self.plan)[best_batch_idx]).tolist(),
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
