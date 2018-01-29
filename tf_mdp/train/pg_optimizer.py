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

import math
import numpy as np
import tensorflow as tf
import time


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
        A_idx = next_state_idx + self.mdp.state_size

        with self.mdp.graph.as_default():

            with tf.name_scope("pg_celll/inputs"):
                inputs = tf.unstack(inputs, axis=1, name="inputs")
                discount = tf.reshape(inputs[discount_idx], [-1, 1], name="discount")
                timestep = tf.reshape(inputs[timestep_idx], [-1, 1], name="timestep")
                state = tf.stack(inputs[state_idx : next_state_idx], axis=1, name="state")
                next_state = tf.stack(inputs[next_state_idx : A_idx], axis=1, name="next_state")
                A = tf.reshape(inputs[A_idx : ], [-1, 1], name="A")

            with tf.name_scope("pg_cell/policy_cell"):
                state_t = tf.concat([state, timestep], axis=1, name="state_t")
                action = self.policy(state_t)

            with tf.name_scope("pg_cell/transition_cell"):
                next_state_dist = self.mdp.transition(state, action)
                log_prob = next_state_dist.log_prob(next_state, name="log_prob")

            with tf.name_scope("pg_cell/reward_cell"):
                reward = tf.multiply(discount, self.mdp.reward(state, action), name="discounted_reward")

            with tf.name_scope("pg_cell/outputs"):
                outputs = tf.concat([reward, log_prob, A], axis=1, name="outputs")

            with tf.name_scope("pg_cell/next_h"):
                state_log_prob = tf.reduce_sum(log_prob, axis=1, keep_dims=True, name="state_log_prob")
                weighted_log_prob = tf.multiply(A, state_log_prob, name="weighted_log_prob")
                next_h = tf.add(h, reward + weighted_log_prob * A, name="next_h")

        return outputs, next_h


class PolicyGradientOptimizer(object):

    def __init__(self, mdp, policy, trajectory, learning_rate, gamma, baseline=None):
        self.mdp = mdp
        self.policy = policy
        self.trajectory = trajectory
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.baseline = baseline

        self.batch_size = self.trajectory.states.shape[0].value
        self.max_time = self.trajectory.states.shape[1].value

        with self.mdp.graph.as_default():
            with tf.name_scope("pg_optimizer"):
                self._discount_schedule()
                self._discounted_reward()
                self._total_discounted_reward()
                self._baseline()
                self._reward_to_go()
                self._advantage()
                self._loss()
                self._surrogate_loss()
                self._train_op()

    def _discount_schedule(self):
        discount_schedule = np.geomspace(1, self.gamma ** (self.max_time - 1), self.max_time, dtype=np.float32)
        discount_schedule = np.repeat([discount_schedule], self.batch_size, axis=0)
        discount_schedule = np.reshape(discount_schedule, (self.batch_size, self.max_time, 1))
        self.discount_schedule = tf.constant(discount_schedule, dtype=tf.float32, name="discount_schedule")

    def _discounted_reward(self):
        self.discounted_reward = tf.multiply(self.trajectory.rewards, self.discount_schedule, name="discount_reward")

    def _total_discounted_reward(self):
        self.total_discounted_reward = tf.reduce_sum(self.discounted_reward, axis=1, name="total_discounted_reward")

    def _baseline(self):
        # self.baseline = tf.reduce_mean(self.Q, axis=0, name="baseline")
        if self.baseline is not None:
            # self.baseline = tf.constant(self.baseline, dtype=tf.float32)
            self.baseline = tf.reshape(self.baseline, [self.max_time, 1], name="baseline")

    def _reward_to_go(self):
        self.Q = tf.cumsum(self.discounted_reward, axis=1, exclusive=True, reverse=True, name="Q")

    def _advantage(self):
        self.A = self.Q
        if self.baseline is not None:
            # Q_mean, Q_var = tf.nn.moments(self.Q, axes=[0])
            # Q_std = tf.sqrt(Q_var)
            # # self.A = (self.Q - Q_mean) / tf.sqrt(Q_std)
            # self.A = -tf.abs(self.Q - Q_mean)

            self.A = (self.Q - self.baseline)

            self.A = -tf.abs((self.Q - self.baseline))

            # A_max = tf.reduce_max(tf.abs(self.Q - self.baseline), axis=0)
            # self.A = (self.Q - self.baseline) / A_max

            # A_max = tf.reduce_max(tf.abs(self.Q - self.baseline), axis=0)
            # self.A = -tf.abs(self.Q - self.baseline) / A_max

            # Q_min = tf.reduce_min(self.Q, axis=0, name="Q_min")
            # self.A = (self.Q - self.baseline) / (Q_min - self.baseline)

    def _loss(self):
        self.loss = tf.reduce_mean(self.total_discounted_reward, axis=0, name="loss")

    def _surrogate_loss(self):
        self._cell = PolicyGradientCell(self.mdp, self.policy)

        self._initial_state = tf.zeros([self.batch_size, 1], name="initial_state")

        timesteps_shape = [self.batch_size, self.max_time, 1]
        states_shape = [self.batch_size, self.max_time, self.mdp.state_size]
        A_shape = [self.batch_size, self.max_time, 1]
        self._timesteps = tf.placeholder(tf.float32, shape=timesteps_shape, name="timesteps")
        self._states = tf.placeholder(tf.float32, shape=states_shape, name="states")
        self._next_states = tf.placeholder(tf.float32, shape=states_shape, name="next_states")
        self._A = tf.placeholder(tf.float32, shape=A_shape, name="A")
        self._inputs = tf.concat([self.discount_schedule, self._timesteps, self._states, self._next_states, self._A], axis=2, name="inputs")

        outputs, final_h = tf.nn.dynamic_rnn(
                                self._cell,
                                self._inputs,
                                initial_state=self._initial_state,
                                dtype=tf.float32,
                                scope="recurrent")

        self.surrogate_loss = tf.reduce_mean(final_h, name="surrogate_loss")

    def _train_op(self):
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.surrogate_loss)
        # self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.surrogate_loss)

    def minimize(self, epochs, show_progress=True):
        losses = []

        start = time.time()
        with tf.Session(graph=self.mdp.graph) as sess:
            sess.run(tf.global_variables_initializer())

            print(">> Optimizing policy ...")
            for step in range(epochs):

                # Sample trajectories
                timesteps, states, next_states, A = sess.run([
                                                        self.trajectory.timesteps,
                                                        self.trajectory.states,
                                                        self.trajectory.next_states,
                                                        self.A])

                # Apply gradients and evaluate loss
                feed_dict = {
                    self._timesteps: timesteps,
                    self._states: states,
                    self._next_states: next_states,
                    self._A: A
                }
                loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

                if math.isnan(loss[0]):
                    break

                losses.append(loss[0])

                # Show progress
                if show_progress and step % 10 == 0:
                    print('Epoch {0:5}: loss = {1:.6f}\r'.format(step, loss[0]), end="")

        end = time.time()
        uptime = end - start
        print("\nDone in {0:.6f} sec.\n".format(uptime))

        return {
            "losses": losses,
            "uptime": uptime
        }
