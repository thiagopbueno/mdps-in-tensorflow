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

import models
from policy.deterministic import DeterministicPolicyNetwork
from train.optimizer import ActionOptimizer, SGDPolicyOptimizer
from evaluation.rnn import DeterministicMarkovCell, MarkovRecurrentModel
from evaluation.montecarlo import MCPolicyEvaluation
from utils.plot import plot_losses

import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stochastic", "deterministic"], default="stochastic")
    parser.add_argument("--model",         "-m",  type=str)
    parser.add_argument("--timesteps",     "-t",  type=int,   default=10)
    parser.add_argument("--batch-size",    "-b",  type=int,   default=1000)
    parser.add_argument("--discount",      "-d",  type=float, default=0.9)
    parser.add_argument("--epochs",        "-e",  type=int,   default=200)
    parser.add_argument("--learning-rate", "-lr", type=float, default=1.0e-5)
    return parser.parse_args()


def show_info(args):
    print()
    print(">> Optimizing {} ...".format(args.model))
    print(">> Hyperparameters:")
    print("timestep      = {}".format(args.timesteps))
    print("batch size    = {}".format(args.batch_size))
    print("discount      = {}".format(args.discount))
    print("epochs        = {}".format(args.epochs))
    print("learning rate = {}".format(args.learning_rate))
    print()


def run_deterministic_model(model, timesteps, batch_size, discount, epochs, learning_rate):
    graph = tf.Graph()

    # MDP
    model, config = models.make(model)
    mdp = model(graph, config)

    # Action variables
    with graph.as_default():
        actions = tf.Variable(
            tf.truncated_normal(shape=[batch_size, timesteps, mdp.state_size], stddev=0.05),
            name="actions")

    # Trajectory evaluation
    cell = DeterministicMarkovCell(mdp)
    rnn = MarkovRecurrentModel(cell)
    initial_state = np.repeat([config["initial"]], batch_size, axis=0).astype(np.float32)
    rewards, states, _ = rnn.unroll(initial_state, actions)
    
    # loss
    with graph.as_default():
        total = tf.reduce_sum(rewards, axis=1, name="total")
        loss = tf.reduce_mean(tf.square(total), name="loss") # Mean-Squared Error (MSE)

    # ActionOptimizer
    metrics = {
        "loss":  loss,
        "total": total,
        "states":  states,
        "actions": rnn.inputs,
        "rewards": rewards
    }
    optimizer = ActionOptimizer(graph, metrics, learning_rate, config["limits"])
    return optimizer.minimize(epochs)


def run_stochastic_model(model, timesteps, batch_size, discount, epochs, learning_rate):
    graph = tf.Graph()

    # MDP
    model, config = models.make(model)
    mdp = model(graph, config)

    # PolicyNetwork
    shape = [mdp.state_size + 1, 20, 5, mdp.action_size]
    policy = DeterministicPolicyNetwork(graph, shape)

    # PolicyEvaluation
    mc = MCPolicyEvaluation(mdp, policy,
                                initial_state=config["initial"],
                                max_time=timesteps,
                                batch_size=batch_size,
                                gamma=discount)

    # PolicyOptimizer
    metrics = {
        "loss":  mc.expected_return,
        "total": mc.total
    }
    optimizer = SGDPolicyOptimizer(graph, metrics, learning_rate)
    return optimizer.minimize(epochs)


def report_results(losses):
    plot_losses(losses)


if __name__ == '__main__':
    args = parse_args()
    show_info(args)

    if args.mode == "stochastic":
        losses, totals, uptime = run_stochastic_model(
                                    args.model,
                                    args.timesteps,
                                    args.batch_size,
                                    args.discount,
                                    args.epochs,
                                    args.learning_rate)
    elif args.mode == "deterministic":
        losses, best_batch, uptime = run_deterministic_model(
                                    args.model,
                                    args.timesteps,
                                    args.batch_size,
                                    args.discount,
                                    args.epochs,
                                    args.learning_rate)

    report_results(losses)
