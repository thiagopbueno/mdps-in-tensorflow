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
from planning import detplan, stplan, pgplan
from utils.plot import plot_losses

import argparse
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pg", "st", "det"])
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


def load_model(model_id):
    graph = tf.Graph()
    model, config = models.make(model_id)
    mdp = model(graph, config)
    start = config["initial"]
    return mdp, start, config


def run_policy_gradient_planner(mdp, start, args):
    return pgplan.run(
                    mdp, start,
                    args.timesteps, args.batch_size,
                    args.discount, args.epochs, args.learning_rate)

def run_stochastic_planner(mdp, start, args):
    return stplan.run(
                    mdp, start,
                    args.timesteps, args.batch_size,
                    args.discount, args.epochs, args.learning_rate)

def run_deterministic_planner(mdp, start, args, config):
    return detplan.run(
                    mdp, start, config["limits"],
                    args.timesteps, args.batch_size,
                    args.discount, args.epochs, args.learning_rate)


def report_results(losses):
    plot_losses(losses)


if __name__ == '__main__':
    args = parse_args()
    show_info(args)

    mdp, start, config = load_model(args.model)

    if args.mode == "pg":
        losses, _ = run_policy_gradient_planner(mdp, start, args)
    elif args.mode == "st":
        losses, _, _ = run_stochastic_planner(mdp, start, args)
    elif args.mode == "det":
        losses, _, _ = run_deterministic_planner(mdp, start, args, config)

    report_results(losses)