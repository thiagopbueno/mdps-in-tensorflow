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
import logz
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
    parser.add_argument("--trials",        "-ts", type=int,   default=5)
    parser.add_argument("--log-dir",       "-lg", type=str,   default="/tmp/")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    show_parameters_info(args)
    return args


def show_parameters_info(args):
    print()
    print(">> Optimizing {} ...".format(args.model))
    print(">> Hyperparameters:")
    print("timestep      = {}".format(args.timesteps))
    print("batch size    = {}".format(args.batch_size))
    print("discount      = {}".format(args.discount))
    print("epochs        = {}".format(args.epochs))
    print("learning rate = {}".format(args.learning_rate))
    print("trials        = {}".format(args.trials))
    print()


def load_model(model_id):
    graph = tf.Graph()
    model, config = models.make(model_id)
    mdp = model(graph, config)
    return mdp, config


def run_planner(planner, args):
    trials = {}
    for i in range(args.trials):
        print(">> Starting trial #{} ...".format(i + 1))
        mdp, config = load_model(args.model)
        losses = planner.run(
                    mdp, config,
                    args.timesteps, args.batch_size,
                    args.discount, args.epochs, args.learning_rate,
                    baseline=args.baseline)
        trials["run{}".format(i)] = losses
    return trials


def save_results(log_dir, trials):
    print(">> Logging results to {} ...".format(log_dir))
    logz.logging(log_dir, trials)


if __name__ == '__main__':
    args = parse_args()

    if args.mode == "pg":
        trials = run_planner(pgplan, args)
    elif args.mode == "st":
        trials = run_planner(stplan, args)
    elif args.mode == "det":
        trials = run_planner(detplan, args)

    save_results(args.log_dir, trials)
