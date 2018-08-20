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

import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--results", "-r", type=str)
    return parser.parse_args()


def plot_grid_bounds(ax, grid):
    lower, upper = grid["size"]
    ax.axis([lower[0], upper[0], lower[1], upper[1]])
    ax.set_aspect("equal")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.grid()


def plot_deceleration(ax, grid, npoints=1000):
    lower, upper = grid["size"]
    X, Y = np.meshgrid(np.linspace(lower[0], upper[0], npoints), np.linspace(lower[1], upper[1], npoints))
    Lambda = 1.0
    for zone in grid["deceleration"]:
        xcenter, ycenter = zone["center"]
        D = np.sqrt((X - xcenter) ** 2 + (Y - ycenter) ** 2)
        Lambda *= 2 / (1 + np.exp(- zone["decay"] * D)) - 1.00
    ticks = np.arange(0.0, 1.01, 0.10)
    cp = ax.contourf(X, Y, Lambda, ticks, cmap=plt.cm.bone)
    plt.colorbar(cp, ticks=ticks)
    cp = ax.contour(X, Y, Lambda, ticks, colors="black", linestyles="dashed")


def plot_start_and_goal_positions(ax, start, end):
    ax.plot([start[0]], [start[1]], marker='X', markersize=15, color='limegreen', label='initial')
    ax.plot([end[0]], [end[1]], marker='X', markersize=15, color='crimson', label='goal')


def plot_state_trajectory(ax, start, path, deltas):
    xpath = [ p[0] for p in path ]
    ypath = [ p[1] for p in path ]
    ax.plot(xpath, ypath, 'b.')

    x0, y0 = start
    xdeltas = [ d[0] for d in deltas ]
    ydeltas = [ d[1] for d in deltas ]
    ax.quiver([x0] + xpath[:-1], [y0] + ypath[:-1], xdeltas, ydeltas,
        angles='xy', scale_units='xy', scale=1, color='dodgerblue', width=0.005,
        label='actions')

def plot_grid(ax, config):
    grid = config["grid"]
    start = config["initial"]
    end = config["grid"]["goal"]
    plot_grid_bounds(ax, grid)
    plot_start_and_goal_positions(ax, start, end)
    plot_deceleration(ax, grid)


def plot_results(ax, config, results):
    start = config["initial"]
    solution = results["run0"]["solution"]
    path = solution["states"]
    deltas = solution["actions"]
    plot_state_trajectory(ax, start, path, deltas)


if __name__ == '__main__':
    args = parse_args()

    model_id = args.model
    config = models.get_config(model_id)

    ax = plt.gca()

    plot_grid(ax, config)

    if args.results:
        results = logz.deserialize(args.results)
        plot_results(ax, config, results)

    plt.title(model_id, fontweight="bold")
    plt.legend(loc="lower right")
    plt.show()
