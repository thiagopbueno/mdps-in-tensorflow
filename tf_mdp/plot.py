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

import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

import logz


def load_results(file_path):
    results = logz.deserialize(file_path)
    return results


def plot_losses(data, **kwargs):
    for trial in data:
        line = data[trial]["losses"]
        plt.plot(line)
    plt.xlabel("Epochs")
    plt.ylabel("Loss function J = $V(\\mathbf{s}_0)$")
    plt.title(title)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    file_path = sys.argv[1]
    title = sys.argv[2]
    data = load_results(file_path)
    plot_losses(data, title=title)
