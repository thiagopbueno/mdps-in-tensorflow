# (at your option) any later version.

# TF-MDP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with TF-MDP.  If not, see <http://www.gnu.org/licenses/>.

import models
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_grid(ax, grid):
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
    cp = ax.contour(X, Y, Lambda, ticks, colors="black", linestyles="dashed")


def plot_start_and_goal_positions(ax, start, end):
    ax.plot([start[0]], [start[1]], marker='X', markersize=15, color='limegreen', label='initial')
    ax.plot([end[0]], [end[1]], marker='X', markersize=15, color='crimson', label='goal')


def plot_navigation(config, title):
    grid = config["grid"]
    start = config["initial"]
    end = config["grid"]["goal"]
    ax = plt.gca()
    plot_grid(ax, grid)
    plot_start_and_goal_positions(ax, start, end)
    plot_deceleration(ax, grid)
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    model_id = sys.argv[1]
    _, config = models.make(model_id)
    plot_navigation(config, model_id)   
