import matplotlib.pyplot as plt
import numpy as np

# Data visualization helper functions

def plot_loss_function(ax, losses, epoch, title=None):
    ax.plot(losses, 'b-')
    ax.set_xlim(0, epoch)
    if title is None:
        ax.set_title('Loss function')
    else:
        ax.set_title('Loss function ({})'.format(title))
    ax.set_xlabel("# iterations")
    ax.set_ylabel("loss")
    ax.grid()

def plot_average_total_cost(ax, totals, epoch, title=None):
    ax.plot(totals, 'b-')
    ax.set_xlim(0, epoch)
    if title is None:
        ax.set_title('Average Total Cost')
    else:
        ax.set_title('Average Total Cost ({})'.format(title))
    ax.set_xlabel("# iterations")
    ax.set_ylabel("total")
    ax.grid()

def plot_total_cost_per_batch(ax, total_cost):
    ax.hist(total_cost, normed=True, histtype='stepfilled')
    ax.set_title('Distribution of cumulative cost per batch')
    ax.set_xlabel('total')
    ax.set_ylabel('density')

def plot_grid(ax, grid):
    xlim, ylim = grid['size']
    ax.axis([0.0, xlim, 0.0, ylim])
    ax.set_aspect('equal')
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.grid()
    
def plot_deceleration(ax, grid, deceleration, npoints=1000):
    xlim, ylim = grid['size']
    xcenter, ycenter = deceleration['center']
    X, Y = np.meshgrid(np.linspace(0.0, xlim, npoints), np.linspace(0.0, ylim, npoints))
    D = np.sqrt((X - xcenter) ** 2 + (Y - ycenter) ** 2)
    Lambda = 2 / (1 + np.exp(-deceleration['decay'] * D)) - 1.00
    ticks = np.arange(0.0, 1.01, 0.10)
    cp = ax.contourf(X, Y, Lambda, ticks, cmap=plt.cm.bone)
    cp = ax.contour(X, Y, Lambda, ticks, colors='black', linestyles='dashed')

def plot_policy(ax, grid, deceleration, action_grid, actions, timestep):

    # title
    ax.set_title("Policy (step = {})".format(timestep), fontweight="bold", fontsize=16)
    
    # plot grid
    plot_grid(ax, grid)

    # plot deceleration
    plot_deceleration(ax, grid, deceleration)
        
    # plot action_grid
    initial_states_x = [ p[0] for p in action_grid ]
    initial_states_y = [ p[1] for p in action_grid ]
    ax.plot(initial_states_x, initial_states_y, 'g.')
 
    # plot actions
    ax.quiver(initial_states_x, initial_states_y, actions[:, 0], actions[:, 1],
              angles='xy', scale_units='xy', scale=1, color='dodgerblue', width=0.005,
              label='actions')

    # plot goal
    end = grid['goal']
    ax.plot([end[0]], [end[1]], marker='X', markersize=15, color='crimson', label='goal')

def plot_trajectory(ax, grid, deceleration, start, end, s_series, a_series):

    # plot grid
    plot_grid(ax, grid)

    # plot deceleration
    plot_deceleration(ax, grid, deceleration)

    # plot actions
    positions = np.concatenate([[start], s_series])
    ax.quiver(positions[:-1, 0], positions[:-1, 1], a_series[:, 0], a_series[:, 1],
              angles='xy', scale_units='xy', scale=1, color='dodgerblue', width=0.005,
              label='actions')

    # plot states
    ax.plot(positions[:, 0], positions[:, 1], '-', marker='o', color='darkblue', markersize=8,
            label='states')

    # plot start and end positions
    ax.plot([start[0]], [start[1]], marker='X', markersize=15, color='limegreen', label='initial')
    ax.plot([end[0]], [end[1]], marker='X', markersize=15, color='crimson', label='goal')

def plot_simulations(fig, grid, deceleration, initial_states, delta_y, states, actions):
    start, end = grid['start'], grid['goal']
    num_plots = int(initial_states.shape[0])
    deltas = [0] + delta_y
    rows = len(delta_y)
    cols = num_plots // len(delta_y)
    for i in range(num_plots):
        ax = fig.add_subplot(len(deltas), num_plots/len(delta_y), i + 1)
        idx = i//cols
        start = (grid['start'][0], grid['start'][1] + delta_y[idx])
        plot_trajectory(ax, grid, deceleration, start, end, states[i], actions[i])
