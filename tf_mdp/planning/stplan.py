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

from evaluation import utils
from evaluation.mrm import MarkovRecurrentModel
from policy.deterministic import DeterministicPolicyNetwork
from train.policy_optimizer import PolicyOptimizer

import tensorflow as tf


def run(mdp, config, max_time, batch_size, discount, epochs, learning_rate):

    # PolicyNetwork
    shape = [mdp.state_size + 1, 20, 5, mdp.action_size]
    policy = DeterministicPolicyNetwork(mdp.graph, shape)

    # MarkovRecurrentModel
    start = config["initial"]
    initial_state = utils.initial_state(start, batch_size)
    timesteps = utils.timesteps(batch_size, max_time)
    trajectory = MarkovRecurrentModel(mdp, policy).unroll(initial_state, timesteps)

    # PolicyOptimizer
    optimizer = PolicyOptimizer(mdp.graph, trajectory, learning_rate, discount)

    return optimizer.minimize(epochs)
