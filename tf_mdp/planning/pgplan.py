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
from train.pg_optimizer import PolicyGradientOptimizer
from . import detplan

import numpy as np
import tensorflow as tf


def run(mdp, config, max_time, batch_size, discount, epochs, learning_rate, **kwargs):
    baseline = None

    if kwargs["baseline"]:
        print(">> Computing most-likely baseline ...")
        result = detplan.run(mdp, mdp.config,
                    max_time, batch_size,
                    discount, epochs, learning_rate)
        rewards = result["solution"]["rewards"]
        total = result["solution"]["total"]
        baseline = total
        # cumsum = np.cumsum(rewards[::-1])[::-1].tolist()
        # baseline = cumsum[1:]+ [0.0]
        # for r, c, b  in zip(rewards, cumsum, baseline):
        #     print("{:10.4f} | {:10.4f} | {:10.4f}".format(r, c, b))

        tf.reset_default_graph()

    # PolicyNetwork
    shape = [mdp.state_size + 1, 20, 5, mdp.action_size]
    policy = DeterministicPolicyNetwork(mdp.graph, shape)

    # MarkovRecurrentModel
    start = config["initial"]
    initial_state = utils.initial_state(start, batch_size)
    timesteps = utils.timesteps(batch_size, max_time)
    trajectory = MarkovRecurrentModel(mdp, policy).unroll(initial_state, timesteps)

    # PolicyOptimizer
    optimizer = PolicyGradientOptimizer(mdp, policy, trajectory, learning_rate, discount, baseline)

    return optimizer.minimize(epochs)
