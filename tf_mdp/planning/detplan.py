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

from policy.deterministic import DeterministicPolicyNetwork
from train.action_optimizer import ActionOptimizer

import numpy as np
import tensorflow as tf


def run(mdp, config, timesteps, batch_size, discount, epochs, learning_rate, **kwargs):

    # action variables
    with mdp.graph.as_default():
        plan = tf.Variable(
            tf.truncated_normal(shape=[batch_size, timesteps, mdp.state_size], stddev=0.05),
            name="plan")

    # ActionOptimizer
    start = config["initial"]
    limits = config.get("limits", None)
    optimizer = ActionOptimizer(mdp, start, plan, learning_rate, limits)

    return optimizer.minimize(epochs)
