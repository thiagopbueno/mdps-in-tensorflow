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
from train.optimizer import SGDPolicyOptimizer
from evaluation.montecarlo import MCPolicyEvaluation

import tensorflow as tf

def run(mdp, start, timesteps, batch_size, discount, epochs, learning_rate):

    # PolicyNetwork
    shape = [mdp.state_size + 1, 20, 5, mdp.action_size]
    policy = DeterministicPolicyNetwork(mdp.graph, shape)

    # PolicyEvaluation
    mc = MCPolicyEvaluation(mdp, policy,
                                initial_state=start,
                                max_time=timesteps,
                                batch_size=batch_size,
                                gamma=discount)

    # PolicyOptimizer
    metrics = {
        "loss":  mc.expected_return,
        "total": mc.total
    }
    optimizer = SGDPolicyOptimizer(mdp.graph, metrics, learning_rate)
    return optimizer.minimize(epochs)