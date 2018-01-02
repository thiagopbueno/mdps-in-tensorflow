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


import unittest
import numpy as np
import tensorflow as tf
import os
import shutil
from tf_mdp.models import mdp # noqa
from tf_mdp.models.reservoir import reservoir # noqa


class ReservoirTest(unittest.TestCase):
    """
    Testing the Class Reservoir.
    """
    @classmethod
    def setUpClass(cls):
        graph = tf.Graph()
        cls.reserv_dict = {'n_reservoirs': 3,
                            'upper_bounds': [200.0,
                                             200.0,
                                             200.0],
                            'lower_bounds': [10.0,
                                             10.0,
                                             10.0],
                            'initial_states': [105.0,
                                               105.0,
                                               105.0],
                            'rain_mean': 5.0,
                            'rain_std': 1.,
                            'evaporation_std': 5.}

        cls.reservoir =  reservoir.Reservoir_non_linear(graph, cls.reserv_dict)

    def test_reward_is_zero_when_state_is_middle_point(self):
        batch = np.array([ReservoirTest.reserv_dict['initial_states'],
                          ReservoirTest.reserv_dict['initial_states']])
        with ReservoirTest.reservoir.graph.as_default():
            states = tf.constant(batch, dtype="float32")
            rewards = ReservoirTest.reservoir.reward(states, None)
        sess = tf.Session(graph=ReservoirTest.reservoir.graph)
        result = np.mean(sess.run(rewards))
        self.assertAlmostEqual(result,
                               0.00,
                               places=2,
                               msg="result = {}".format(result))
