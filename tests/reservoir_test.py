import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import shutil
from src.Reservoir import Reservoir_non_linear  # noqa

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
                                               105.0]}
        cls.reservoir =  Reservoir_non_linear(graph, cls.reserv_dict)

    def test_Reservoir_reward(self):
        """
        Testing the reward of the MDP.
        In this case the state is the middle
        point of all the reservoirs. And so,
        the reward should be 0 (no costs).
        """
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
