#!/usr/bin/env python

# suite of unit tests for particle_environments/mager/observation.py. 
# To run test, simply call:
#
#   in a shell with conda environment ergo_particle_gym activated:
#   nosetests test_particle_environments_mager_observation.py
#
#   in ipython:
#   run test_particle_environments_mager_observation.py

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import unittest
import numpy as np
import particle_environments.mager.observation as ol
import particle_environments.mager.world as wl
from particle_environments.common import RadialPolynomialRewardFunction2D as RadialReward
from particle_environments.common import RadialBernoulliRiskFunction2D as RadialRisk
from particle_environments.common import (
    RewardFunction, RiskFunction, check_2way_communicability)

class TestObservationFunctions(unittest.TestCase):
    ''' test functions in observation.py
    '''

    def test_format_observation_1(self):
        ''' test observation formatting basic functionality
        '''

        # trial 1
        ag = [0,0]
        a1 = [1,1]
        a2 = [-1.0, 1.0]
        observe = lambda a: [a[0]-ag[0], a[1]-ag[1]]
        objects = [a1, a2]
        num_observations = 2
        observation_size = 2
        observation = ol.format_observation(observe=observe, 
                                            objects=objects,    
                                            num_observations=num_observations, 
                                            observation_size=observation_size)
        self.assertEqual(len(observation), 4)
        self.assertAlmostEqual(observation[0], 1.0)
        self.assertAlmostEqual(observation[1], 1.0)
        self.assertAlmostEqual(observation[2], -1.0)
        self.assertAlmostEqual(observation[3], 1.0)

    def test_format_observation_2(self):
        ''' test observation formatting for padding
        '''

        # trial 1
        ag = [0,0]
        a1 = [1,1]
        a2 = [-1.0, 1.0]
        observe = lambda a: [a[0]-ag[0], a[1]-ag[1]]
        objects = [a1, a2]
        num_observations = 5
        observation_size = 2
        observation = ol.format_observation(observe=observe, 
                                            objects=objects,    
                                            num_observations=num_observations, 
                                            observation_size=observation_size)
        self.assertEqual(len(observation), 10)
        self.assertAlmostEqual(observation[0], 1.0)
        self.assertAlmostEqual(observation[1], 1.0)
        self.assertAlmostEqual(observation[2], -1.0)
        self.assertAlmostEqual(observation[3], 1.0)
        for i in range(4, len(observation)):
            self.assertAlmostEqual(observation[i], 0.0)




class TestSensingLimitedMortalAgent1(unittest.TestCase):
    def setUp(self):
        self.ag1 = wl.SensingLimitedMortalAgent(5, 1)
        self.ag2 = wl.SensingLimitedMortalAgent(5.0, 1.0)
        self.ag3 = wl.SensingLimitedMortalAgent(5, np.sqrt(2))
        self.ag4 = wl.SensingLimitedMortalAgent(5.0, np.sqrt(2))
        self.ag5 = wl.SensingLimitedMortalAgent(4, 6)
        self.ag6 = wl.SensingLimitedMortalAgent(7.7311309 , 5.08263049)

    def test_sensing_limited_agents_0(self):
        self.ag1.state.p_pos = np.array([-3.76250553, -2.23691489])
        self.assertTrue(self.ag1.is_entity_observable(self.ag1))
        self.assertTrue(self.ag1.is_entity_transmittable(self.ag1))
        # self.assertTrue(self.ag1.is_entity_communicable(self.ag1))
        self.assertTrue(check_2way_communicability(self.ag1, self.ag1))

    def test_sensing_limited_agents_1(self):
        self.ag1.state.p_pos = np.array([0.0, 0.0])
        self.ag2.state.p_pos = np.array([0.0, 0.0])
        self.assertTrue(self.ag1.is_entity_observable(self.ag2))
        self.assertTrue(self.ag2.is_entity_observable(self.ag1))
        self.assertTrue(self.ag1.is_entity_transmittable(self.ag2))
        self.assertTrue(self.ag2.is_entity_transmittable(self.ag1))
        # self.assertTrue(self.ag1.is_entity_communicable(self.ag2))
        # self.assertTrue(self.ag2.is_entity_communicable(self.ag1))
        self.assertTrue(check_2way_communicability(self.ag1, self.ag2))
        self.assertTrue(check_2way_communicability(self.ag2, self.ag1))

        self.ag3.state.p_pos = np.array([0.17336086, 0.86521205])
        self.ag4.state.p_pos = np.array([0.85691831, 0.09510027])
        self.assertTrue(self.ag3.is_entity_observable(self.ag4))
        self.assertTrue(self.ag4.is_entity_observable(self.ag3))
        self.assertTrue(self.ag3.is_entity_transmittable(self.ag4))
        self.assertTrue(self.ag4.is_entity_transmittable(self.ag3))
        # self.assertTrue(self.ag3.is_entity_communicable(self.ag4))
        # self.assertTrue(self.ag4.is_entity_communicable(self.ag3))
        self.assertTrue(check_2way_communicability(self.ag3, self.ag4))
        self.assertTrue(check_2way_communicability(self.ag4, self.ag2))

    def test_sensing_limited_agents_2(self):
        self.ag1.state.p_pos = np.array([-1.0, 0.0])
        self.ag2.state.p_pos = np.array([1.0, 0.0])
        self.assertTrue(self.ag1.is_entity_observable(self.ag2))
        self.assertTrue(self.ag2.is_entity_observable(self.ag1))
        self.assertFalse(self.ag1.is_entity_transmittable(self.ag2))
        self.assertFalse(self.ag2.is_entity_transmittable(self.ag1))
        # self.assertFalse(self.ag1.is_entity_communicable(self.ag2))
        # self.assertFalse(self.ag2.is_entity_communicable(self.ag1))
        self.assertFalse(check_2way_communicability(self.ag1, self.ag2))
        self.assertFalse(check_2way_communicability(self.ag2, self.ag1))


    def test_sensing_limited_agents_3(self):
        self.ag3.state.p_pos = np.array([ 0.31806357, -0.27622187])
        self.ag4.state.p_pos = np.array([-0.48336693,  0.33957211])
        self.ag3.silent = False
        self.ag3.blind = True
        self.ag3.deaf = False
        self.ag4.silent = True
        self.ag4.blind = False
        self.ag4.deaf = True
        self.assertFalse(self.ag3.is_entity_observable(self.ag4))
        self.assertTrue(self.ag4.is_entity_observable(self.ag3))
        self.assertFalse(self.ag3.is_entity_transmittable(self.ag4))
        self.assertFalse(self.ag4.is_entity_transmittable(self.ag3))
        # self.assertFalse(self.ag3.is_entity_communicable(self.ag4))
        # self.assertFalse(self.ag4.is_entity_communicable(self.ag3))
        self.assertFalse(check_2way_communicability(self.ag3, self.ag4))
        self.assertFalse(check_2way_communicability(self.ag4, self.ag3))

        self.ag3.silent = False
        self.ag3.blind = False
        self.ag3.deaf = True
        self.ag4.silent = False
        self.ag4.blind = False
        self.ag4.deaf = False
        self.assertTrue(self.ag3.is_entity_observable(self.ag4))
        self.assertTrue(self.ag4.is_entity_observable(self.ag3))
        self.assertTrue(self.ag3.is_entity_transmittable(self.ag4))
        self.assertFalse(self.ag4.is_entity_transmittable(self.ag3))
        # self.assertFalse(self.ag3.is_entity_communicable(self.ag4))
        # self.assertFalse(self.ag4.is_entity_communicable(self.ag3))
        self.assertFalse(check_2way_communicability(self.ag3, self.ag4))
        self.assertFalse(check_2way_communicability(self.ag4, self.ag3))

class TestSensingLimitedMortalAgentWithLandmarks1(unittest.TestCase):
    def setUp(self):
        self.ag1 = wl.SensingLimitedMortalAgent(18.531416849813848, 8.355584098967974)
        self.ag2 = wl.SensingLimitedMortalAgent(9.696857344937083, 2.3796399171621907)
        self.ag3 = wl.SensingLimitedMortalAgent(4.929761044490957, 2.343242357491906)
        self.ag4 = wl.SensingLimitedMortalAgent(1.852236204402471, 9.072028221658348)
        self.ag5 = wl.SensingLimitedMortalAgent(17.518788609209874, 8.054266892577962)
        self.ag6 = wl.SensingLimitedMortalAgent(15.754973134000608, 0.49169297509907506)
        self.lnd1 = wl.RiskRewardLandmark(RiskFunction(), RewardFunction())

    def test_sensing_limited_agents_landmarks_1(self):
        self.ag1.state.p_pos = np.array([-0.40324806, -0.77577631])
        self.ag2.state.p_pos = np.array([-3.65523633,  1.6878564 ])
        self.ag3.state.p_pos = np.array([-5.92980527,  1.8940491 ])
        self.ag4.state.p_pos = np.array([ 2.72110377, -4.32510047])
        self.ag5.state.p_pos = np.array([8.59344534, 8.94993718])
        self.ag6.state.p_pos = np.array([ 7.2732259 , -9.33902694])
        self.lnd1.state.p_pos = np.array([3.42048654, 5.90631669])
        lnd1_dists = (  7.698786472677766, 
                        8.237794689803087, 
                        10.174784918902095, 
                        10.255292943690849, 
                        6.001927060007084, 
                        15.724633635569218)
        self.assertTrue(self.ag1.is_entity_observable(self.lnd1))
        self.assertTrue(self.ag2.is_entity_observable(self.lnd1))
        self.assertFalse(self.ag3.is_entity_observable(self.lnd1))
        self.assertFalse(self.ag4.is_entity_observable(self.lnd1))
        self.assertTrue(self.ag5.is_entity_observable(self.lnd1))
        self.assertTrue(self.ag6.is_entity_observable(self.lnd1))

        self.assertTrue(self.ag1.is_entity_transmittable(self.lnd1))
        self.assertFalse(self.ag2.is_entity_transmittable(self.lnd1))
        self.assertFalse(self.ag3.is_entity_transmittable(self.lnd1))
        self.assertFalse(self.ag4.is_entity_transmittable(self.lnd1))
        self.assertTrue(self.ag5.is_entity_transmittable(self.lnd1))
        self.assertFalse(self.ag6.is_entity_transmittable(self.lnd1))

        # self.assertTrue(self.ag1.is_entity_communicable(self.lnd1))
        # self.assertFalse(self.ag2.is_entity_communicable(self.lnd1))
        # self.assertFalse(self.ag3.is_entity_communicable(self.lnd1))
        # self.assertFalse(self.ag4.is_entity_communicable(self.lnd1))
        # self.assertTrue(self.ag5.is_entity_communicable(self.lnd1))
        # self.assertFalse(self.ag6.is_entity_communicable(self.lnd1))

        self.assertTrue(check_2way_communicability(self.ag1, self.lnd1))
        self.assertFalse(check_2way_communicability(self.ag2, self.lnd1))
        self.assertFalse(check_2way_communicability(self.ag3, self.lnd1))
        self.assertFalse(check_2way_communicability(self.ag4, self.lnd1))
        self.assertTrue(check_2way_communicability(self.ag5, self.lnd1))
        self.assertFalse(check_2way_communicability(self.ag6, self.lnd1))

        



if __name__ == '__main__':
    unittest.main()
