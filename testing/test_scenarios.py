#!/usr/bin/env python

# suite of unit, integration, system, and/or acceptance tests for particle_environments/mager/scenarios. 
# To run test, simply call:
#
#   in a shell with conda environment ergo_particle_gym activated:
#   nosetests test_scenarios.py
#
#   in ipython:
#   run test_scenarios.py

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'particle_environments/multiagent_particle_envs'))

import unittest
import numpy as np
import tensorflow as tf
from random import shuffle
from copy import deepcopy
from train import DeepMLP
from shapely.geometry import Point
from particle_environments.mager.environment import MultiAgentRiskEnv
import particle_environments.mager.scenarios.ergo_circuit as ergo_circuit
import particle_environments.mager.scenarios.ergo_circuit_simplified as ergo_circuit_simplified
import particle_environments.mager.scenarios.ergo_hazards as ergo_hazards
import particle_environments.mager.scenarios.ergo_graph_large as ergo_graph_large
import particle_environments.mager.scenarios.ergo_spread_small as ergo_spread_small
import particle_environments.mager.scenarios.ergo_spread_variable as ergo_spread_variable
import particle_environments.mager.scenarios.ergo_graph_variable as ergo_graph_variable
import particle_environments.mager.scenarios.simple_graph_large as simple_graph_large
import particle_environments.mager.scenarios.ergo_perimeter_small as ergo_perimeter_small
import particle_environments.mager.scenarios.ergo_perimeter_variable as ergo_perimeter_variable
import particle_environments.mager.scenarios.ergo_perimeter2_variable as ergo_perimeter2_variable
from particle_environments.mager.world import TemporarilyObservableRiskRewardLandmark as TORRLandmark
from collections import namedtuple
from particle_environments.mager.world import MortalAgent
from particle_environments.common import RadialPolynomialRewardFunction2D as RadialReward
from particle_environments.common import RadialBernoulliRiskFunction2D as RadialRisk
from particle_environments.common import is_collision, MVP2D
from rl_algorithms.mappo import PPOAgentComputer, PPOGroupTrainer


class TestErgoCircuitCase1(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_circuit.Scenario()
        self.scenario.num_agents = 2

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.landmarks = self.world.landmarks[0:2]
        self.world.origin_terminal_landmark.state.p_pos = np.array([0.0, 0.0])
        self.world.destination_terminal_landmark.state.p_pos = np.array([3.0, 0.0])
        self.world.agents[0].state.p_pos = np.array([1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([2.0, 0.0])
        self.world.max_communication_distance = 1.1
        self.world.distance_resistance_gain = 1.0
        self.world.distance_resistance_exponent = 1.0


    def test_setup(self):
        ''' check setup created objects as expected'''
        self.assertEqual(self.scenario.num_agents, 2)
        self.assertEqual(len(self.world.landmarks), 2)
        self.assertEqual(self.world.landmarks[0], self.world.origin_terminal_landmark)
        self.assertEqual(self.world.landmarks[1], self.world.destination_terminal_landmark)

    def test_non_systemic_reward(self):
        '''check that individual rewards are zero'''
        for agent in self.world.agents:
            reward = self.scenario.reward(agent=agent, world=self.world)
            self.assertAlmostEqual(reward, 0.0)

    def test_systemic_reward_case_1(self):
        '''check systemic reward matches conductance of series circuit'''

        # resistance of a series circuit
        expected_resistance = 3.0
        expected_reward = 1.0/expected_resistance
        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, expected_reward)

    def test_systemic_reward_case_2(self):
        '''check that terminated agents do not affect communication'''
        self.world.agents.append(MortalAgent())
        self.world.agents[2].state.p_pos = np.array([1.5,0.0])
        self.world.agents[2].terminated = True

        # resistance of a series circuit
        expected_resistance = 3.0
        expected_reward = 1.0/expected_resistance
        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, expected_reward)

    def test_systemic_reward_case_3(self):
        '''check that there is no direct communication between terminals'''
        self.world.agents = []
        self.world.destination_terminal_landmark.state.p_pos = np.array([1.0, 0.0])

        # resistance of a series circuit
        expected_reward = 0.0
        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, expected_reward)

    def test_systemic_reward_case_4(self):
        '''check systemic reward matches conductance of parallel circuit'''

        # arrange agents such that they are 1 m from each terminal but 1.2 m from each other
        # i.e. no inter-agent comm
        self.world.max_communication_distance = 1.1
        self.world.agents[0].state.p_pos = np.array([np.sqrt(0.64),0.6])
        self.world.agents[1].state.p_pos = np.array([np.sqrt(0.64),-0.6])
        self.world.destination_terminal_landmark.state.p_pos = np.array([2.0*np.sqrt(0.64), 0.0])

        # resistance of a parallel circuit
        expected_resistance = 1.0/(1.0/2.0 + 1.0/2.0)
        expected_reward = 1.0/expected_resistance
        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, expected_reward)

    def test_systemic_reward_case_5(self):
        '''check systemic reward matches conductance of bridge circuit'''

        # arrange agents such that they are 1 m from each terminal and each other
        # forms "diamond bridge" circuit with unit resistance
        self.world.max_communication_distance = 1.1
        self.world.agents[0].state.p_pos = np.array([np.sqrt(0.75),0.5])
        self.world.agents[1].state.p_pos = np.array([np.sqrt(0.75),-0.5])
        self.world.destination_terminal_landmark.state.p_pos = np.array([2.0*np.sqrt(0.75), 0.0])

        # resistance of a diamond bridge circuit
        expected_resistance = 1.0
        expected_reward = 1.0/expected_resistance
        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, expected_reward)

class TestErgoCircuitCase2(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_circuit.Scenario()
        self.scenario.num_agents = 5

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.max_communication_distance = 0.51
        self.world.origin_terminal_landmark.state.p_pos = np.array([-1.0, 0.0])
        self.world.destination_terminal_landmark.state.p_pos = np.array([1.0, 0.0])

    def test_reset_world_spawn_terminals(self):
        '''check terminals are spawned in expected areas upon world reset
        '''

        num_trials = 1000
        dists = np.zeros(num_trials)
        for i in range(num_trials):
            self.scenario.reset_world(self.world)
            self.assertEqual(self.world.landmarks[0], self.world.origin_terminal_landmark)
            self.assertEqual(self.world.landmarks[1], self.world.destination_terminal_landmark)
            dists[i] = np.linalg.norm(self.world.landmarks[0].state.p_pos - self.world.landmarks[1].state.p_pos)

        # count instances where distance between terminals is within expected 3 sigma
        counts = ((dists > 1.7) & (dists < 2.3)).sum()
        self.assertGreater(float(counts)/float(num_trials), 0.99)

    def test_systemic_reward_case_6(self):
        '''check systemic reward for 5 agents, two uninvolved'''

        # arrange agents in a line with 2 agents uninvolved
        self.world.max_communication_distance = 0.51
        self.world.agents[0].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[1].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[3].state.p_pos = np.array([-10.0, 0.0])
        self.world.agents[4].state.p_pos = np.array([10.0, 0.0])

        # resistance of a diamond bridge circuit
        expected_resistance = 2.0
        expected_reward = 1.0/expected_resistance
        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, expected_reward)

    def test_systemic_reward_case_7(self):
        '''check systemic reward for 5 agents is always greater than straigh line'''

        # arrange agents in a line with 2 agents randomly placed
        # self.scenario.num_agents = 25
        # self.world = self.scenario.make_world()
        num_trials = 1000
        for i in range(num_trials):
            self.scenario.reset_world(self.world)
            self.world.max_communication_distance = 0.51
            self.world.origin_terminal_landmark.state.p_pos = np.array([-1.0, 0.0])
            self.world.destination_terminal_landmark.state.p_pos = np.array([1.0, 0.0])
            self.world.agents[0].state.p_pos = np.array([-0.5, 0.0])
            self.world.agents[1].state.p_pos = np.array([0.0, 0.0])
            self.world.agents[2].state.p_pos = np.array([0.5, 0.0])

            max_expected_resistance = 2.0
            min_expected_reward = 1.0/max_expected_resistance - np.finfo(np.float32).eps
            reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
            for r in reward:
                self.assertGreaterEqual(r, min_expected_reward)


class TestErgoCircuitSimplifiedCase2(unittest.TestCase):
    ''' test get_trainer_actions from train.py
    '''

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_circuit_simplified.Scenario()
        self.scenario.num_agents = 3

        # create world and modify to test conditions
        l = 0.75
        self.r = 0.7
        self.world = self.scenario.make_world()
        self.world.agents[0].state.p_pos = np.array([-l/2.0, -l/2.0])
        self.world.agents[1].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[2].state.p_pos = np.array([l/2.0, l/2.0])
        self.world.distance_resistance_gain = 1.0
        self.l = l

    def test_setup(self):
        ''' check setup created objects as expected'''
        self.assertEqual(self.scenario.num_agents, 3)
        self.assertEqual(len(self.world.landmarks), 2)
        self.assertEqual(self.world.landmarks[0], self.world.origin_terminal_landmark)
        self.assertEqual(self.world.landmarks[1], self.world.destination_terminal_landmark)
        self.assertAlmostEqual(self.world.landmarks[0].state.p_pos[0], -self.l)
        self.assertAlmostEqual(self.world.landmarks[0].state.p_pos[1], -self.l)
        self.assertAlmostEqual(self.world.landmarks[1].state.p_pos[0], self.l)
        self.assertAlmostEqual(self.world.landmarks[1].state.p_pos[1], self.l)
        self.assertAlmostEqual(self.world.max_communication_distance, self.r)

    def test_even_space_systemic_reward(self):
        '''check the systemic reward for evenly spaced agents'''
        L = 2*np.sqrt(2*self.l**2)
        expected_reward = 1.0/L
        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, expected_reward)

class TestErgoGraphLargeCase1(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_graph_large.Scenario()
        self.scenario.num_agents = 5

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.origin_terminal_landmark.state.p_pos = np.array([-1.0, 0.0])
        self.world.destination_terminal_landmark.state.p_pos = np.array([1.0, 0.0])
        for a in self.world.agents:
            a.observation_range = 1.0
            a.transmit_range = 1.01

    def test_ergo_graph_large_case1_systemic_reward_1(self):
        ''' ergo_graph_large: systemic reward for evenly spaced agents
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, self.world.connection_reward)

    def test_ergo_graph_large_case1_systemic_reward_2(self):
        ''' ergo_graph_large: systemic reward for disconnected agents
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 10.0])
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for i, r in enumerate(reward):
            if i != 2:
                self.assertAlmostEqual(r, 1*self.world.connection_reward)
            else:
                self.assertAlmostEqual(r, 0)

    def test_ergo_graph_large_case1_systemic_reward_3(self):
        ''' ergo_graph_large: systemic reward with terminated agents
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[2].terminated = True
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for i, r in enumerate(reward):
            if i != 2:
                self.assertAlmostEqual(r, 1*self.world.connection_reward)
            else:
                self.assertAlmostEqual(r, self.world.termination_reward)


    def test_ergo_graph_large_case1_observations_1(self):
        ''' ergo_graph_large: test observations with 5 agents
        Notes:
         - Assume 4 radial and 8 angular bins with a max obs distance of 1.0
         - Formatted to work for either linear or logspaced bins
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.1, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[2].state.p_vel = np.array([0.0, 0.0])
        self.world.agents[3].state.p_pos = np.array([0.1, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        expected_histogram = np.array([[0]*8]*4)
        expected_histogram[3][4] = 1
        expected_histogram[0][4] = 1
        expected_histogram[0][0] = 1
        expected_histogram[3][0] = 1
        expected_histogram = [val for sublist in expected_histogram for val in sublist]
        expected_observation = ([0,0,0,0] + # vel and pos
                                [-1,0, 1,0] + # terminal locations
                                expected_histogram + # agent histogram
                                [0.0]*5*2 + # observed failures
                                [0]*50 # previous observation
                                )

        observation = self.scenario.observation(agent=self.world.agents[2], world=self.world)
        for i, obs in enumerate(observation):
            self.assertAlmostEqual(obs, expected_observation[i])


class TestErgoGraphLargeCase2(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_graph_large.Scenario()
        self.scenario.num_agents = 20

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.origin_terminal_landmark.state.p_pos = np.array([-1.0, 0.0])
        self.world.destination_terminal_landmark.state.p_pos = np.array([1.0, 0.0])
        for a in self.world.agents:
            a.observation_range = 1.0
            a.transmit_range = 0.51

    def test_ergo_graph_large_case2_observations_1(self):
        ''' ergo_graph_large: test observations with 20 agents
        Notes:
         - Assume 4 radial and 8 angular bins with a max obs distance of 1.0
         - Formatted to work for either logspaced radial bins with partitions:
            [0.1       , 0.21544347, 0.46415888, 1.        ]
         - Assumes terminated agents are always observable
        '''

        rt2_2 = np.sqrt(2.0)/2.0 - np.finfo(float).eps
        self.world.agents[0].state.p_pos = np.array([0.0, 0.0])

        # immediate radial bins
        self.world.agents[1].state.p_pos = np.array([0.1, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.1*rt2_2, 0.1*rt2_2])
        self.world.agents[3].state.p_pos = np.array([0.0, 0.1])
        self.world.agents[4].state.p_pos = np.array([-0.1*rt2_2, 0.1*rt2_2])
        self.world.agents[5].state.p_pos = np.array([-0.1, 0.0])
        self.world.agents[6].state.p_pos = np.array([-0.1*rt2_2, -0.1*rt2_2])
        self.world.agents[7].state.p_pos = np.array([0.0, -0.1])
        self.world.agents[8].state.p_pos = np.array([0.1*rt2_2, -0.1*rt2_2])

        # 5 agents in radial bin 1 (0.1-0.21544347], angular bin 5 (9*pi/8-11*pi/8]
        for i in range(9,14):
            dist = np.random.uniform(0.1, 0.21544347)
            ang = np.random.uniform(9.0*np.pi/8.0, 11.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.agents[i].state.p_pos = np.array([x,y])

        # 3 agents in radial bin 3, angular bin 7
        for i in range(14,17):
            dist = np.random.uniform(0.46415888, 1.)
            ang = np.random.uniform(13.0*np.pi/8.0, 15.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.agents[i].state.p_pos = np.array([x,y])

        # 3 remaining are terminated at random locations
        failure_locations = []
        failure_distances = []
        for i in range(17,20):
            failure_locations.append(np.random.rand(2)*2-1)
            failure_distances.append(np.linalg.norm(failure_locations[-1]))
            self.world.agents[i].state.p_pos = failure_locations[-1]
            self.world.agents[i].terminated = True


        expected_histogram = np.array([[0]*8]*4)
        expected_histogram[0][0] = 1
        expected_histogram[0][1] = 1
        expected_histogram[0][2] = 1
        expected_histogram[0][3] = 1
        expected_histogram[0][4] = 1
        expected_histogram[0][5] = 1
        expected_histogram[0][6] = 1
        expected_histogram[0][7] = 1
        expected_histogram[1][5] = 5
        expected_histogram[3][7] = 3
        expected_histogram = [val for sublist in expected_histogram for val in sublist]

        expected_failures = [x for _,x in sorted(zip(failure_distances,failure_locations))]
        expected_failures = [val for sublist in expected_failures for val in sublist]
        expected_failures.extend([0.0, 0.0, 0.0, 0.0])

        expected_observation = ([0,0,0,0] + # vel and pos
                                [-1,0, 1,0] + # terminal locations
                                expected_histogram + # agent histogram
                                expected_failures + # observed failures
                                [0]*50 # previous observation
                                )

        observation = self.scenario.observation(agent=self.world.agents[0], world=self.world)
        for i, obs in enumerate(observation):
            self.assertAlmostEqual(obs, expected_observation[i])

class TestSimpleGraphLargeCase1(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = simple_graph_large.Scenario()
        self.scenario.num_agents = 5

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.origin_terminal_landmark.state.p_pos = np.array([-1.0, 0.0])
        self.world.destination_terminal_landmark.state.p_pos = np.array([1.0, 0.0])
        for a in self.world.agents:
            a.observation_range = 1.0
            a.transmit_range = 1.01

    def test_simple_graph_large_case1_systemic_reward_1(self):
        ''' simple_graph_large: systemic reward for evenly spaced agents
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, self.world.connection_reward)

    def test_simple_graph_large_case1_systemic_reward_2(self):
        ''' simple_graph_large: systemic reward with one agent disconnected
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 10.0])
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, self.world.connection_reward)

    def test_simple_graph_large_case1_systemic_reward_3(self):
        ''' simple_graph_large: systemic reward with no connection
        '''

        self.world.agents[0].state.p_pos = np.array([-10.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 10.0])
        self.world.agents[3].state.p_pos = np.array([5, 0.0])
        self.world.agents[4].state.p_pos = np.array([10.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for i, r in enumerate(reward):
            self.assertAlmostEqual(r, 0)

    def test_simple_graph_large_case1_systemic_reward_4(self):
        ''' simple_graph_large: systemic reward with all agents connected, but not to terminals
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 10.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 10.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 10.0])
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for i, r in enumerate(reward):
            self.assertAlmostEqual(r, 0)

    def test_simple_graph_large_case1_observations_1(self):
        ''' simple_graph_large: test observations with 5 agents
        Notes:
         - Assume 4 radial and 8 angular bins with a max obs distance of 1.0
         - Formatted to work for either linear or logspaced bins
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.1, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[2].state.p_vel = np.array([0.0, 0.0])
        self.world.agents[3].state.p_pos = np.array([0.1, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        expected_histogram = np.array([[0]*8]*4)
        expected_histogram[3][4] = 1
        expected_histogram[0][4] = 1
        expected_histogram[0][0] = 1
        expected_histogram[3][0] = 1
        expected_histogram = [val for sublist in expected_histogram for val in sublist]
        expected_observation = ([0] + # termination
                                [0,0,0,0] + # vel and pos
                                [-1,0, 1,0] + # terminal locations
                                expected_histogram + # agent histogram
                                [0.0]*5*2 + # observed failures
                                [0]*51 # previous observation
                                )

        observation = self.scenario.observation(agent=self.world.agents[2], world=self.world)
        for i, obs in enumerate(observation):
            self.assertAlmostEqual(obs, expected_observation[i])


class TestErgoGraphVariableCase1(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_graph_variable.Scenario(num_agents=5, num_hazards=1, identical_rewards=False, observation_type="histogram")

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.origin_terminal_landmark.state.p_pos = np.array([-1.0, 0.0])
        self.world.destination_terminal_landmark.state.p_pos = np.array([1.0, 0.0])
        for a in self.world.agents:
            a.observation_range = 1.0
            a.transmit_range = 1.01

    def test_ergo_graph_variable_case1_systemic_reward_1(self):
        ''' ergo_graph_variable: systemic reward for 5 evenly spaced agents (1 hazard, local rewards, histogram observation)
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, self.world.connection_reward)

    def test_ergo_graph_variable_case1_systemic_reward_2(self):
        ''' ergo_graph_variable: systemic reward for 5 disconnected agents (1 hazard, local rewards, histogram observation)
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 10.0])
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for i, r in enumerate(reward):
            if i != 2:
                self.assertAlmostEqual(r, 1*self.world.connection_reward)
            else:
                self.assertAlmostEqual(r, 0)

    def test_ergo_graph_variable_case1_systemic_reward_3(self):
        ''' ergo_graph_variable: systemic reward with terminated agents (1 hazard, local rewards, histogram observation)
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[2].terminated = True
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for i, r in enumerate(reward):
            if i != 2:
                self.assertAlmostEqual(r, 1*self.world.connection_reward)
            else:
                self.assertAlmostEqual(r, self.world.termination_reward)


    def test_ergo_graph_variable_case1_observations_1(self):
        ''' ergo_graph_variable: test histogram observations with 5 agents (1 hazard, local rewards, histogram observation)
        Notes:
         - Assume 4 radial and 8 angular bins with a max obs distance of 1.0
         - Formatted to work for either linear or logspaced bins
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.1, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[2].state.p_vel = np.array([0.0, 0.0])
        self.world.agents[3].state.p_pos = np.array([0.1, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        expected_histogram = np.array([[0]*8]*4)
        expected_histogram[3][4] = 1
        expected_histogram[0][4] = 1
        expected_histogram[0][0] = 1
        expected_histogram[3][0] = 1
        expected_histogram = [val for sublist in expected_histogram for val in sublist]
        expected_observation = ([0] + # agent terminated
                                [0,0,0,0] + # vel and pos
                                [-1,0, 1,0] + # terminal locations
                                expected_histogram + # agent histogram
                                [0.0]*5*2 + # observed failures
                                [0]*51 # previous observation
                                )

        observation = self.scenario.observation(agent=self.world.agents[2], world=self.world)
        for i, obs in enumerate(observation):
            self.assertAlmostEqual(obs, expected_observation[i])


class TestErgoGraphVariableCase2(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        # self.scenario = ergo_graph_large.Scenario()
        # self.scenario.num_agents = 20
        self.scenario = ergo_graph_variable.Scenario(num_agents=20, num_hazards=1, identical_rewards=False, observation_type="histogram")

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.origin_terminal_landmark.state.p_pos = np.array([-1.0, 0.0])
        self.world.destination_terminal_landmark.state.p_pos = np.array([1.0, 0.0])
        for a in self.world.agents:
            a.observation_range = 1.0
            a.transmit_range = 0.51

    def test_ergo_graph_variable_case2_observations_1(self):
        ''' ergo_graph_variable: test observations with 20 agents (1 hazard, local rewards, histogram observation)
        Notes:
         - Assume 4 radial and 8 angular bins with a max obs distance of 1.0
         - Formatted to work for either logspaced radial bins with partitions:
            [0.1       , 0.21544347, 0.46415888, 1.        ]
         - Assumes terminated agents are always observable
        '''

        rt2_2 = np.sqrt(2.0)/2.0 - np.finfo(float).eps
        self.world.agents[0].state.p_pos = np.array([0.0, 0.0])

        # immediate radial bins
        self.world.agents[1].state.p_pos = np.array([0.1, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.1*rt2_2, 0.1*rt2_2])
        self.world.agents[3].state.p_pos = np.array([0.0, 0.1])
        self.world.agents[4].state.p_pos = np.array([-0.1*rt2_2, 0.1*rt2_2])
        self.world.agents[5].state.p_pos = np.array([-0.1, 0.0])
        self.world.agents[6].state.p_pos = np.array([-0.1*rt2_2, -0.1*rt2_2])
        self.world.agents[7].state.p_pos = np.array([0.0, -0.1])
        self.world.agents[8].state.p_pos = np.array([0.1*rt2_2, -0.1*rt2_2])

        # 5 agents in radial bin 1 (0.1-0.21544347], angular bin 5 (9*pi/8-11*pi/8]
        for i in range(9,14):
            dist = np.random.uniform(0.1, 0.21544347)
            ang = np.random.uniform(9.0*np.pi/8.0, 11.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.agents[i].state.p_pos = np.array([x,y])

        # 3 agents in radial bin 3, angular bin 7
        for i in range(14,17):
            dist = np.random.uniform(0.46415888, 1.)
            ang = np.random.uniform(13.0*np.pi/8.0, 15.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.agents[i].state.p_pos = np.array([x,y])

        # 3 remaining are terminated at random locations
        failure_locations = []
        failure_distances = []
        for i in range(17,20):
            failure_locations.append(np.random.rand(2)*2-1)
            failure_distances.append(np.linalg.norm(failure_locations[-1]))
            self.world.agents[i].state.p_pos = failure_locations[-1]
            self.world.agents[i].terminated = True


        expected_histogram = np.array([[0]*8]*4)
        expected_histogram[0][0] = 1
        expected_histogram[0][1] = 1
        expected_histogram[0][2] = 1
        expected_histogram[0][3] = 1
        expected_histogram[0][4] = 1
        expected_histogram[0][5] = 1
        expected_histogram[0][6] = 1
        expected_histogram[0][7] = 1
        expected_histogram[1][5] = 5
        expected_histogram[3][7] = 3
        expected_histogram = [val for sublist in expected_histogram for val in sublist]

        expected_failures = [x for _,x in sorted(zip(failure_distances,failure_locations))]
        expected_failures = [val for sublist in expected_failures for val in sublist]
        expected_failures.extend([0.0, 0.0, 0.0, 0.0])

        expected_observation = ([0] + # agent terminated
                                [0,0,0,0] + # vel and pos
                                [-1,0, 1,0] + # terminal locations
                                expected_histogram + # agent histogram
                                expected_failures + # observed failures
                                [0]*51 # previous observation
                                )

        observation = self.scenario.observation(agent=self.world.agents[0], world=self.world)
        for i, obs in enumerate(observation):
            self.assertAlmostEqual(obs, expected_observation[i])

class TestErgoGraphVariableCase3(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        # self.scenario = simple_graph_large.Scenario()
        # self.scenario.num_agents = 5
        self.scenario = ergo_graph_variable.Scenario(num_agents=5, num_hazards=0, identical_rewards=True, observation_type="histogram")

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.origin_terminal_landmark.state.p_pos = np.array([-1.0, 0.0])
        self.world.destination_terminal_landmark.state.p_pos = np.array([1.0, 0.0])
        for a in self.world.agents:
            a.observation_range = 1.0
            a.transmit_range = 1.01

    def test_ergo_graph_variable_case3_systemic_reward_1(self):
        ''' ergo_graph_variable: systemic reward for evenly spaced agents (no hazards, identical rewards, histogram observation)
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, self.world.connection_reward)

    def test_ergo_graph_variable_case3_systemic_reward_2(self):
        ''' ergo_graph_variable: systemic reward with one agent disconnected (no hazards, identical rewards, histogram observation)
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 10.0])
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, self.world.connection_reward)

    def test_ergo_graph_variable_case3_systemic_reward_3(self):
        ''' ergo_graph_variable: systemic reward with no connection (no hazard, identical rewards, histogram observation)
        '''

        self.world.agents[0].state.p_pos = np.array([-10.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 10.0])
        self.world.agents[3].state.p_pos = np.array([5, 0.0])
        self.world.agents[4].state.p_pos = np.array([10.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for i, r in enumerate(reward):
            self.assertAlmostEqual(r, 0)

    def test_ergo_graph_variable_case3_systemic_reward_4(self):
        ''' ergo_graph_variable: systemic reward with all agents connected, but not to terminals (no hazards, identical rewards, histogram observation)
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 10.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 10.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 10.0])
        self.world.agents[3].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for i, r in enumerate(reward):
            self.assertAlmostEqual(r, 0)

    def test_ergo_graph_variable_case3_observations_1(self):
        ''' ergo_graph_variable: test observations with 5 agents (no hazards, identical rewards, histogram observation)
        Notes:
         - Assume 4 radial and 8 angular bins with a max obs distance of 1.0
         - Formatted to work for either linear or logspaced bins
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.1, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[2].state.p_vel = np.array([0.0, 0.0])
        self.world.agents[3].state.p_pos = np.array([0.1, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        expected_histogram = np.array([[0]*8]*4)
        expected_histogram[3][4] = 1
        expected_histogram[0][4] = 1
        expected_histogram[0][0] = 1
        expected_histogram[3][0] = 1
        expected_histogram = [val for sublist in expected_histogram for val in sublist]
        expected_observation = ([0] + # termination
                                [0,0,0,0] + # vel and pos
                                [-1,0, 1,0] + # terminal locations
                                expected_histogram + # agent histogram
                                [0.0]*5*2 + # observed failures
                                [0]*51 # previous observation
                                )

        observation = self.scenario.observation(agent=self.world.agents[2], world=self.world)
        for i, obs in enumerate(observation):
            self.assertAlmostEqual(obs, expected_observation[i])

class TestErgoGraphVariableCase4(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        # self.scenario = simple_graph_large.Scenario()
        # self.scenario.num_agents = 5
        self.scenario = ergo_graph_variable.Scenario(num_agents=5, num_hazards=0, identical_rewards=True, observation_type="direct")

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.origin_terminal_landmark.state.p_pos = np.array([-1.0, 0.0])
        self.world.destination_terminal_landmark.state.p_pos = np.array([1.0, 0.0])
        for a in self.world.agents:
            a.observation_range = 1.0
            a.transmit_range = 1.01

    def test_ergo_graph_variable_case4_observations_1(self):
        ''' ergo_graph_variable: test observations with 5 agents (no hazards, identical rewards, direct observation)
        Notes:
         - Assume 4 radial and 8 angular bins with a max obs distance of 1.0
         - Formatted to work for either linear or logspaced bins
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.1, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[2].state.p_vel = np.array([0.0, 0.0])
        self.world.agents[3].state.p_pos = np.array([0.1, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        expected_observation = ([0] + # termination
                                [0.0, 0.0, 0.0, 0.0] + # pos and vel of observing agent
                                [-1.0, 0.0, 1.0, 0.0] + # pos of terminals
                                [0, -1.0, 0.0, 0.0, 0.0] + # comm from agent 0
                                [0, -0.1, 0.0, 0.0, 0.0] + # comm from agent 1
                                [0, 0.1, 0.0, 0.0, 0.0] +  # comm from agent 3
                                [0, 1.0, 0.0, 0.0, 0.0]    # comm from agent 4
                                )


        observation = self.scenario.observation(agent=self.world.agents[2], world=self.world)
        self.assertEqual(len(observation), len(expected_observation))
        for i, obs in enumerate(observation):
            self.assertAlmostEqual(obs, expected_observation[i])


class TestErgoSpreadVariableCase1(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_spread_variable.Scenario(num_agents=3, num_hazards=1, identical_rewards=True, observation_type="direct")

        # create world and modify to test conditions
        self.world = self.scenario.make_world()

    def test_ergo_spread_variable_hazard_rewards_1(self):
        ''' ergo_spread_variable: hazards provide no reward and is randomly shuffled, termination doesn't affect reward '''

        hazard_index = []
        n_trials = 100
        for i in range(n_trials):
            agent_indices = np.arange(self.scenario.num_agents)
            shuffle(agent_indices)
            ag_count = 0
            self.scenario.reset_world(self.world)
            for l_ind, l in enumerate(self.world.landmarks):
                if l.is_hazard:
                    hazard_index.append(l_ind)
                else:
                    # move an agent directly on top of 
                    self.world.agents[agent_indices[ag_count]].state.p_pos = deepcopy(l.state.p_pos)
                    ag_count += 1

            for a in self.world.agents:
                # non hazard landmarks should be covered by an agent and
                # all agents should have same reward, thus all rewards should
                # be zero
                # agents are randomly terminated, but this should have no effect
                # on reward
                a.collide = False
                a.terminated = bool(np.random.binomial(1,0.5))
                reward = self.scenario.reward(a, self.world)
                self.assertAlmostEqual(reward, 0.0)

        # check that only one hazard exits per trial
        self.assertEqual(len(hazard_index), n_trials)

        # check that hazard index is shuffled ((1/n_landmarks)^n_trials probability of false negative)
        self.assertFalse(all(np.array(hazard_index)==hazard_index[0]))

    def test_ergo_spread_variable_hazard_rewards_2(self):
        ''' ergo_spread_variable: test non-zero rewards with collocated agents and landmarks'''

        hazard_index = []
        n_trials = 100
        for i in range(n_trials):
            ag_x = np.random.uniform(-1,1)
            ag_y = np.random.uniform(-1,1)
            ln_x = np.random.uniform(-1,1)
            ln_y = np.random.uniform(-1,1)
            exp_reward = -np.sqrt((ag_x-ln_x)**2 + (ag_y-ln_y)**2)
            exp_reward *= len([l for l in self.world.landmarks if not l.is_hazard])
            self.scenario.reset_world(self.world)
            for l_ind, l in enumerate(self.world.landmarks):
                if l.is_hazard:
                    hazard_index.append(l_ind)
                else:
                    # move non-hazard landmark to random but known location 
                    l.state.p_pos = np.array([ln_x, ln_y])


            for a in self.world.agents:
                # move agents to random but know location
                # randomly set agents as terminated or not
                a.collide = False
                a.terminated = bool(np.random.binomial(1,0.5))
                a.state.p_pos = np.array([ag_x, ag_y])

            for a in self.world.agents:
                # check reward
                reward = self.scenario.reward(a, self.world)
                self.assertAlmostEqual(reward, exp_reward)

        # check that only one hazard exits per trial
        self.assertEqual(len(hazard_index), n_trials)

        # check that hazard index is shuffled ((1/n_landmarks)^n_trials probability of false negative)
        self.assertFalse(all(np.array(hazard_index)==hazard_index[0]))

    def test_ergo_spread_variable_hazard_rewards_3(self):
        ''' ergo_spread_variable: test non-zero rewards with distributed landmarks and agents'''
        hazard_index = []
        n_trials = 100
        x_arr = [-1.0, 1.0, 1.0, -1.0]
        y_arr = [1.0, 1.0, -1.0, -1.0]
        d_range = [0.1, 0.11]
        for i in range(n_trials):
            self.scenario.reset_world(self.world)
            ag_count = 0
            special_ag_index = np.random.randint(len(self.world.agents))
            for l_ind, l in enumerate(self.world.landmarks):
                l_x = x_arr[l_ind % 4]
                l_y = y_arr[l_ind % 4]
                if l.is_hazard:
                    # hazard ends up in random location based on reset, but
                    # should not affect reward
                    hazard_index.append(l_ind)
                else:
                    # move non-hazard landmark to random but known location 
                    l.state.p_pos = np.array([l_x, l_y])
                    if ag_count == special_ag_index:
                        # a_x = l_x + np.random.choice([-1.0,1.0])*np.random.uniform(*d_range)
                        # a_y = l_y + np.random.choice([-1.0,1.0])*np.random.uniform(*d_range)
                        theta = np.random.uniform(0, 2.0*np.pi)
                        d = np.random.uniform(*d_range)
                        a_x = l_x + d*np.cos(theta)
                        a_y = l_y + d*np.sin(theta)
                    else:
                        # a_x = l_x + np.random.uniform(-0.01, 0.01)
                        # a_y = l_y + np.random.uniform(-0.01, 0.01)
                        a_x = l_x
                        a_y = l_y
                    self.world.agents[ag_count].state.p_pos = np.array([a_x, a_y])
                    ag_count += 1

            for a in self.world.agents:
                # check reward
                reward = self.scenario.reward(a, self.world)
                self.assertLessEqual(reward, -d_range[0])
                self.assertLessEqual(-d_range[1], reward)

    def test_ergo_spread_variable_collisions_1(self):
        ''' ergo_spread_variable: collisions don't cause failures '''

        n_trials = 100
        n_steps = 100
        for trial_count in range(n_trials):

            # reset world and remove landmarks
            self.scenario.reset_world(self.world)
            self.world.landmarks = []

            for step_count in range(n_steps):
                ag_x = np.random.uniform(-1,1)
                ag_y = np.random.uniform(-1,1)
                for a in self.world.agents:
                    n_x = np.random.uniform(-1e-4,1e-4)
                    n_y = np.random.uniform(-1e-4,1e-4)
                    a.state.p_pos = np.array([ag_x+n_x, ag_y+n_y])
                    a.action.u = np.zeros(2)

                for a in self.world.agents:
                    for oa in self.world.agents:
                        if oa is a:
                            self.assertFalse(is_collision(a, oa))
                        else:
                            self.assertTrue(is_collision(a, oa))

                self.world.step()

                for a in self.world.agents:
                    self.assertFalse(a.terminated)

class TestErgoSpreadVariableCase2(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_spread_variable.Scenario(num_agents=5, num_hazards=1, identical_rewards=True, observation_type="histogram")

        # create world and modify to test conditions
        self.world = self.scenario.make_world()

    def test_ergo_spread_variable_case2_observations_1(self):
        ''' ergo_spread_variable: test observations with 5 agents (1 hazard, identical rewards, histogram observation)
        Notes:
         - Assume 4 radial and 8 angular bins with a max obs distance of 1.0
         - Formatted to work for either linear or logspaced bins
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.1, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[2].state.p_vel = np.array([0.0, 0.0])
        self.world.agents[3].state.p_pos = np.array([0.1, 0.0])
        self.world.agents[4].state.p_pos = np.array([1.0, 0.0])

        self.world.landmarks[0].state.p_pos = np.array([-1.0, 0.0]); self.world.landmarks[0].is_hazard = False
        self.world.landmarks[1].state.p_pos = np.array([-0.1, 0.0]); self.world.landmarks[1].is_hazard = False
        self.world.landmarks[2].state.p_pos = np.array([0.0, 0.0]); self.world.landmarks[2].is_hazard = True
        self.world.landmarks[3].state.p_pos = np.array([0.1, 0.0]); self.world.landmarks[3].is_hazard = False
        self.world.landmarks[4].state.p_pos = np.array([1.0, 0.0]); self.world.landmarks[4].is_hazard = False
        self.world.landmarks[5].state.p_pos = np.array([0.0, 10.0]); self.world.landmarks[5].is_hazard = False

        expected_histogram = np.array([[0]*8]*4)
        expected_histogram[3][4] = 1
        expected_histogram[0][4] = 1
        expected_histogram[0][0] = 1
        expected_histogram[3][0] = 1
        expected_histogram = [val for sublist in expected_histogram for val in sublist]
        expected_observation = ([0] + # termination
                                [0,0,0,0] + # vel and pos
                                expected_histogram + # landmark histogram
                                [0.0, 0.0] + # observed hazard
                                expected_histogram + # agent histogram
                                [0.0]*5*2 + # observed failures
                                [0]*81 # previous observation
                                )
        
        observation = self.scenario.observation(agent=self.world.agents[2], world=self.world)
        for i, obs in enumerate(observation):
            self.assertAlmostEqual(obs, expected_observation[i])

class TestErgoSpreadVariableCase3(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_spread_variable.Scenario(num_agents=20, num_hazards=1, identical_rewards=True, observation_type="histogram")

        # create world and modify to test conditions
        self.world = self.scenario.make_world()

    def test_ergo_spread_variable_case3_observations_1(self):
        ''' ergo_spread_variable: test observations with 20 agents (1 hazard, identical rewards, histogram observation)
        Notes:
         - Assume 4 radial and 8 angular bins with a max obs distance of 1.0
         - Formatted to work for either logspaced radial bins with partitions:
            [0.1       , 0.21544347, 0.46415888, 1.        ]
         - Assumes terminated agents are always observable
        '''

        rt2_2 = np.sqrt(2.0)/2.0 - np.finfo(float).eps
        self.world.agents[0].state.p_pos = np.array([0.0, 0.0])

        # immediate radial bins
        self.world.agents[1].state.p_pos = np.array([0.1, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.1*rt2_2, 0.1*rt2_2])
        self.world.agents[3].state.p_pos = np.array([0.0, 0.1])
        self.world.agents[4].state.p_pos = np.array([-0.1*rt2_2, 0.1*rt2_2])
        self.world.agents[5].state.p_pos = np.array([-0.1, 0.0])
        self.world.agents[6].state.p_pos = np.array([-0.1*rt2_2, -0.1*rt2_2])
        self.world.agents[7].state.p_pos = np.array([0.0, -0.1])
        self.world.agents[8].state.p_pos = np.array([0.1*rt2_2, -0.1*rt2_2])

        # 2 agents in radial bin 1 (0.1-0.21544347], angular bin 5 (9*pi/8-11*pi/8]
        for i in range(9,11):
            dist = np.random.uniform(0.1, 0.21544347)
            ang = np.random.uniform(9.0*np.pi/8.0, 11.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.agents[i].state.p_pos = np.array([x,y])

        # 1 agents in radial bin 3, angular bin 7
        for i in range(11,12):
            dist = np.random.uniform(0.46415888, 1.)
            ang = np.random.uniform(13.0*np.pi/8.0, 15.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.agents[i].state.p_pos = np.array([x,y])

        # 1 agents in radial bin 2, angular bin 1
        for i in range(12,13):
            dist = np.random.uniform(0.21544347, 0.46415888)
            ang = np.random.uniform(1.0*np.pi/8.0, 3.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.agents[i].state.p_pos = np.array([x,y])

        # 2 more agents in radial bin 0, angular bin 0
        for i in range(13,15):
            dist = np.random.uniform(0., 0.1)
            ang = np.random.uniform(-1.0*np.pi/8.0, 1.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.agents[i].state.p_pos = np.array([x,y])

        # 2 agents outside of range
        for i in range(15,17):
            dist = np.random.uniform(1.0, 1000.0)
            ang = np.random.uniform(-2.0*np.pi/8.0, 2.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.agents[i].state.p_pos = np.array([x,y])

        # 3 remaining are terminated at random locations
        failure_locations = []
        failure_distances = []
        for i in range(17,20):
            failure_locations.append(np.random.rand(2)*2-1)
            failure_distances.append(np.linalg.norm(failure_locations[-1]))
            self.world.agents[i].state.p_pos = failure_locations[-1]
            self.world.agents[i].terminated = True

        # landmarks
        for lm in self.world.landmarks:
            lm.is_hazard = False
            lm.hazard_tag = 0.0
        self.world.landmarks[1].state.p_pos = np.array([0.1, 0.0])
        self.world.landmarks[2].state.p_pos = np.array([0.1*rt2_2, 0.1*rt2_2])
        self.world.landmarks[3].state.p_pos = np.array([0.0, 0.1])
        self.world.landmarks[4].state.p_pos = np.array([-0.1*rt2_2, 0.1*rt2_2])
        self.world.landmarks[5].state.p_pos = np.array([-0.1, 0.0])
        self.world.landmarks[6].state.p_pos = np.array([-0.1*rt2_2, -0.1*rt2_2])
        self.world.landmarks[7].state.p_pos = np.array([0.0, -0.1])
        self.world.landmarks[8].state.p_pos = np.array([0.1*rt2_2, -0.1*rt2_2])

        # 2 landmarks in radial bin 1 (0.1-0.21544347], angular bin 3 (9*pi/8-11*pi/8]
        for i in range(9,11):
            dist = np.random.uniform(0.1, 0.21544347)
            ang = np.random.uniform(5.0*np.pi/8.0, 7.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.landmarks[i].state.p_pos = np.array([x,y])

        # 1 landmarks in radial bin 3, angular bin 2
        for i in range(11,12):
            dist = np.random.uniform(0.46415888, 1.)
            ang = np.random.uniform(3.0*np.pi/8.0, 5.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.landmarks[i].state.p_pos = np.array([x,y])

        # 1 landmarks in radial bin 2, angular bin 5
        for i in range(12,13):
            dist = np.random.uniform(0.21544347, 0.46415888)
            ang = np.random.uniform(9.0*np.pi/8.0, 11.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.landmarks[i].state.p_pos = np.array([x,y])

        # 2 landmarks in radial bin 1, angular bin 0
        for i in range(13,15):
            dist = np.random.uniform(0.1, 0.21544347)
            ang = np.random.uniform(-1.0*np.pi/8.0, 1.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.landmarks[i].state.p_pos = np.array([x,y])

        # 5 landmarks outside of range
        for i in [0, 15, 16, 17, 18, 19]:
            dist = np.random.uniform(1.0, 1000.0)
            ang = np.random.uniform(-2.0*np.pi/8.0, 2.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.landmarks[i].state.p_pos = np.array([x,y])

        # 1 hazard outside of range
        for i in [20]:
            dist = np.random.uniform(0.0, 1000.0)
            ang = np.random.uniform(-2.0*np.pi/8.0, 2.0*np.pi/8.0)
            x = dist*np.cos(ang)
            y = dist*np.sin(ang)
            self.world.landmarks[i].state.p_pos = np.array([x,y])
            self.world.landmarks[i].is_hazard = True
            self.world.landmarks[i].hazard_tag = 1.0


        expected_agent_histogram = np.array([[0]*8]*4)
        expected_agent_histogram[0][0] = 1
        expected_agent_histogram[0][1] = 1
        expected_agent_histogram[0][2] = 1
        expected_agent_histogram[0][3] = 1
        expected_agent_histogram[0][4] = 1
        expected_agent_histogram[0][5] = 1
        expected_agent_histogram[0][6] = 1
        expected_agent_histogram[0][7] = 1
        expected_agent_histogram[1][5] = 2
        expected_agent_histogram[3][7] = 1
        expected_agent_histogram[2][1] = 1
        expected_agent_histogram[0][0] = 3
        expected_agent_histogram = [val for sublist in expected_agent_histogram for val in sublist]

        expected_failures = [x for _,x in sorted(zip(failure_distances,failure_locations))]
        expected_failures = [val for sublist in expected_failures for val in sublist]
        expected_failures.extend([0.0, 0.0, 0.0, 0.0])

        expected_landmark_histogram = np.array([[0]*8]*4)
        expected_landmark_histogram[0][0] = 1
        expected_landmark_histogram[0][1] = 1
        expected_landmark_histogram[0][2] = 1
        expected_landmark_histogram[0][3] = 1
        expected_landmark_histogram[0][4] = 1
        expected_landmark_histogram[0][5] = 1
        expected_landmark_histogram[0][6] = 1
        expected_landmark_histogram[0][7] = 1
        expected_landmark_histogram[1][3] = 2
        expected_landmark_histogram[3][2] = 1
        expected_landmark_histogram[2][5] = 1
        expected_landmark_histogram[1][0] = 2
        expected_landmark_histogram = [val for sublist in expected_landmark_histogram for val in sublist]

        expected_hazard = list(self.world.landmarks[20].state.p_pos)

        expected_observation = ([0] + # agent terminated
                                [0,0,0,0] + # vel and pos
                                expected_landmark_histogram + 
                                expected_hazard +
                                expected_agent_histogram + # agent histogram
                                expected_failures + # observed failures
                                [0]*81 # previous observation
                                )

        observation = self.scenario.observation(agent=self.world.agents[0], world=self.world)
        for i, obs in enumerate(observation):
            self.assertAlmostEqual(obs, expected_observation[i])

class TestErgoHazardsCase1(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_hazards.Scenario()
        self.scenario.num_agents = 2

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.landmarks = [TORRLandmark( risk_fn=RadialRisk(0.1), reward_fn=RadialReward(0.1, 1.0))]
        self.world.landmarks[0].state.p_pos = np.array([0.0, 0.0])
        self.world.landmarks[0].collide = False
        self.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([1.0, 0.0])
        self.world.max_communication_distance = 1.1


    def test_ergo_hazards_case1_setup(self):
        ''' check setup ErgoHazardsCase1'''
        self.assertEqual(self.scenario.num_agents, 2)
        self.assertEqual(len(self.world.landmarks), 1)
        self.assertAlmostEqual(self.world.landmarks[0].cumulative_distributed_reward, 0.0)
        self.assertAlmostEqual(self.world.landmarks[0].observe_clock, 0.0)
        self.assertFalse(self.world.landmarks[0].is_observable())
        self.assertAlmostEqual(self.world.landmarks[0].risk_fn.sample_failure(0.0,0), 1.0)
        self.assertAlmostEqual(self.world.landmarks[0].risk_fn.sample_failure(0.0,0.2), 0.0)
        self.assertAlmostEqual(self.world.landmarks[0].reward_fn.get_value(0.0,0), 1.0)
        self.assertAlmostEqual(self.world.landmarks[0].reward_fn.get_value(0.2,0), 0.0)


    def test_ergo_hazards_case1_world_step_1(self):
        ''' check ErgoHazardsCase1 steps world as expected'''

        # set agent actions
        # self.world._set_action([0.0, 0.0], self.world.agents[0], self.world.action_space[0])
        # self.world._set_action([0.0, 0.0], self.world.agents[1], self.world.action_space[1])
        self.world.agents[0].action.u = np.zeros(2)
        self.world.agents[1].action.u = np.zeros(2)

        # time step world
        self.world.step()

        # Agent 0 should have been terminated with probability 1
        self.assertTrue(self.world.agents[0].terminated)

        # Agent 1 should have been terminated with probability 0
        self.assertFalse(self.world.agents[1].terminated)

        # Landmark should be observable and incremented reward count
        self.assertTrue(self.world.landmarks[0].is_observable())
        

        # To test reward updates, we need to manually call for each agent
        # since this is normally handled by the environment, not world
        # record observation for each agent
        # obs_n = [None]*2
        reward_n = [0.0]*2
        done_n = [None]*2
        for i, agent in enumerate(self.world.agents):
            # obs_n[i] = self.scenario.observation(agent, self.world)
            reward_n[i] = self.scenario.reward(agent, self.world)
            done_n[i] = self.scenario.done_callback(agent, self.world)


        # Even though landmark is observable, no reward should have been distributed
        self.assertAlmostEqual(self.world.landmarks[0].cumulative_distributed_reward, 0.0)
        self.assertAlmostEqual(reward_n[0], 0.0)
        self.assertAlmostEqual(reward_n[1], 0.0)
        self.assertTrue(done_n[0])
        self.assertFalse(done_n[1])

    def test_ergo_hazards_case1_world_step_2(self):
        ''' check ErgoHazardsCase1 steps world as expected'''

        # adjust landmark and agent 1
        self.world.landmarks[0].reward_fn = RadialReward(1.0, 1.0)
        self.world.agents[1].state.p_pos = np.array([0.2, 0.0])

        # set agent actions
        self.world.agents[0].action.u = np.zeros(2)
        self.world.agents[1].action.u = np.zeros(2)

        # time step world
        self.world.step()

        # Agents terminated with known probability and landmark made observable
        self.assertTrue(self.world.agents[0].terminated)
        self.assertFalse(self.world.agents[1].terminated)
        self.assertTrue(self.world.landmarks[0].is_observable())
        
        # record reward for each agent
        reward_n = [0.0]*2
        done_n = [None]*2
        for i, agent in enumerate(self.world.agents):
            reward_n[i] = self.scenario.reward(agent, self.world)
            done_n[i] = self.scenario.done_callback(agent, self.world)


        # Even though landmark is observable, no reward should have been distributed
        self.assertAlmostEqual(self.world.landmarks[0].cumulative_distributed_reward, 1.0 - 0.2**2)
        self.assertAlmostEqual(reward_n[0], 0.0)
        self.assertAlmostEqual(reward_n[1], 1 - 0.2**2)
        self.assertTrue(done_n[0])
        self.assertFalse(done_n[1])

    def test_ergo_hazards_case1_world_reset_1(self):
        ''' check ErgoHazardsCase1 resets world as expected'''

        # set agent actions
        self.world.agents[0].action.u = np.zeros(2)
        self.world.agents[1].action.u = np.zeros(2)

        # time step world
        self.world.step()
        self.scenario.reset_world(self.world)

        self.world.agents[0].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([1.0, 0.0])

        # Agents terminated with known probability and landmark made observable
        self.assertFalse(self.world.agents[0].terminated)
        self.assertFalse(self.world.agents[1].terminated)
        self.assertFalse(self.world.landmarks[0].is_observable())
        self.assertAlmostEqual(self.world.landmarks[0].cumulative_distributed_reward, 0.0)


class TestErgoSpreadSmallCase1(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_spread_small.Scenario()

        # create world and modify to test conditions
        self.world = self.scenario.make_world()

    def test_ergo_spread_small_hazard_rewards_1(self):
        ''' ergo_spread_small: hazards provide no reward and is randomly shuffled, termination doesn't affect reward '''

        hazard_index = []
        n_trials = 100
        for i in range(n_trials):
            agent_indices = np.arange(self.scenario.num_agents)
            shuffle(agent_indices)
            ag_count = 0
            self.scenario.reset_world(self.world)
            for l_ind, l in enumerate(self.world.landmarks):
                if l.is_hazard:
                    hazard_index.append(l_ind)
                else:
                    # move an agent directly on top of 
                    self.world.agents[agent_indices[ag_count]].state.p_pos = deepcopy(l.state.p_pos)
                    ag_count += 1

            for a in self.world.agents:
                # non hazard landmarks should be covered by an agent and
                # all agents should have same reward, thus all rewards should
                # be zero
                # agents are randomly terminated, but this should have no effect
                # on reward
                a.collide = False
                a.terminated = bool(np.random.binomial(1,0.5))
                reward = self.scenario.reward(a, self.world)
                self.assertAlmostEqual(reward, 0.0)

        # check that only one hazard exits per trial
        self.assertEqual(len(hazard_index), n_trials)

        # check that hazard index is shuffled ((1/n_landmarks)^n_trials probability of false negative)
        self.assertFalse(all(np.array(hazard_index)==hazard_index[0]))

    def test_ergo_spread_small_hazard_rewards_2(self):
        ''' ergo_spread_small: test non-zero rewards with collocated agents and landmarks'''

        hazard_index = []
        n_trials = 100
        for i in range(n_trials):
            ag_x = np.random.uniform(-1,1)
            ag_y = np.random.uniform(-1,1)
            ln_x = np.random.uniform(-1,1)
            ln_y = np.random.uniform(-1,1)
            exp_reward = -np.sqrt((ag_x-ln_x)**2 + (ag_y-ln_y)**2)
            exp_reward *= len([l for l in self.world.landmarks if not l.is_hazard])
            self.scenario.reset_world(self.world)
            for l_ind, l in enumerate(self.world.landmarks):
                if l.is_hazard:
                    hazard_index.append(l_ind)
                else:
                    # move non-hazard landmark to random but known location 
                    l.state.p_pos = np.array([ln_x, ln_y])


            for a in self.world.agents:
                # move agents to random but know location
                # randomly set agents as terminated or not
                a.collide = False
                a.terminated = bool(np.random.binomial(1,0.5))
                a.state.p_pos = np.array([ag_x, ag_y])

            for a in self.world.agents:
                # check reward
                reward = self.scenario.reward(a, self.world)
                self.assertAlmostEqual(reward, exp_reward)

        # check that only one hazard exits per trial
        self.assertEqual(len(hazard_index), n_trials)

        # check that hazard index is shuffled ((1/n_landmarks)^n_trials probability of false negative)
        self.assertFalse(all(np.array(hazard_index)==hazard_index[0]))

    def test_ergo_spread_small_hazard_rewards_3(self):
        ''' ergo_spread_small: test non-zero rewards with distributed landmarks and agents'''
        hazard_index = []
        n_trials = 100
        x_arr = [-1.0, 1.0, 1.0, -1.0]
        y_arr = [1.0, 1.0, -1.0, -1.0]
        d_range = [0.1, 0.11]
        for i in range(n_trials):
            self.scenario.reset_world(self.world)
            ag_count = 0
            special_ag_index = np.random.randint(len(self.world.agents))
            for l_ind, l in enumerate(self.world.landmarks):
                l_x = x_arr[l_ind % 4]
                l_y = y_arr[l_ind % 4]
                if l.is_hazard:
                    # hazard ends up in random location based on reset, but
                    # should not affect reward
                    hazard_index.append(l_ind)
                else:
                    # move non-hazard landmark to random but known location 
                    l.state.p_pos = np.array([l_x, l_y])
                    if ag_count == special_ag_index:
                        # a_x = l_x + np.random.choice([-1.0,1.0])*np.random.uniform(*d_range)
                        # a_y = l_y + np.random.choice([-1.0,1.0])*np.random.uniform(*d_range)
                        theta = np.random.uniform(0, 2.0*np.pi)
                        d = np.random.uniform(*d_range)
                        a_x = l_x + d*np.cos(theta)
                        a_y = l_y + d*np.sin(theta)
                    else:
                        # a_x = l_x + np.random.uniform(-0.01, 0.01)
                        # a_y = l_y + np.random.uniform(-0.01, 0.01)
                        a_x = l_x
                        a_y = l_y
                    self.world.agents[ag_count].state.p_pos = np.array([a_x, a_y])
                    ag_count += 1

            for a in self.world.agents:
                # check reward
                reward = self.scenario.reward(a, self.world)
                self.assertLessEqual(reward, -d_range[0])
                self.assertLessEqual(-d_range[1], reward)

    def test_ergo_spread_small_collisions_1(self):
        ''' ergo_spread_small: collisions don't cause failures '''

        n_trials = 100
        n_steps = 100
        for trial_count in range(n_trials):

            # reset world and remove landmarks
            self.scenario.reset_world(self.world)
            self.world.landmarks = []

            for step_count in range(n_steps):
                ag_x = np.random.uniform(-1,1)
                ag_y = np.random.uniform(-1,1)
                for a in self.world.agents:
                    n_x = np.random.uniform(-1e-4,1e-4)
                    n_y = np.random.uniform(-1e-4,1e-4)
                    a.state.p_pos = np.array([ag_x+n_x, ag_y+n_y])
                    a.action.u = np.zeros(2)

                for a in self.world.agents:
                    for oa in self.world.agents:
                        if oa is a:
                            self.assertFalse(is_collision(a, oa))
                        else:
                            self.assertTrue(is_collision(a, oa))

                self.world.step()

                for a in self.world.agents:
                    self.assertFalse(a.terminated)
                
class TestErgoPerimeterSmallCase1(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_perimeter_small.Scenario()

        # create world and modify to test conditions
        self.world = self.scenario.make_world()

    def test_ergo_perimeter_small_rewards_1(self):
        ''' ergo_perimeter_small: equilateral triangle around unit circle landmark'''

        # check scenario has one landmark of unit radius
        self.assertEqual(len(self.world.landmarks), 1)
        # self.assertAlmostEqual(self.world.landmarks[0].size, 1.0)

        # calculate expected reward
        r = float(ergo_perimeter_small._LANDMARK_SIZE)
        coefs = [1.0, 0.0, 0.0, -1.0/float(r**2), 0.0, -1.0/float(r**2)]
        mvp = MVP2D(coefs)
        l1 = r*np.cos(np.pi/6.0)
        l2 = r*np.sin(np.pi/6.0)
        expected_reward = 0.0
        expected_reward += mvp.augmented_line_integral([0.0, r], [-l1, -l2])
        expected_reward += mvp.augmented_line_integral([-l1, -l2], [l1, -l2])
        expected_reward += mvp.augmented_line_integral([l1, -l2], [0.0, r])

        # normalize by total positive area integral
        # Note: this is the area under a parabolic curve, not a circle
        # expected_reward /= (8./15. * np.pi * 1.0**2 * r)
        expected_reward /= 0.5 * np.pi * r**2

        agent_indices = np.arange(self.scenario.num_agents)
        n_trials = 10
        for i in range(n_trials):
            shuffle(agent_indices)
            self.scenario.reset_world(self.world)

            # enforce position of landmark
            self.world.landmarks[0].state.p_pos = np.array([0.0, 0.0])

            # move agents to equalateral triangle around landmark
            self.world.agents[agent_indices[0]].state.p_pos = np.array([0.0, r])
            self.world.agents[agent_indices[1]].state.p_pos = np.array([-l1, -l2])
            self.world.agents[agent_indices[2]].state.p_pos = np.array([l1, -l2])

            # calculate reward
            reward_n = self.scenario.reward(None, self.world, systemic_call=True)
            for ind, rew in enumerate(reward_n):
                self.assertAlmostEqual(self.world.landmarks[0].reward_fn.get_value(*self.world.agents[ind].state.p_pos), 0.0)
                self.assertGreaterEqual(rew, 0.0)
                self.assertLessEqual(rew, 1.0)
                self.assertAlmostEqual(rew,  expected_reward)

class TestErgoPerimeterVariableCase1(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_perimeter_variable.Scenario(num_agents=4, num_hazards=0, identical_rewards=True, observation_type="direct")

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.landmarks[0].state.p_pos = np.zeros(self.world.dim_p)

    def test_ergo_perimeter_variable_case1_systemic_reward_1(self):
        ''' ergo_perimeter_variable: systemic reward for 4 agents arranged in square (0 hazard, identical rewards, direct observation)
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, -1.0])
        self.world.agents[1].state.p_pos = np.array([-1.0, 1.0])
        self.world.agents[2].state.p_pos = np.array([1.0, 1.0])
        self.world.agents[3].state.p_pos = np.array([1.0, -1.0])

        # calculate expected reward (peak values divide out to 1)
        lmr = ergo_perimeter_variable._LANDMARK_SIZE
        expect_reward = 4. - 8./(3.*lmr**2)
        expect_reward /= np.pi*lmr**2/2.

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, expect_reward)

    def test_ergo_perimeter_variable_case1_systemic_reward_2(self):
        ''' ergo_perimeter_variable: systemic reward for 4 collinear agents (0 hazard, identical rewards, direct observation)
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[3].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, 0.0)

    def test_ergo_perimeter_variable_case1_observations_1(self):
        ''' ergo_perimeter_variable: observations of 4 agents around landmark (0 hazard, identical rewards, direct observation)
        Notes:
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, -1.0])
        self.world.agents[1].state.p_pos = np.array([-1.0, 1.0])
        self.world.agents[2].state.p_pos = np.array([1.0, 1.0])
        self.world.agents[3].state.p_pos = np.array([1.0, -1.0])

        self.world.agents[0].state.p_vel = np.zeros(2)
        self.world.agents[1].state.p_vel = np.zeros(2)
        self.world.agents[2].state.p_vel = np.zeros(2)
        self.world.agents[3].state.p_vel = np.zeros(2)


        pkv = ergo_perimeter_variable._PEAK_REWARD
        lmr = ergo_perimeter_variable._LANDMARK_SIZE
        obs_rew = pkv*(1.-2./(lmr**2))
        expect_obs0 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                -1., -1.,                # pos
                                obs_rew,                # landmark observation (sensed reward)
                                0, 0., 2., 0., 0., obs_rew, # agent 1 observation
                                0, 2., 2., 0., 0., obs_rew, # agent 2 observation
                                0, 2., 0., 0., 0., obs_rew, # agent 3 observation
                                ])
        expect_obs0 = np.concatenate([expect_obs0, np.zeros(24)])
        expect_obs1 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                -1., 1.,                # pos
                                obs_rew,                # landmark observation (sensed reward)
                                0, 0., -2., 0., 0., obs_rew, # agent 0 observation
                                0, 2., 0., 0., 0., obs_rew,  # agent 2 observation
                                0, 2., -2., 0., 0., obs_rew, # agent 3 observation
                                ])
        expect_obs1 = np.concatenate([expect_obs1, np.zeros(24)])
        expect_obs2 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                1., 1.,                 # pos
                                obs_rew,                # landmark observation (sensed reward)
                                0, -2., -2., 0., 0., obs_rew, # agent 0 observation
                                0, -2., 0., 0., 0., obs_rew,  # agent 1 observation
                                0, 0., -2., 0., 0., obs_rew,  # agent 3 observation
                                ])
        expect_obs2 = np.concatenate([expect_obs2, np.zeros(24)])
        expect_obs3 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                1., -1.,                # pos
                                obs_rew,                # landmark observation (sensed reward)
                                0, -2., 0., 0., 0., obs_rew,  # agent 0 observation
                                0, -2., 2., 0., 0., obs_rew,  # agent 1 observation
                                0, 0.,  2., 0., 0., obs_rew,  # agent 2 observation
                                ])
        expect_obs3 = np.concatenate([expect_obs3, np.zeros(24)])

        observation0 = self.scenario.observation(agent=self.world.agents[0], world=self.world)
        observation1 = self.scenario.observation(agent=self.world.agents[1], world=self.world)
        observation2 = self.scenario.observation(agent=self.world.agents[2], world=self.world)
        observation3 = self.scenario.observation(agent=self.world.agents[3], world=self.world)
        for i, _ in enumerate(observation0):
            self.assertAlmostEqual(observation0[i], expect_obs0[i])
            self.assertAlmostEqual(observation1[i], expect_obs1[i])
            self.assertAlmostEqual(observation2[i], expect_obs2[i])
            self.assertAlmostEqual(observation3[i], expect_obs3[i])


    def test_ergo_perimeter_variable_case1_observations_2(self):
        ''' ergo_perimeter_variable: observations with 4 agents aligned w/ landmark (0 hazard, identical rewards, direct observation)
        Notes:
        '''

        self.world.agents[0].state.p_pos = np.array([-1.5, 0.0])
        self.world.agents[1].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[2].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[3].state.p_pos = np.array([-0.0, 0.0])

        self.world.agents[0].state.p_vel = np.zeros(2)
        self.world.agents[1].state.p_vel = np.zeros(2)
        self.world.agents[2].state.p_vel = np.zeros(2)
        self.world.agents[3].state.p_vel = np.zeros(2)


        pkv = ergo_perimeter_variable._PEAK_REWARD
        lmr = ergo_perimeter_variable._LANDMARK_SIZE
        obs_rew0 = pkv*(1.-(1.5/lmr)**2)
        obs_rew1 = pkv*(1.-(1.0/lmr)**2)
        obs_rew2 = pkv*(1.-(0.5/lmr)**2)
        obs_rew3 = pkv
        expect_obs0 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                -1.5, 0.,                # pos
                                obs_rew0,                # landmark observation (sensed reward)
                                0, 0.5, 0., 0., 0., obs_rew1, # agent 1 observation
                                0, 1.0, 0., 0., 0., obs_rew2, # agent 2 observation
                                0, 1.5, 0., 0., 0., obs_rew3, # agent 3 observation
                                ])
        expect_obs0 = np.concatenate([expect_obs0, np.zeros(24)])
        expect_obs1 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                -1.0, 0.,                # pos
                                obs_rew1,                # landmark observation (sensed reward)
                                0, -0.5, 0., 0., 0., obs_rew0, # agent 0 observation
                                0, 0.5, 0., 0., 0., obs_rew2,  # agent 2 observation
                                0, 1.0, 0., 0., 0., obs_rew3, # agent 3 observation
                                ])
        expect_obs1 = np.concatenate([expect_obs1, np.zeros(24)])
        expect_obs2 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                -0.5, 0.,                 # pos
                                obs_rew2,                # landmark observation (sensed reward)
                                0, -1.0, 0., 0., 0., obs_rew0, # agent 0 observation
                                0, -0.5, 0., 0., 0., obs_rew1,  # agent 1 observation
                                0, 0.5, 0., 0., 0., obs_rew3,  # agent 3 observation
                                ])
        expect_obs2 = np.concatenate([expect_obs2, np.zeros(24)])
        expect_obs3 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                0., 0.,                # pos
                                obs_rew3,                # landmark observation (sensed reward)
                                0, -1.5, 0., 0., 0., obs_rew0,  # agent 0 observation
                                0, -1.0, 0., 0., 0., obs_rew1,  # agent 1 observation
                                0, -0.5, 0., 0., 0., obs_rew2,  # agent 2 observation
                                ])
        expect_obs3 = np.concatenate([expect_obs3, np.zeros(24)])

        observation0 = self.scenario.observation(agent=self.world.agents[0], world=self.world)
        observation1 = self.scenario.observation(agent=self.world.agents[1], world=self.world)
        observation2 = self.scenario.observation(agent=self.world.agents[2], world=self.world)
        observation3 = self.scenario.observation(agent=self.world.agents[3], world=self.world)
        for i, _ in enumerate(observation0):
            self.assertAlmostEqual(observation0[i], expect_obs0[i])
            self.assertAlmostEqual(observation1[i], expect_obs1[i])
            self.assertAlmostEqual(observation2[i], expect_obs2[i])
            self.assertAlmostEqual(observation3[i], expect_obs3[i])

class TestErgoPerimeterVariableMAPPOCase1(unittest.TestCase):

    def setUp(self):
        ''' the with tf.Graph.as_default()... command allows for multiple calls to setUp
        without causing variable scopes to "clash". See baselines/common/tests/util.py for examples
        '''

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default():
           
            self.test_n_training_iterations = 1024
            self.test_episode_len = 4
            self.test_n_episodes_per_batch = 256
            self.test_num_layers = 8
            self.test_num_units = 64
            self.test_activation = 'elu'
            self.test_learning_rate = 1e-2
            self.test_n_opt_epochs = 8
            self.test_n_minibatches = 8
            self.test_gamma = 0.0001
            self.test_n_agents = 4
            self.test_n_hazards = 1
            # self.test_joint_state_space_len = 5*self.test_n_agents + 5*self.test_n_hazards
            deep_mlp = DeepMLP(num_layers=self.test_num_layers, activation=self.test_activation)

            # Create Scenario
            # create Scenario object and modify to test conditions
            scenario = ergo_perimeter_variable.Scenario(   
                num_agents=self.test_n_agents, 
                num_hazards=self.test_n_hazards, 
                identical_rewards=True, observation_type="direct")

            # create world and modify to test conditions
            world = scenario.make_world()
            # world.landmarks[0].state.p_pos = np.zeros(world.dim_p)

            # create environment
            self.env = MultiAgentRiskEnv(
                        world=world, 
                        reset_callback=scenario.reset_world, 
                        reward_callback=scenario.reward, 
                        observation_callback=scenario.observation, 
                        done_callback=scenario.done_callback,
                        discrete_action_space=False,
                        legacy_multidiscrete=False)

            joint_state = self.env.get_joint_state()
            self.test_joint_state_space_len = 0
            for js in joint_state['state']:
                self.test_joint_state_space_len += len(js)

            # create trainer that would live in a simple 2D environment
            self.group_trainer = PPOGroupTrainer(
                    n_agents=self.test_n_agents, 
                    obs_space=self.env.observation_space[0], # observation space should be irrelevant
                    act_space=self.env.action_space[0], # action space should be irrelevant
                    n_steps_per_episode=self.test_episode_len, 
                    ent_coef=0.0,                       # should be irrelevant 
                    local_actor_learning_rate=3e-4,     # should be irrelevant
                    vf_coef=0.5,                        # should be irrelevant
                    num_layers=2, num_units=4,          # should be irrelevant 
                    activation=self.test_activation,    # should be irrelevant
                    cliprange=0.2,                      # should be irrelevant 
                    n_episodes_per_batch=self.test_n_episodes_per_batch, 
                    shared_reward=True,
                    critic_type='central_joint_state', 
                    central_critic_model=deep_mlp.deep_mlp_model, 
                    central_critic_learning_rate=self.test_learning_rate, 
                    central_critic_num_units=self.test_num_units,
                    joint_state_space_len=self.test_joint_state_space_len,
                    max_grad_norm = 0.5, n_opt_epochs=self.test_n_opt_epochs, n_minibatches=self.test_n_minibatches)


            # Populate the group with stripped out versions of agents
            Args = namedtuple('Args', ['max_episode_len', 'gamma'])
            args = Args(self.test_episode_len, self.test_gamma)
            agent_0 = PPOAgentComputer(
                name="agent_0", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=0, args=args, local_q_func=None, lam=1.0)
            agent_1 = PPOAgentComputer(
                name="agent_1", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=1, args=args, local_q_func=None, lam=1.0)
            agent_2 = PPOAgentComputer(
                name="agent_2", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=2, args=args, local_q_func=None, lam=1.0)
            agent_3 = PPOAgentComputer(
                name="agent_3", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=2, args=args, local_q_func=None, lam=1.0)
            self.group_trainer.update_agent_trainer_group([agent_0, agent_1, agent_2, agent_3])

    def tearDown(self):
        '''Don't actually tearDown the tf graph
        Note: it may seem tempting to use tf.reset_default_graph(), but this
        causes an error in subsequent setUp calls with something to do with 
        op: NoOp ... is not an element of this graph
        Instead use the with tf.Graph.as_default()... in setUp
        '''
        pass

    def regtest_ergo_perimeter_variable_central_critic_network_learning_1(self):
        '''ergo_perimeter_variable+central_critic_network: critic learning with terminated, non-moving agents'''
        

        # in order to make calls to the central value function, we need to operate within the tf session
        # and initialize variables
        # with self.group_trainer.sess:
        #     tf.global_variables_initializer()
        with self.group_trainer.sess as sess:
            sess.run(tf.global_variables_initializer())

            central_vf_loss = []
            central_vf_expvar = []
            for train_iter in range(self.test_n_training_iterations):
                for ep in range(self.test_n_episodes_per_batch):

                    # reset world
                    self.env.reset()

                    # terminate all agents so that policy is irrelevant
                    for agi, ag in enumerate(self.env.world.agents):
                        self.assertTrue(ag.name == "agent_{}".format(agi)) # check ordering is always preserved
                        ag.terminated = True
                        ag.action.u = np.zeros(self.env.world.dim_p)

                    # collect initial observation after all agents are terminated
                    observation_n = []
                    for agi, ag in enumerate(self.env.world.agents):
                        self.assertTrue(ag.name == "agent_{}".format(agi))
                        self.env._get_obs(ag)   # run extra time to "flush out" agent.previous_observation
                        observation_n.append(self.env._get_obs(ag))
                    init_observation_n = observation_n

                    # get systemic reward which should remain constant given the non-moving agents
                    init_reward = self.env.reward_callback(None, self.env.world, systemic_call=True)
                    init_reward = self.group_trainer.n_agents*np.asarray(init_reward)
                    self.assertTrue(np.allclose(len(init_reward)*[init_reward[0]], init_reward))

                    # for each episode, the group batch data should grow by number of agents*time steps
                    self.assertEqual(len(self.group_trainer.batch_observations), 
                        self.group_trainer.n_agents*self.group_trainer.n_steps_per_episode*ep)
                    self.assertEqual(len(self.group_trainer.batch_factual_values),
                        len(self.group_trainer.batch_observations))
                    self.assertEqual(len(self.group_trainer.batch_counterfactual_values),
                        len(self.group_trainer.batch_observations))
                    self.assertEqual(len(self.group_trainer.batch_returns),
                        len(self.group_trainer.batch_observations))
                    self.assertEqual(len(self.group_trainer.batch_actions),
                        len(self.group_trainer.batch_observations))
                    self.assertEqual(len(self.group_trainer.batch_neglogp_actions),
                        len(self.group_trainer.batch_observations))
                    self.assertEqual(len(self.group_trainer.batch_dones),
                        len(self.group_trainer.batch_observations))

                    # for each episode, joint data should grow by number of time steps per ep + 1
                    self.assertEqual(len(self.group_trainer.batch_joint_state_stamped), 
                        (self.group_trainer.n_steps_per_episode+1)*ep)
                    self.assertEqual(len(self.group_trainer.batch_joint_returns), 
                        (self.group_trainer.n_steps_per_episode+1)*ep)

                    # walk through each step of the episode with all agents terminated
                    cur_joint_state = self.env.get_joint_state()
                    init_joint_state = cur_joint_state
                    init_joint_state_arr = np.concatenate(init_joint_state['state'], axis=0)
                    for tstep in range(self.test_episode_len):

                        self.group_trainer.record_joint_state(cur_joint_state)

                        # Get and record agent's observations and actions
                        for agi, trainer in enumerate(self.group_trainer.agent_trainer_group):
                            
                            # check physical agent (world) aligns with agent computer (trainer)
                            world_agent = self.env.world.agents[agi]
                            self.assertTrue(world_agent.name == trainer.name)
                            self.assertTrue(world_agent.terminated)

                            # get and store observation and check that it aligns with joint state and intial observation
                            trainer.mbi_observations.append(self.env.observation_callback(world_agent, self.env.world)) # shouldn't matter but still used for testing
                            self.assertTrue(np.allclose(init_observation_n[agi],observation_n[agi]))
                            self.assertTrue(np.allclose(init_observation_n[agi],trainer.mbi_observations[-1]))
                            ag_js_ind = self.group_trainer.joint_state_labels.index(world_agent.name)
                            # check termination
                            self.assertAlmostEqual(trainer.mbi_observations[-1][0],
                                                    self.group_trainer.episode_joint_state[-1][ag_js_ind*5+4])
                            # check velocity
                            self.assertAlmostEqual(trainer.mbi_observations[-1][1],
                                                    self.group_trainer.episode_joint_state[-1][ag_js_ind*5+2])
                            self.assertAlmostEqual(trainer.mbi_observations[-1][2],
                                                    self.group_trainer.episode_joint_state[-1][ag_js_ind*5+3])
                            # check position
                            self.assertAlmostEqual(trainer.mbi_observations[-1][3],
                                                    self.group_trainer.episode_joint_state[-1][ag_js_ind*5+0])
                            self.assertAlmostEqual(trainer.mbi_observations[-1][4],
                                                    self.group_trainer.episode_joint_state[-1][ag_js_ind*5+1])

                            # Note: this test highlighted to me the important distinction between the action the policy "commands"
                            # vs the action that is executed. Not sure how to more completely handle this, so just doing a 
                            # brute force overwrite
                            ag_act_cmd, ag_obs_val, ag_neglogp_cmd = trainer.action(trainer.mbi_observations[-1])
                            ag_act_exe = ag_act_cmd * 0.0
                            ag_neglogp_exe = ag_neglogp_cmd[0] * 0.0
                            trainer.mbi_actions.append(ag_act_exe) # should be zero (but perhaps not because this is what the policy requests which is different than what is physically acted)
                            trainer.mbi_obs_values.append(ag_obs_val) # shouldn't matter
                            trainer.mbi_neglogp_actions.append(ag_neglogp_exe) # should be zero because probability of action is 1.0 (but perhaps not because this is what the policy requests which is different than what is physically acted)
                            trainer.mbi_healths.append(0.0) # Agents are always terminated

                        # Time step the environment (which calls the world step)
                        observation_n, reward_n, done_n, _ = self.env.step(np.asarray([trainer.mbi_actions[-1] for trainer in self.group_trainer.agent_trainer_group]))
                        cur_joint_state = self.env.get_joint_state()

                        # Check that nothing has moved since all agents are terminated
                        # if not np.allclose(np.concatenate(cur_joint_state['state'],axis=0), np.concatenate(init_joint_state['state'],axis=0), atol=1e-5, rtol=1e-8):
                        # self.assertTrue(np.allclose(np.concatenate(cur_joint_state['state'],axis=0), np.concatenate(init_joint_state['state'],axis=0), atol=1e-4, rtol=1e-8))
                        for cjs, ijs in zip(np.concatenate(cur_joint_state['state'],axis=0), init_joint_state_arr):
                            self.assertAlmostEqual(cjs, ijs, places=6)


                        # record and check new rewards
                        # self.assertTrue(np.allclose(reward_n, init_reward, atol=1e-3, rtol=1e-8))
                        for agi, trainer in enumerate(self.group_trainer.agent_trainer_group):
                            self.assertAlmostEqual(reward_n[agi], init_reward[agi], places=6)
                            self.assertFalse(done_n[agi])
                            trainer.mbi_rewards.append(reward_n[agi])
                            trainer.mbi_dones.append(done_n[agi])
                    
                    # Handle final time step
                    self.group_trainer.record_joint_state(cur_joint_state)
                    for agi, trainer in enumerate(self.group_trainer.agent_trainer_group):
                        trainer.mbi_observations.append(observation_n[agi])
                        trainer.mbi_dones.append(True)

                    # self.group_trainer.update_group_policy(terminal=1)
                    episode_factual_values, episode_counterfactual_values = self.group_trainer.process_episode_value_centralization_and_credit_assignment()
                    self.group_trainer.process_episode_returns_and_store_group_training_batch(episode_factual_values, episode_counterfactual_values)
                    self.group_trainer.process_episode_clear_data()

                    # check that batch joint state and batch returns are discounted as expected
                    cur_return = 0.0
                    for tstep in range(self.test_episode_len+1):
                        self.assertAlmostEqual(self.group_trainer.batch_joint_state_stamped[(self.group_trainer.n_steps_per_episode+1)*ep + tstep][0], self.test_episode_len+1-tstep)
                        for bjssi, bjss in enumerate(self.group_trainer.batch_joint_state_stamped[(self.group_trainer.n_steps_per_episode+1)*ep + tstep][1:]):
                            self.assertAlmostEqual(bjss, init_joint_state_arr[bjssi])
                        self.assertAlmostEqual(self.group_trainer.batch_joint_returns[-tstep-1], cur_return, places=5)
                        cur_return = init_reward[0] + self.test_gamma*cur_return

                    # check that individuals' memories are properly cleared out
                    for trainer in self.group_trainer.agent_trainer_group:
                        self.assertTrue(len(trainer.mbi_observations) == 0)
                        self.assertTrue(len(trainer.mbi_actions) == 0)
                        self.assertTrue(len(trainer.mbi_rewards) == 0)
                        self.assertTrue(len(trainer.mbi_obs_values) == 0)
                        self.assertTrue(len(trainer.mbi_neglogp_actions) == 0)
                        self.assertTrue(len(trainer.mbi_dones) == 0)
                        self.assertTrue(len(trainer.mbi_returns) == 0)
                        self.assertTrue(len(trainer.mbi_credits) == 0)
                        self.assertTrue(len(trainer.mbi_factual_advantages) == 0)

                # after episodes per batch, update policy
                self.group_trainer.batch_credit_assignment()
                batch_loss_stats = self.group_trainer.execute_group_training()
                central_vf_loss.append(np.mean([bls[-2] for bls in batch_loss_stats]))
                central_vf_expvar.append(np.mean([bls[-1] for bls in batch_loss_stats]))
                # training_loss_stats += [[self.test_episode_len*self.test_n_episodes_per_batch*(train_iter+1)] + L for L in batch_loss_stats]
                self.assertEqual(len(self.group_trainer.batch_observations), 0)
                self.assertEqual(len(self.group_trainer.batch_joint_state_stamped), 0)
                self.assertEqual(len(self.group_trainer.batch_joint_returns), 0)
                self.assertEqual(len(self.group_trainer.batch_factual_values), 0)
                self.assertEqual(len(self.group_trainer.batch_counterfactual_values), 0)
                self.assertEqual(len(self.group_trainer.batch_actions), 0)
                self.assertEqual(len(self.group_trainer.batch_returns), 0)
                self.assertEqual(len(self.group_trainer.batch_dones), 0)
                self.assertEqual(len(self.group_trainer.batch_neglogp_actions), 0)
                self.assertEqual(len(self.group_trainer.batch_effective_returns), 0)

                print("iter {} | loss={:7.3E} | expln var={:7.3E}".format(
                        train_iter, 
                        central_vf_loss[-1],
                        central_vf_expvar[-1]
                        ))


class TestErgoPerimeter2VariableCase1(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_perimeter2_variable.Scenario(num_agents=4, num_hazards=0, identical_rewards=True, observation_type="direct")

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.landmarks[0].state.p_pos = np.zeros(self.world.dim_p)

    def test_ergo_perimeter2_variable_case1_systemic_reward_1(self):
        ''' ergo_perimeter2_variable: systemic reward for 4 agents arranged in square (0 hazard, identical rewards, direct observation)
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, -1.0])
        self.world.agents[1].state.p_pos = np.array([-1.0, 1.0])
        self.world.agents[2].state.p_pos = np.array([1.0, 1.0])
        self.world.agents[3].state.p_pos = np.array([1.0, -1.0])

        # calculate expected reward (peak values divide out to 1)
        Ae = 4.0
        Be = Point(self.world.landmarks[0].state.p_pos).buffer(ergo_perimeter2_variable._LANDMARK_SIZE).area
        expect_reward = Be/Ae # other term is expected to be zero for this case
        expect_reward /= 4.0    # normalize by number of agents

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, expect_reward)

    def test_ergo_perimeter2_variable_case1_systemic_reward_2(self):
        ''' ergo_perimeter2_variable: systemic reward for 4 collinear agents (0 hazard, identical rewards, direct observation)
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.5, 0.0])
        self.world.agents[3].state.p_pos = np.array([1.0, 0.0])

        reward = self.scenario.reward(agent=None, world=self.world, systemic_call=True)
        for r in reward:
            self.assertAlmostEqual(r, 0.0)

    def test_ergo_perimeter2_variable_case1_observations_1(self):
        ''' ergo_perimeter2_variable: observations of 4 agents around landmark (0 hazard, identical rewards, direct observation)
        Notes:
        '''

        self.world.agents[0].state.p_pos = np.array([-1.0, -1.0])
        self.world.agents[1].state.p_pos = np.array([-1.0, 1.0])
        self.world.agents[2].state.p_pos = np.array([1.0, 1.0])
        self.world.agents[3].state.p_pos = np.array([1.0, -1.0])

        self.world.agents[0].state.p_vel = np.zeros(2)
        self.world.agents[1].state.p_vel = np.zeros(2)
        self.world.agents[2].state.p_vel = np.zeros(2)
        self.world.agents[3].state.p_vel = np.zeros(2)


        pkv = ergo_perimeter2_variable._PEAK_REWARD
        lmr = ergo_perimeter2_variable._LANDMARK_SIZE
        obs_rew = pkv*(1.-2./(lmr**2))
        expect_obs0 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                -1., -1.,                # pos
                                obs_rew,                # landmark observation (sensed reward)
                                0, 0., 2., 0., 0., obs_rew, # agent 1 observation
                                0, 2., 2., 0., 0., obs_rew, # agent 2 observation
                                0, 2., 0., 0., 0., obs_rew, # agent 3 observation
                                ])
        expect_obs0 = np.concatenate([expect_obs0, np.zeros(24)])
        expect_obs1 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                -1., 1.,                # pos
                                obs_rew,                # landmark observation (sensed reward)
                                0, 0., -2., 0., 0., obs_rew, # agent 0 observation
                                0, 2., 0., 0., 0., obs_rew,  # agent 2 observation
                                0, 2., -2., 0., 0., obs_rew, # agent 3 observation
                                ])
        expect_obs1 = np.concatenate([expect_obs1, np.zeros(24)])
        expect_obs2 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                1., 1.,                 # pos
                                obs_rew,                # landmark observation (sensed reward)
                                0, -2., -2., 0., 0., obs_rew, # agent 0 observation
                                0, -2., 0., 0., 0., obs_rew,  # agent 1 observation
                                0, 0., -2., 0., 0., obs_rew,  # agent 3 observation
                                ])
        expect_obs2 = np.concatenate([expect_obs2, np.zeros(24)])
        expect_obs3 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                1., -1.,                # pos
                                obs_rew,                # landmark observation (sensed reward)
                                0, -2., 0., 0., 0., obs_rew,  # agent 0 observation
                                0, -2., 2., 0., 0., obs_rew,  # agent 1 observation
                                0, 0.,  2., 0., 0., obs_rew,  # agent 2 observation
                                ])
        expect_obs3 = np.concatenate([expect_obs3, np.zeros(24)])

        observation0 = self.scenario.observation(agent=self.world.agents[0], world=self.world)
        observation1 = self.scenario.observation(agent=self.world.agents[1], world=self.world)
        observation2 = self.scenario.observation(agent=self.world.agents[2], world=self.world)
        observation3 = self.scenario.observation(agent=self.world.agents[3], world=self.world)
        for i, _ in enumerate(observation0):
            self.assertAlmostEqual(observation0[i], expect_obs0[i])
            self.assertAlmostEqual(observation1[i], expect_obs1[i])
            self.assertAlmostEqual(observation2[i], expect_obs2[i])
            self.assertAlmostEqual(observation3[i], expect_obs3[i])


    def test_ergo_perimeter2_variable_case1_observations_2(self):
        ''' ergo_perimeter2_variable: observations with 4 agents aligned w/ landmark (0 hazard, identical rewards, direct observation)
        Notes:
        '''

        self.world.agents[0].state.p_pos = np.array([-1.5, 0.0])
        self.world.agents[1].state.p_pos = np.array([-1.0, 0.0])
        self.world.agents[2].state.p_pos = np.array([-0.5, 0.0])
        self.world.agents[3].state.p_pos = np.array([-0.0, 0.0])

        self.world.agents[0].state.p_vel = np.zeros(2)
        self.world.agents[1].state.p_vel = np.zeros(2)
        self.world.agents[2].state.p_vel = np.zeros(2)
        self.world.agents[3].state.p_vel = np.zeros(2)


        pkv = ergo_perimeter2_variable._PEAK_REWARD
        lmr = ergo_perimeter2_variable._LANDMARK_SIZE
        obs_rew0 = pkv*(1.-(1.5/lmr)**2)
        obs_rew1 = pkv*(1.-(1.0/lmr)**2)
        obs_rew2 = pkv*(1.-(0.5/lmr)**2)
        obs_rew3 = pkv
        expect_obs0 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                -1.5, 0.,                # pos
                                obs_rew0,                # landmark observation (sensed reward)
                                0, 0.5, 0., 0., 0., obs_rew1, # agent 1 observation
                                0, 1.0, 0., 0., 0., obs_rew2, # agent 2 observation
                                0, 1.5, 0., 0., 0., obs_rew3, # agent 3 observation
                                ])
        expect_obs0 = np.concatenate([expect_obs0, np.zeros(24)])
        expect_obs1 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                -1.0, 0.,                # pos
                                obs_rew1,                # landmark observation (sensed reward)
                                0, -0.5, 0., 0., 0., obs_rew0, # agent 0 observation
                                0, 0.5, 0., 0., 0., obs_rew2,  # agent 2 observation
                                0, 1.0, 0., 0., 0., obs_rew3, # agent 3 observation
                                ])
        expect_obs1 = np.concatenate([expect_obs1, np.zeros(24)])
        expect_obs2 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                -0.5, 0.,                 # pos
                                obs_rew2,                # landmark observation (sensed reward)
                                0, -1.0, 0., 0., 0., obs_rew0, # agent 0 observation
                                0, -0.5, 0., 0., 0., obs_rew1,  # agent 1 observation
                                0, 0.5, 0., 0., 0., obs_rew3,  # agent 3 observation
                                ])
        expect_obs2 = np.concatenate([expect_obs2, np.zeros(24)])
        expect_obs3 = np.array([0,                      # terminated
                                0., 0.,                 # vel
                                0., 0.,                # pos
                                obs_rew3,                # landmark observation (sensed reward)
                                0, -1.5, 0., 0., 0., obs_rew0,  # agent 0 observation
                                0, -1.0, 0., 0., 0., obs_rew1,  # agent 1 observation
                                0, -0.5, 0., 0., 0., obs_rew2,  # agent 2 observation
                                ])
        expect_obs3 = np.concatenate([expect_obs3, np.zeros(24)])

        observation0 = self.scenario.observation(agent=self.world.agents[0], world=self.world)
        observation1 = self.scenario.observation(agent=self.world.agents[1], world=self.world)
        observation2 = self.scenario.observation(agent=self.world.agents[2], world=self.world)
        observation3 = self.scenario.observation(agent=self.world.agents[3], world=self.world)
        for i, _ in enumerate(observation0):
            self.assertAlmostEqual(observation0[i], expect_obs0[i])
            self.assertAlmostEqual(observation1[i], expect_obs1[i])
            self.assertAlmostEqual(observation2[i], expect_obs2[i])
            self.assertAlmostEqual(observation3[i], expect_obs3[i])

class TestErgoPerimeter2VariableCase2(unittest.TestCase):

    def setUp(self):

        # create Scenario object and modify to test conditions
        self.scenario = ergo_perimeter2_variable.Scenario(num_agents=5, num_hazards=0, identical_rewards=True, observation_type="histogram")

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        self.world.landmarks[0].state.p_pos = np.zeros(self.world.dim_p)

    def test_ergo_perimeter2_variable_case2_observations_1(self):
        ''' ergo_graph_variable: test histogram observations with 5 agents (1 hazard, shared rewards, histogram observation)
        Notes:
         - Assume 4 radial and 8 angular bins with a max obs distance of 1.0
         - Formatted to work for either linear or logspaced bins
        '''

        self.world.agents[0].state.p_pos = np.array([-2.0, 0.0])
        self.world.agents[1].state.p_pos = np.array([-0.1, 0.0])
        self.world.agents[2].state.p_pos = np.array([0.0, 0.0])
        self.world.agents[2].state.p_vel = np.array([0.0, 0.0])
        self.world.agents[3].state.p_pos = np.array([0.1, 0.0])
        self.world.agents[4].state.p_pos = np.array([2.0, 0.0])


        expected_agent_histogram = np.array([[0]*8]*4)
        expected_agent_histogram[3][4] = 1
        expected_agent_histogram[0][4] = 1
        expected_agent_histogram[0][0] = 1
        expected_agent_histogram[3][0] = 1
        expected_agent_histogram = [val for sublist in expected_agent_histogram for val in sublist]

        pkv = ergo_perimeter2_variable._PEAK_REWARD
        lmr = ergo_perimeter2_variable._LANDMARK_SIZE
        obs_rew0 = pkv*(1.-(2.0/lmr)**2)
        obs_rew1 = pkv*(1.-(0.1/lmr)**2)
        obs_rew2 = pkv
        obs_rew3 = pkv*(1.-(0.1/lmr)**2)
        obs_rew4 = pkv*(1.-(2.0/lmr)**2)
        expected_reward_histogram = np.array([[0.0]*8]*4)
        expected_reward_histogram[3][4] = obs_rew0
        expected_reward_histogram[0][4] = obs_rew1
        expected_reward_histogram[0][0] = obs_rew3
        expected_reward_histogram[3][0] = obs_rew4
        expected_reward_histogram = [val for sublist in expected_reward_histogram for val in sublist]

        expected_observation = ([0] + # agent terminated
                                [0,0,0,0] + # vel and pos
                                [pkv] + # local landmark observation
                                expected_agent_histogram + # agent histogram
                                expected_reward_histogram + # reward histogram
                                [0.0]*3*2 + # observed failures
                                [0]*76 # previous observation
                                )
        
        observation = self.scenario.observation(agent=self.world.agents[2], world=self.world)
        for i, obs in enumerate(observation):
            self.assertAlmostEqual(obs, expected_observation[i])
if __name__ == '__main__':
    unittest.main()
