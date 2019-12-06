#!/usr/bin/env python

# suite of unit, integration, system, and/or acceptance tests for train.py. 
# To run test, simply call:
#
#   in a shell with conda environment ergo_particle_gym activated:
#   nosetests test_train.py
#
#   in ipython:
#   run test_train.py

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from numpy.random import rand
from collections import namedtuple
from train import get_trainer_actions, OrderingException, parse_args
from rl_algorithms.mclearning import ScenarioHeuristicGroupTrainer
import particle_environments.mager.scenarios.ergo_graph_large as ergo_graph_large
import rl_algorithms.baselines.baselines.common.tf_util as U

# from particle_environments.mager.world import MortalAgent


class TestParseArgsCase1(unittest.TestCase):
    ''' test parse_arg function, particularly boolean parsing because it's non-standard
    '''

    def setUp(self):
        pass

    def test_default_arguments_1(self):
        ''' train.py:parse_args: check that default arguments match expectations '''
        arglist = parse_args("")
        self.assertTrue(arglist.environment=="MultiAgentEnv")
        self.assertTrue(arglist.scenario=="simple")
        self.assertTrue(arglist.max_episode_len==50)
        self.assertTrue(arglist.num_episodes==60000)
        self.assertTrue(arglist.num_adversaries==0)
        self.assertTrue(arglist.good_policy=="maddpg")
        self.assertTrue(arglist.adv_policy=="maddpg")
        self.assertTrue(arglist.variable_num_agents==4)
        self.assertTrue(arglist.variable_num_hazards==0)
        self.assertTrue(arglist.variable_local_rewards==False)
        self.assertTrue(arglist.variable_observation_type=="direct")
        self.assertTrue(arglist.training_algorithm=="MADDPGAgentTrainer")
        self.assertTrue(arglist.learning_rate==1e-2)
        self.assertTrue(arglist.variable_learning_rate==False)
        self.assertTrue(arglist.learning_rate_min==1e-5)
        self.assertTrue(arglist.learning_rate_period==10)
        self.assertTrue(arglist.entropy_coef==0.0)
        self.assertTrue(arglist.value_coef==0.5)
        self.assertTrue(arglist.gamma==0.95)
        self.assertTrue(arglist.batch_size==1024)
        self.assertTrue(arglist.num_layers==2)
        self.assertTrue(arglist.num_units==64)
        self.assertTrue(arglist.activation=="relu")
        self.assertTrue(arglist.cliprange==0.2)
        self.assertTrue(arglist.critic_type=="distributed_local_observations")
        self.assertTrue(arglist.num_minibatches==4)
        self.assertTrue(arglist.num_opt_epochs==4)
        self.assertTrue(arglist.experiment_name=='default')
        self.assertTrue(arglist.save_dir=="./experiments/default/policy/")
        self.assertTrue(arglist.save_rate==1000)
        self.assertTrue(arglist.load_dir=="")

    def test_boolean_arguments_case_1(self):
        ''' train.py:parse_args: check that booleans are correctly parsed '''
        arglist = parse_args("--variable-local-rewards".split())
        self.assertTrue(arglist.variable_local_rewards)
        arglist = parse_args("--variable-local-rewards True".split())
        self.assertTrue(arglist.variable_local_rewards)
        arglist = parse_args("--variable-local-rewards 1".split())
        self.assertTrue(arglist.variable_local_rewards)
        arglist = parse_args("--variable-local-rewards y".split())
        self.assertTrue(arglist.variable_local_rewards)

        arglist = parse_args("")
        self.assertFalse(arglist.variable_local_rewards)
        arglist = parse_args("--variable-local-rewards False".split())
        self.assertFalse(arglist.variable_local_rewards)
        arglist = parse_args("--variable-local-rewards 0".split())
        self.assertFalse(arglist.variable_local_rewards)
        arglist = parse_args("--variable-local-rewards n".split())
        self.assertFalse(arglist.variable_local_rewards)

        arglist = parse_args("--variable-learning-rate".split())
        self.assertTrue(arglist.variable_learning_rate)
        arglist = parse_args("--variable-learning-rate True".split())
        self.assertTrue(arglist.variable_learning_rate)
        arglist = parse_args("--variable-learning-rate 1".split())
        self.assertTrue(arglist.variable_learning_rate)
        arglist = parse_args("--variable-learning-rate y".split())
        self.assertTrue(arglist.variable_learning_rate)

        arglist = parse_args("")
        self.assertFalse(arglist.variable_learning_rate)
        arglist = parse_args("--variable-learning-rate False".split())
        self.assertFalse(arglist.variable_learning_rate)
        arglist = parse_args("--variable-learning-rate 0".split())
        self.assertFalse(arglist.variable_learning_rate)
        arglist = parse_args("--variable-learning-rate n".split())
        self.assertFalse(arglist.variable_learning_rate)


class TestGetTrainerActionsCase1(unittest.TestCase):
    ''' test get_trainer_actions from train.py
    '''

    def setUp(self):
        # self.agents = []
        # self.trainers = []
        # self.observations = []
        Trainer = namedtuple('Trainer', ['name', 'action'])
        Agent = namedtuple('Agent', ['terminated'])
        act_func = lambda obs: [o for o in obs]
        self.trainers = [Trainer('agent_0', act_func), 
                    Trainer('agent_1', act_func), 
                    Trainer('agent_2', act_func)]
        self.agents = [Agent(False), Agent(False), Agent(False)]
        self.observations = [[0, 1], [1, 2], [2, 3]]

    def test_action_output_case_1(self):
        ''' train.py: test feedthrough action output list'''
        action_n, _, _, _ = get_trainer_actions(agents=self.agents, trainers=self.trainers, observations=self.observations)
        self.assertTrue(action_n == self.observations)

    def test_ordering_assertion_case_1(self):
        ''' train.py: test exception for disorder trainers'''
        t = self.trainers[0]
        self.trainers[0] = self.trainers[1]
        self.trainers[1] = t
        with self.assertRaises(OrderingException):
            get_trainer_actions(agents=self.agents, trainers=self.trainers, observations=self.observations)

    def test_length_assertion_case_1(self):
        ''' train.py: test length missalignment '''
        self.trainers.pop()
        with self.assertRaises(AssertionError):
            get_trainer_actions(agents=self.agents, trainers=self.trainers, observations=self.observations)


class TestScenarioHeuristicGroupTrainerCase1(unittest.TestCase):

    def setUp(self):

        MCAgent = namedtuple('MCAgent', ['behavior_params', 'cumulative_reward'])
        agent_group = []
        n_agents = 10
        for i in range(n_agents):
            agent_group.append(MCAgent({'bp1':1.1, 'bp2':2.2, 'bp3':3.3}, float(i)))

        init_group_policy = {'bp1':{'clambda':rand()+1.0, 'ctheta':(0.1, 0.0)}, 
                            'bp2':{'clambda':rand()+1.0, 'ctheta':(0.2, 0.0)},
                            'bp3':{'clambda':rand()+1.0, 'ctheta':(0.3, 0.0)}}

        with U.single_threaded_session():
            self.mc_group_trainer = ScenarioHeuristicGroupTrainer(
                agent_trainer_group=agent_group, 
                init_group_policy=init_group_policy, 
                n_episodes_per_batch = np.random.randint(101) + 21,
                n_elite=20)

    def test_monte_carlo_trainer_case1_setup(self):
        ''' ScenarioHeuristicGroupTrainer: case1 setup test
        '''
        self.assertEqual(len(self.mc_group_trainer.agent_trainer_group), 10)
        self.assertEqual(len(self.mc_group_trainer.policy_batch), 0)

    def test_monte_carlo_trainer_case1_sample_group_policy_1(self):
        ''' ScenarioHeuristicGroupTrainer: case1 sample policy w no variance
        '''
        self.assertNotAlmostEqual(self.mc_group_trainer.group_policy['bp1']['clambda'], 0.1)
        self.assertNotAlmostEqual(self.mc_group_trainer.group_policy['bp2']['clambda'], 0.2)
        self.assertNotAlmostEqual(self.mc_group_trainer.group_policy['bp3']['clambda'], 0.3)

        self.mc_group_trainer.sample_group_policy()
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp1']['clambda'], 0.1)
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp2']['clambda'], 0.2)
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp3']['clambda'], 0.3)

    def test_monte_carlo_trainer_case1_evaluate_group_policy_1(self):
        ''' ScenarioHeuristicGroupTrainer: case1 evaluate group policy with reward equal agent index
        '''
        expected_policy_value = sum(np.linspace(0,9,10))
        policy_value = self.mc_group_trainer.evaluate_group_policy()
        self.assertAlmostEqual(expected_policy_value, policy_value)

    def test_monte_carlo_trainer_case1_select_elite_policies_1(self):
        ''' ScenarioHeuristicGroupTrainer: case1 select elite policies from uniform batch
        '''

        # generate random values for batch of uniform policies
        policy_value = rand()*20 - 10
        group_policy = {    'bp1':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)}, 
                            'bp2':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)},
                            'bp3':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)}}

        # create a history of recent policies
        self.mc_group_trainer.policy_batch = []
        for i in range(100):
            self.mc_group_trainer.policy_batch.append((policy_value, group_policy))

        elite_clambda = self.mc_group_trainer.select_elite_policies()
        self.assertEqual(len(elite_clambda), 3)
        self.assertEqual(len(elite_clambda['bp1']), self.mc_group_trainer.n_elite)
        self.assertEqual(len(elite_clambda['bp2']), self.mc_group_trainer.n_elite)
        self.assertEqual(len(elite_clambda['bp3']), self.mc_group_trainer.n_elite)
        for i in range(self.mc_group_trainer.n_elite):
            self.assertAlmostEqual(elite_clambda['bp1'][i], group_policy['bp1']['clambda'])
            self.assertAlmostEqual(elite_clambda['bp2'][i], group_policy['bp2']['clambda'])
            self.assertAlmostEqual(elite_clambda['bp3'][i], group_policy['bp3']['clambda'])

    def test_monte_carlo_trainer_case1_select_elite_policies_2(self):
        ''' ScenarioHeuristicGroupTrainer: case1 select elite policies from nonuniform batch
        '''

        # generate random values for batch of uniform policies
        low_policy_value = rand()*20 - 10
        low_group_policy = {    'bp1':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)}, 
                            'bp2':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)},
                            'bp3':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)}}

        high_policy_value = rand()*20 + 100
        high_group_policy = {    'bp1':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)}, 
                            'bp2':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)},
                            'bp3':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)}}

        # create a history of recent policies
        self.mc_group_trainer.policy_batch = []
        n_high = 0
        n_low = 0
        n_high_quota = 50
        while n_low + n_high < 100 or n_high < n_high_quota:
            if np.random.randint(2) == 1:
                self.mc_group_trainer.policy_batch.append((low_policy_value, low_group_policy))
                n_low += 1
            else:
                self.mc_group_trainer.policy_batch.append((high_policy_value, high_group_policy))
                n_high += 1


        elite_clambda = self.mc_group_trainer.select_elite_policies()
        self.assertEqual(len(elite_clambda), 3)
        self.assertEqual(len(elite_clambda['bp1']), self.mc_group_trainer.n_elite)
        self.assertEqual(len(elite_clambda['bp2']), self.mc_group_trainer.n_elite)
        self.assertEqual(len(elite_clambda['bp3']), self.mc_group_trainer.n_elite)
        for i in range(self.mc_group_trainer.n_elite):
            self.assertAlmostEqual(elite_clambda['bp1'][i], high_group_policy['bp1']['clambda'])
            self.assertAlmostEqual(elite_clambda['bp2'][i], high_group_policy['bp2']['clambda'])
            self.assertAlmostEqual(elite_clambda['bp3'][i], high_group_policy['bp3']['clambda'])

    def test_monte_carlo_trainer_case1_update_group_policy_distribution_1(self):
        ''' ScenarioHeuristicGroupTrainer: case1 update group policy distribution w quasi-uniform policy batch
        '''

        # generate random values for batch of uniform policies
        low_policy_value = rand()*20 - 10
        low_group_policy = {    'bp1':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, rand())}, 
                            'bp2':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, rand())},
                            'bp3':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, rand())}}

        high_policy_value = rand()*20 + 100
        high_group_policy = {    'bp1':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)}, 
                            'bp2':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)},
                            'bp3':{'clambda':rand()*2-1, 'ctheta':(rand()*2-1, 0.0)}}

        # create a history of recent policies
        self.mc_group_trainer.policy_batch = []
        n_high = 0
        n_low = 0
        n_high_quota = 50
        while n_low + n_high < 100 or n_high < n_high_quota:
            if np.random.randint(2) == 1:
                self.mc_group_trainer.policy_batch.append((low_policy_value, low_group_policy))
                n_low += 1
            else:
                self.mc_group_trainer.policy_batch.append((high_policy_value, high_group_policy))
                n_high += 1

        # update the group policy distribution (leaves clambda values as None)
        self.mc_group_trainer.update_group_policy_distribution()
        self.assertTrue(self.mc_group_trainer.group_policy['bp1']['clambda'] is None)
        self.assertTrue(self.mc_group_trainer.group_policy['bp2']['clambda'] is None)
        self.assertTrue(self.mc_group_trainer.group_policy['bp3']['clambda'] is None)
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp1']['ctheta'][0], high_group_policy['bp1']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp2']['ctheta'][0], high_group_policy['bp2']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp3']['ctheta'][0], high_group_policy['bp3']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp1']['ctheta'][1], 0.0)
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp2']['ctheta'][1], 0.0)
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp3']['ctheta'][1], 0.0)


        self.mc_group_trainer.sample_group_policy()
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp1']['clambda'], high_group_policy['bp1']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp2']['clambda'], high_group_policy['bp2']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bp3']['clambda'], high_group_policy['bp3']['clambda'])

class TestMonteCarloErgoGraphLargeCase1(unittest.TestCase):
    ''' integration test between ergo_graph_large and monte carlo group trainer
    '''

    def setUp(self):
    
        # create Scenario object and modify to test conditions
        self.scenario = ergo_graph_large.Scenario()
        self.scenario.num_agents = 5

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        obs_shape_n = [(np.random.randint(100), np.random.randint(100)) for i in range(self.scenario.num_agents)]
        agent_trainer_group = []
        for i, a in enumerate(self.world.agents):
            agent_trainer = ergo_graph_large.ScenarioHeuristicComputer(name=a.name, model=None, obs_shape_n=obs_shape_n, act_space_n=None, agent_index=i, args=None)
            agent_trainer.behavior_params = dict()
            agent_trainer.behavior_params['foo1'] = rand()*20 - 10
            agent_trainer.behavior_params['foo2'] = rand()*20 - 10
            agent_trainer.behavior_params['bar1'] = rand()*20 - 10
            agent_trainer.behavior_params['bar2'] = rand()*20 - 10
            agent_trainer_group.append(agent_trainer)

        init_group_policy = {'foo1':{'clambda':rand()+1.0, 'ctheta':(0.1, 0.0)}, 
                            'foo2':{'clambda':rand()+1.0, 'ctheta':(0.1, 0.0)},
                            'bar1':{'clambda':rand()+1.0, 'ctheta':(0.1, 0.0)},
                            'bar2':{'clambda':rand()+1.0, 'ctheta':(0.1, 0.0)}}

        with U.single_threaded_session():
            self.mc_group_trainer = ScenarioHeuristicGroupTrainer(
                agent_trainer_group=agent_trainer_group, 
                init_group_policy=init_group_policy, 
                n_episodes_per_batch = 107,
                n_elite=23)

    def test_monte_carlo_ergo_graph_large_case1_setup(self):
        ''' ScenarioHeuristicGroupTrainer: case1 ergo_graph_large + monte carlo group trainer setup
        '''

        self.assertEqual(len(self.mc_group_trainer.policy_batch), 0)

        # check that physical agents and agent trainers names align
        for i, a in enumerate(self.world.agents):
            self.assertTrue(a.name == self.mc_group_trainer.agent_trainer_group[i].name)

    def test_monte_carlo_ergo_graph_large_case1_update_group_policy_1(self):
        ''' ScenarioHeuristicGroupTrainer: case1 update group policy based on quasi-uniform batch
        '''

        # create a history of recent policies
        self.mc_group_trainer.policy_batch = []
        n_high = 0
        n_low = 0
        high_group_policy = {'foo1':{'clambda':1.48446622, 'ctheta':(0.1, 0.0)}, 
                            'foo2':{'clambda':3.18891587, 'ctheta':(0.1, 0.0)},
                            'bar1':{'clambda':9.31067831, 'ctheta':(0.1, 0.0)},
                            'bar2':{'clambda':-14.60020169, 'ctheta':(0.1, 0.0)}}

        while ((n_low + n_high < self.mc_group_trainer.n_episodes_per_batch) or 
            (n_high < self.mc_group_trainer.n_elite+5)):

            if np.random.randint(2) == 1:
                low_policy_value = np.random.normal(0.0, 1.0)

                # clamba randomized since it shouldn't affect outcome since value
                # almost guaranteed to be much smaller than high value
                low_group_policy = {'foo1':{'clambda':np.random.normal(0, 2.0), 'ctheta':(0.1, 0.0)}, 
                            'foo2':{'clambda':np.random.normal(0, 2.0), 'ctheta':(0.1, 0.0)},
                            'bar1':{'clambda':np.random.normal(0, 2.0), 'ctheta':(0.1, 0.0)},
                            'bar2':{'clambda':np.random.normal(0, 2.0), 'ctheta':(0.1, 0.0)}}
                self.mc_group_trainer.policy_batch.append((low_policy_value, low_group_policy))
                n_low += 1

            else:
                high_policy_value = np.random.normal(100.0, 2.0)
                self.mc_group_trainer.policy_batch.append((high_policy_value, high_group_policy))
                n_high += 1


        # update the group policy distribution (leaves clambda values as None)
        self.mc_group_trainer.update_group_policy(terminal=True)

        self.assertAlmostEqual(self.mc_group_trainer.group_policy['foo1']['clambda'], high_group_policy['foo1']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['foo2']['clambda'], high_group_policy['foo2']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bar1']['clambda'], high_group_policy['bar1']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bar2']['clambda'], high_group_policy['bar2']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['foo1']['ctheta'][0], high_group_policy['foo1']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['foo2']['ctheta'][0], high_group_policy['foo2']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bar1']['ctheta'][0], high_group_policy['bar1']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bar2']['ctheta'][0], high_group_policy['bar2']['clambda'])
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['foo1']['ctheta'][1], 0.0)
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['foo2']['ctheta'][1], 0.0)
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bar1']['ctheta'][1], 0.0)
        self.assertAlmostEqual(self.mc_group_trainer.group_policy['bar2']['ctheta'][1], 0.0)


        for agtr in self.mc_group_trainer.agent_trainer_group:
            self.assertAlmostEqual(agtr.behavior_params['foo1'], high_group_policy['foo1']['clambda'])
            self.assertAlmostEqual(agtr.behavior_params['foo2'], high_group_policy['foo2']['clambda'])
            self.assertAlmostEqual(agtr.behavior_params['bar1'], high_group_policy['bar1']['clambda'])
            self.assertAlmostEqual(agtr.behavior_params['bar2'], high_group_policy['bar2']['clambda'])

    # def test_monte_carlo_ergo_graph_large_case1_update_group_policy_2(self):
    #     ''' ScenarioHeuristicGroupTrainerCase1: check policy distribution doesn't update until batch full
    #     '''

    #     # create a history of recent policies
    #     self.mc_group_trainer.policy_batch = []
    #     n_high = 0
    #     n_low = 0
    #     train_step = 0
    #     high_group_policy = {'foo1':{'clambda':-0.29776537, 'ctheta':(-0.29776537, 0.0)}, 
    #                         'foo2':{'clambda':0.78748377, 'ctheta':(0.78748377, 0.0)},
    #                         'bar1':{'clambda':0.76835996, 'ctheta':(0.76835996, 0.0)},
    #                         'bar2':{'clambda':0.13583169, 'ctheta':(0.13583169, 0.0)}}

    #     # assign high group policy and cumulative reward to all agents
    #     for agtr in self.mc_group_trainer.agent_trainer_group:
    #         agent_trainer.behavior_params['foo1'] = high_group_policy['foo1']['clambda']
    #         agent_trainer.behavior_params['foo2'] = high_group_policy['foo2']['clambda']
    #         agent_trainer.behavior_params['bar1'] = high_group_policy['bar1']['clambda']
    #         agent_trainer.behavior_params['bar2'] = high_group_policy['bar2']['clambda']


    #     # run training steps
    #     for i in range(self.mc_group_trainer.n_episodes_per_batch):
    #         self.mc_group_trainer.update_group_policy( 
    #             terminal=(i==self.mc_group_trainer.n_episodes_per_batch))
    #         self.assertAlmostEqual(self.mc_group_trainer.group_policy['foo1']['ctheta'][0], 0.1)
    #         self.assertAlmostEqual(self.mc_group_trainer.group_policy['foo2']['ctheta'][0], 0.1)
    #         self.assertAlmostEqual(self.mc_group_trainer.group_policy['bar1']['ctheta'][0], 0.1)
    #         self.assertAlmostEqual(self.mc_group_trainer.group_policy['bar2']['ctheta'][0], 0.1)


class TestMonteCarloErgoGraphLargeSystemTest1(unittest.TestCase):
    ''' system test monte carlo converges to predefined optimal policy
    '''

    def setUp(self):
    
        # create Scenario object and modify to test conditions
        self.scenario = ergo_graph_large.Scenario()
        self.scenario.num_agents = 18

        # create world and modify to test conditions
        self.world = self.scenario.make_world()
        obs_shape_n = [(42,) for i in range(self.scenario.num_agents)]

        # create agent trainers
        agent_trainer_group = []
        for i, a in enumerate(self.world.agents):
            agent_trainer = ergo_graph_large.ScenarioHeuristicComputer(name=a.name, model=None, obs_shape_n=obs_shape_n, act_space_n=None, agent_index=i, args=None)
            agent_trainer.behavior_params = dict()
            agent_trainer.behavior_params['E1WT8'] = 0.0
            agent_trainer.behavior_params['9CFB0'] = 0.0
            agent_trainer.behavior_params['N2AEU'] = 0.0
            agent_trainer.behavior_params['J6176'] = 0.0
            agent_trainer.behavior_params['GK63E'] = 0.0
            agent_trainer.behavior_params['GJ102'] = 0.0
            agent_trainer.behavior_params['8FEGV'] = 0.0
            agent_trainer.behavior_params['F1DZ1'] = 0.0
            agent_trainer.behavior_params['L8W3V'] = 0.0
            agent_trainer_group.append(agent_trainer)


        # initial group policy
        init_group_policy = {   'E1WT8':{'clambda':0.0, 'ctheta':(0.0, 20.0)},
                                '9CFB0':{'clambda':0.0, 'ctheta':(0.0, 20.0)},
                                'N2AEU':{'clambda':0.0, 'ctheta':(0.0, 20.0)},
                                'J6176':{'clambda':0.0, 'ctheta':(0.0, 20.0)},
                                'GK63E':{'clambda':0.0, 'ctheta':(0.0, 20.0)},
                                'GJ102':{'clambda':0.0, 'ctheta':(0.0, 20.0)},
                                '8FEGV':{'clambda':0.0, 'ctheta':(0.0, 20.0)},
                                'F1DZ1':{'clambda':0.0, 'ctheta':(0.0, 20.0)},
                                'L8W3V':{'clambda':0.0, 'ctheta':(0.0, 20.0)}}


        with U.single_threaded_session():
            self.mc_group_trainer = ScenarioHeuristicGroupTrainer(
                agent_trainer_group=agent_trainer_group, 
                init_group_policy=init_group_policy, 
                n_episodes_per_batch = 107,
                n_elite=23)


    def test_monte_carlo_ergo_graph_large_system_test_1(self):
        ''' ScenarioHeuristicGroupTrainer: system test convergence to predefined optimal policy
        '''

        # set predefined opitmal policy
        optimal_policy = dict()
        optimal_policy['E1WT8'] = 73.02350724757949
        optimal_policy['9CFB0'] = 48.96725526428631
        optimal_policy['N2AEU'] = 1.035312157323121
        optimal_policy['J6176'] = -24.53464424927825
        optimal_policy['GK63E'] = 74.53637829562955
        optimal_policy['GJ102'] = 63.7730254457293
        optimal_policy['8FEGV'] = 17.972315967456986
        optimal_policy['F1DZ1'] = 8.133858471873188
        optimal_policy['L8W3V'] = -34.34657297476912


        # execute training steps
        total_trials = 2000
        group_policy_value_history = []
        for trial in range(total_trials):

            # get reward for each agent simply based on distance to optimal policy
            expected_cumulative_reward = sum([-abs(self.mc_group_trainer.group_policy[k]['clambda'] - optimal_policy[k]) for k in optimal_policy])
            for agtr in self.mc_group_trainer.agent_trainer_group:
                for k in agtr.behavior_params:
                    agtr.cumulative_reward += -abs(agtr.behavior_params[k] - optimal_policy[k])
                self.assertAlmostEqual(agtr.cumulative_reward, expected_cumulative_reward)

            # evaluate group policy
            group_policy_value = self.mc_group_trainer.evaluate_group_policy()
            self.assertAlmostEqual(expected_cumulative_reward*self.scenario.num_agents, group_policy_value)
            group_policy_value_history.append(group_policy_value)

            # sample new policy or/and update distribution
            self.mc_group_trainer.update_group_policy(terminal=True)


        # check that the policy value has increased and variance decreased
        self.assertGreater(np.mean(group_policy_value_history[-100:]), np.mean(group_policy_value_history[:100]))
        self.assertLess(np.std(group_policy_value_history[-100:]), np.std(group_policy_value_history[:100]))


if __name__ == '__main__':
    unittest.main()
