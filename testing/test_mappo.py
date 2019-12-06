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
import tensorflow as tf
from gym import spaces
from numpy.random import rand
from train import OrderingException, DeepMLP
from collections import namedtuple
from rl_algorithms.mappo import PPOAgentComputer, PPOGroupTrainer, UpdateException, redistributed_softmax, central_critic_network
import rl_algorithms.maddpg.maddpg.common.tf_util as U
from rl_algorithms.baselines.baselines.common import explained_variance

# from particle_environments.mager.world import MortalAgent

_DEBUG = False

if _DEBUG:
    import matplotlib.pyplot as plt

class TestPPOAgentComputer1(unittest.TestCase):
    ''' test PPOAgentComputer class from mappo.py
    '''

    def setUp(self):
        pass

    def test_process_individual_agent_episode_returns_and_advantages_1(self):
        ''' one-step return and advantage calculation with float rewards'''

        Model = namedtuple('Model', ['value'])
        Args = namedtuple('Args', ['max_episode_len', 'gamma'])

        value_func = lambda obs, M: sum(obs)
        model = Model(value_func)
        gamma = 1.0
        args = Args(1, gamma)
        ppo_agent = PPOAgentComputer(name="ppo_agent_0", model=model,
            obs_shape_n=None, act_space_n=None, agent_index=0, args=args, local_q_func=None, lam=1.0)

        ppo_agent.mbi_observations = [np.array([ 0.52141883, -0.66102998]), np.array([-0.39118867, -0.08772333])]
        ppo_agent.mbi_rewards = [0.0]
        ppo_agent.mbi_obs_values = [-0.13961115000000002]
        ppo_agent.mbi_dones = [False, True]
        ppo_agent.mbi_actions = [np.random.uniform(-1,1,2)]
        ppo_agent.mbi_neglogp_actions = [np.random.uniform(0,1)]
        ppo_agent.mbi_healths = [1.0]
        ppo_agent.process_individual_agent_episode_returns_and_advantages(factual_values=None, counterfactual_values=None)

        # check return and advantage
        self.assertAlmostEqual(ppo_agent.mbi_returns[0], 0.0)
        self.assertAlmostEqual(ppo_agent.mbi_factual_advantages[0], 0.13961115000000002)

    def test_process_individual_agent_episode_returns_and_advantages_3(self):
        '''mappo: two-step return and advantage calculation'''

        Model = namedtuple('Model', ['value'])
        Args = namedtuple('Args', ['max_episode_len', 'gamma'])

        value_func = lambda obs, M: np.mean(obs)
        model = Model(value_func)
        gamma = 0.9627477525841408
        lam = 0.9447698026141256
        args = Args(2, gamma)
        ppo_agent = PPOAgentComputer(name="ppo_agent_0", model=model,
            obs_shape_n=None, act_space_n=None, agent_index=0, args=args, local_q_func=None, lam=lam)

        ppo_agent.mbi_observations = [np.array([ 0.4660721 , -3.39177499]), 
                                      np.array([-4.13104788, -4.52925146]), 
                                      np.array([ 3.16713255, -2.30391816])]
        ppo_agent.mbi_rewards = [-0.71486004, -1.92588795]
        ppo_agent.mbi_obs_values = [value_func(ppo_agent.mbi_observations[0], M=None),
                                value_func(ppo_agent.mbi_observations[1], M=None)]
        ppo_agent.mbi_dones = [False, False, True]
        ppo_agent.mbi_actions = [np.random.uniform(-1,1,2), np.random.uniform(-1,1,2)]
        ppo_agent.mbi_neglogp_actions = [np.random.uniform(0,1), np.random.uniform(0,1)]
        ppo_agent.mbi_healths = [1.0, 1.0]

        # calculate expected values
        delta_1 = -1.92588795 - np.mean([-4.13104788, -4.52925146])
        exp_returns_1 = -1.92588795
        exp_advantages_1 = delta_1
        delta_0 = -0.71486004 + gamma*np.mean([-4.13104788, -4.52925146]) - np.mean([ 0.4660721 , -3.39177499])
        exp_advantages_0 = delta_0 + gamma*lam*delta_1
        exp_returns_0 = exp_advantages_0 + np.mean([ 0.4660721 , -3.39177499])

        # check return and advantage
        ppo_agent.process_individual_agent_episode_returns_and_advantages(factual_values=None, counterfactual_values=None)
        self.assertAlmostEqual(ppo_agent.mbi_returns[1], exp_returns_1,places=5)
        self.assertAlmostEqual(ppo_agent.mbi_factual_advantages[1], exp_advantages_1,places=5)
        self.assertAlmostEqual(ppo_agent.mbi_returns[0], exp_returns_0,places=5)
        self.assertAlmostEqual(ppo_agent.mbi_factual_advantages[0], exp_advantages_0,places=5)

    def test_process_individual_agent_episode_returns_and_advantages_4(self):
        '''mappo: error handling for multi-step batch with inconsistent dones'''

        Model = namedtuple('Model', ['value'])
        Args = namedtuple('Args', ['max_episode_len', 'gamma'])

        value_func = lambda obs, M: np.mean(obs)
        model = Model(value_func)
        gamma = 0.9627477525841408
        lam = 0.9447698026141256
        args = Args(2, gamma)
        ppo_agent = PPOAgentComputer(name="ppo_agent_0", model=model,
            obs_shape_n=None, act_space_n=None, agent_index=0, args=args, local_q_func=None, lam=lam)

        ppo_agent.mbi_observations = [np.array([ 0.4660721 , -3.39177499]), 
                                      np.array([-4.13104788, -4.52925146]), 
                                      np.array([ 3.16713255, -2.30391816])]
        ppo_agent.mbi_rewards = [-0.71486004, -1.92588795]
        ppo_agent.mbi_obs_values = [value_func(ppo_agent.mbi_observations[0], M=None),
                                value_func(ppo_agent.mbi_observations[1], M=None)]
        ppo_agent.mbi_dones = [False, True, False]
        ppo_agent.mbi_actions = [np.random.uniform(-1,1,2), np.random.uniform(-1,1,2)]
        ppo_agent.mbi_neglogp_actions = [np.random.uniform(0,1), np.random.uniform(0,1)]
        ppo_agent.mbi_healths = [1.0, 1.0]

        # check error is raised
        with self.assertRaises(UpdateException):
            ppo_agent.process_individual_agent_episode_returns_and_advantages(factual_values=None, counterfactual_values=None)

    def test_process_individual_agent_episode_returns_and_advantages_5(self):
        '''mappo: extended sequence returns don't depend on value func'''

        Model = namedtuple('Model', ['value'])
        Args = namedtuple('Args', ['max_episode_len', 'gamma'])

        value_func = lambda obs, M: np.mean(obs)
        reward_func = lambda obs: np.sum(obs)
        model = Model(value_func)
        gamma = 1.0
        lam = 1.0
        args = Args(10, gamma)
        ppo_agent = PPOAgentComputer(name="ppo_agent_0", model=model,
            obs_shape_n=None, act_space_n=None, agent_index=0, args=args, local_q_func=None, lam=lam)

        ppo_agent.mbi_observations = [np.array([-0.61322181,  0.60141474]),
                                        np.array([-0.68131643, -0.46429067]),
                                        np.array([-0.32310118, -0.21411603]),
                                        np.array([ 0.59954657, -0.09719427]),
                                        np.array([0.20816313, 0.15251241]),
                                        np.array([0.14608069, 0.69522925]),
                                        np.array([-0.03096035,  0.10213929]),
                                        np.array([ 0.66119021, -0.69454451]),
                                        np.array([-0.69480874,  0.09734647]),
                                        np.array([0.74504277, 0.20447294]),
                                        np.array([0.16639411, 0.67739031])]
        ppo_agent.mbi_dones = 11*[False]
        ppo_agent.mbi_dones[-1] = True
        ppo_agent.mbi_actions = [np.random.uniform(-1,1,2), 
                                 np.random.uniform(-1,1,2),
                                 np.random.uniform(-1,1,2), 
                                 np.random.uniform(-1,1,2),
                                 np.random.uniform(-1,1,2), 
                                 np.random.uniform(-1,1,2),
                                 np.random.uniform(-1,1,2), 
                                 np.random.uniform(-1,1,2),
                                 np.random.uniform(-1,1,2), 
                                 np.random.uniform(-1,1,2)]
        ppo_agent.mbi_neglogp_actions = list(np.random.uniform(0,1,10))
        ppo_agent.mbi_healths = list(np.ones(10))
        for obs in ppo_agent.mbi_observations[:-1]:
            ppo_agent.mbi_rewards.append(reward_func(obs))
            ppo_agent.mbi_obs_values.append(value_func(obs, M=None))

        # check returns
        ppo_agent.process_individual_agent_episode_returns_and_advantages(factual_values=None, counterfactual_values=None)
        for i, ret in enumerate(ppo_agent.mbi_returns):
            self.assertAlmostEqual(ret, np.sum([reward_func(obs) for obs in ppo_agent.mbi_observations[i:-1]]), places=5)
            


class TestPPOGroupTrainer1(unittest.TestCase):
    ''' test PPOGroupTrainer class from mappo.py
    '''

    def setUp(self):
        ''' the with tf.Graph.as_default()... command allows for multiple calls to setUp
        without causing variable scopes to "clash". See baselines/common/tests/util.py for examples
        '''
        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default():
            # create trainer that would live in a simple 1D environment
            # with 1D continuous observations and actions
            # and single step episodes
            self.group_trainer = PPOGroupTrainer(
                    n_agents=2, 
                    obs_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), 
                    act_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    n_steps_per_episode=1, ent_coef=0.0, local_actor_learning_rate=3e-4, vf_coef=0.5,
                    num_layers=2, num_units=64, activation='tanh', cliprange=0.2, shared_reward=False,
                    critic_type='distributed_local_observations', central_critic_model=None, central_critic_learning_rate=None,
                    central_critic_num_units=None,
                    joint_state_space_len=3*4, max_grad_norm = 0.5, n_opt_epochs = 4,
                    n_episodes_per_batch=1, n_minibatches=1)

            # overwrite model value estimator with simple pass-through function
            # to simplify testing
            self.group_trainer.local_actor_critic_model.value = lambda obs, M: obs

            # Populate the group with stripped out versions of agents
            Args = namedtuple('Args', ['max_episode_len', 'gamma'])
            args = Args(1, 0.99)
            self.agent_0 = PPOAgentComputer(
                name="agent_0", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=0, args=args, local_q_func=None)
            self.agent_1 = PPOAgentComputer(
                name="agent_1", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=1, args=args, local_q_func=None)
            self.group_trainer.update_agent_trainer_group([self.agent_0, self.agent_1])

            # give agents artificially, randomly generated experience
            self.agent_0.mbi_observations = [np.array([-0.78438007]), np.array([-0.62432])]   
            self.agent_0.mbi_rewards = [-0.78438007]
            self.agent_0.mbi_obs_values = [-0.78438007] # value func just passes through input (ie observations)
            self.agent_0.mbi_actions = [np.array([-0.90892982])]
            self.agent_0.mbi_dones = [False, True]
            self.agent_0.mbi_neglogp_actions = [0.0]
            self.agent_0.mbi_healths = [0.0]

            self.agent_1.mbi_observations = [np.array([0.03254343]), np.array([0.24190804])]  
            self.agent_1.mbi_rewards = [0.03254343]
            self.agent_1.mbi_obs_values = [0.03254343] # value func just passes through input (ie observations)
            self.agent_1.mbi_actions = [np.array([-0.61390828])]
            self.agent_1.mbi_dones = [False, True]
            self.agent_1.mbi_neglogp_actions = [0.0]
            self.agent_1.mbi_healths = [0.0]

    def tearDown(self):
        '''Don't actually tearDown the tf graph
        Note: it may seem tempting to use tf.reset_default_graph(), but this
        causes an error in subsequent setUp calls with something to do with 
        op: NoOp ... is not an element of this graph
        Instead use the with tf.Graph.as_default()... in setUp
        '''
        pass


    def test_process_individual_agent_episode_returns_and_advantages_1(self):
        '''mappo: one-step with zero advantage '''
        self.agent_0.process_individual_agent_episode_returns_and_advantages(factual_values=None, counterfactual_values=None)
        self.assertAlmostEqual(self.agent_0.mbi_returns[0], -0.78438007, places=5)
        self.assertAlmostEqual(self.agent_0.mbi_factual_advantages[0], 0.0, places=5)

        self.agent_1.process_individual_agent_episode_returns_and_advantages(factual_values=None, counterfactual_values=None)
        self.assertAlmostEqual(self.agent_1.mbi_returns[0], 0.03254343, places=5)
        self.assertAlmostEqual(self.agent_0.mbi_factual_advantages[0], 0.0, places=5)

    def test_update_group_policy_1(self):
        '''mappo: smoke test - update_group_policy without throwing an error'''
        self.assertEqual(len(self.group_trainer.agent_trainer_group[0].mbi_rewards), 1)
        self.assertEqual(len(self.group_trainer.agent_trainer_group[1].mbi_rewards), 1)
        self.group_trainer.update_group_policy(terminal=1)
        self.assertEqual(len(self.group_trainer.agent_trainer_group[0].mbi_rewards), 0)
        self.assertEqual(len(self.group_trainer.agent_trainer_group[1].mbi_rewards), 0)


class TestPPOGroupTrainer2(unittest.TestCase):
    ''' test PPOGroupTrainer class from mappo.py
    '''

    def setUp(self):
        ''' the with tf.Graph.as_default()... command allows for multiple calls to setUp
        without causing variable scopes to "clash". See baselines/common/tests/util.py for examples
        '''
        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default():
            # create trainer that would live in a simple 1D environment
            # with 1D continuous observations and actions
            # and single step episodes
            self.episode_len = 5
            self.group_trainer = PPOGroupTrainer(
                    n_agents=3, 
                    obs_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), 
                    act_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    n_steps_per_episode=self.episode_len, ent_coef=0.0, local_actor_learning_rate=3e-4, vf_coef=0.5,
                    num_layers=2, num_units=64, activation='tanh', cliprange=0.2, 
                    n_episodes_per_batch=10, shared_reward=False,
                    critic_type='distributed_local_observations', central_critic_model=None, central_critic_learning_rate=None,
                    central_critic_num_units=None,
                    joint_state_space_len=3*4, max_grad_norm = 0.5, n_opt_epochs = 4, n_minibatches=4)

            # overwrite model value estimator with simple pass-through function
            # to simplify testing
            self.group_trainer.local_actor_critic_model.value = lambda obs, M: obs

            # Populate the group with stripped out versions of agents
            Args = namedtuple('Args', ['max_episode_len', 'gamma'])
            args = Args(self.episode_len, 0.99)
            self.agent_0 = PPOAgentComputer(
                name="agent_0", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=0, args=args, local_q_func=None, lam=1.0)
            self.agent_1 = PPOAgentComputer(
                name="agent_1", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=1, args=args, local_q_func=None, lam=1.0)
            self.agent_2 = PPOAgentComputer(
                name="agent_1", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=2, args=args, local_q_func=None, lam=1.0)
            self.group_trainer.update_agent_trainer_group([self.agent_0, self.agent_1, self.agent_2])

    def tearDown(self):
        '''Don't actually tearDown the tf graph
        Note: it may seem tempting to use tf.reset_default_graph(), but this
        causes an error in subsequent setUp calls with something to do with 
        op: NoOp ... is not an element of this graph
        Instead use the with tf.Graph.as_default()... in setUp
        '''
        pass


    def test_iterative_update_group_policy_1(self):
        '''mappo: run several iterations of update_group_policy calls and check minibatch sizes'''
        for ep in range(10):

            # for each episode, the group batch data should grow by number of agents
            self.assertEqual(len(self.group_trainer.batch_observations), 
                self.group_trainer.n_agents*self.group_trainer.n_steps_per_episode*ep)
            self.assertEqual(len(self.group_trainer.batch_observations), 
                len(self.group_trainer.batch_factual_values))
            self.assertEqual(len(self.group_trainer.batch_observations), 
                len(self.group_trainer.batch_counterfactual_values))
            self.assertEqual(len(self.group_trainer.batch_observations), 
                len(self.group_trainer.batch_returns))
            self.assertEqual(len(self.group_trainer.batch_observations), 
                len(self.group_trainer.batch_actions))
            self.assertEqual(len(self.group_trainer.batch_observations), 
                len(self.group_trainer.batch_neglogp_actions))
            self.assertEqual(len(self.group_trainer.batch_observations), 
                len(self.group_trainer.batch_dones))


            for ag in self.group_trainer.agent_trainer_group:
                for step in range(5):
                    ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                    ag.mbi_actions.append(np.random.uniform(-1., +1., 1))
                    ag.mbi_rewards.append(np.random.uniform(0, +1.))
                    ag.mbi_obs_values.append(np.random.uniform(0, +1.))
                    ag.mbi_dones.append(False)
                    ag.mbi_neglogp_actions.append(-np.log(np.random.uniform(0,1)))
                    ag.mbi_healths.append(1.0)
                ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                ag.mbi_dones.append(True)

            self.group_trainer.update_group_policy(terminal=1)
            self.assertEqual(len(self.group_trainer.agent_trainer_group[0].mbi_rewards), 0)
            self.assertEqual(len(self.group_trainer.agent_trainer_group[1].mbi_rewards), 0)
            self.assertEqual(len(self.group_trainer.agent_trainer_group[2].mbi_rewards), 0)

        # after 10 episode, a policy update should have occurred and cleared the group
        # minibatch
        self.assertEqual(len(self.group_trainer.batch_observations), 0)
        self.assertEqual(len(self.group_trainer.batch_factual_values), 0)
        self.assertEqual(len(self.group_trainer.batch_counterfactual_values), 0)
        self.assertEqual(len(self.group_trainer.batch_actions), 0)
        self.assertEqual(len(self.group_trainer.batch_returns), 0)
        self.assertEqual(len(self.group_trainer.batch_dones), 0)
        self.assertEqual(len(self.group_trainer.batch_neglogp_actions), 0)
        self.assertEqual(len(self.group_trainer.batch_effective_returns), 0)

    def test_multi_agent_returns_1(self):
        '''mappo: equal returns when shared rewards and lamba=1, regardless of individual value estimates'''
        n_episodes = 10
        for ep in range(n_episodes):

            # Generate true global state of system
            state = np.zeros(len(self.group_trainer.agent_trainer_group))
            for step in range(self.episode_len):
                for ag_ind, ag in enumerate(self.group_trainer.agent_trainer_group):
                    ag.mbi_observations.append(np.random.normal(state[ag_ind], 0.1, 1))
                    ag.mbi_actions.append(np.random.normal(1.0, 0.1, 1))
                    ag.mbi_obs_values.append(np.random.normal(ag.mbi_observations[-1][0], 10.0))
                    ag.mbi_dones.append(False)
                    ag.mbi_neglogp_actions.append(-np.log(np.random.uniform(0,1)))
                    ag.mbi_healths.append(1.0)

                    # update state
                    state[ag_ind] += ag.mbi_actions[-1][0]

                # calculate reward:
                reward = np.mean(state)
                for ag_ind, ag in enumerate(self.group_trainer.agent_trainer_group):
                    ag.mbi_rewards.append(reward)

                if step == self.episode_len-1:
                    for ag_ind, ag in enumerate(self.group_trainer.agent_trainer_group):
                        ag.mbi_observations.append(np.random.normal(state[ag_ind], 0.1, 1))
                        ag.mbi_dones.append(True)

            # test that returns are same for all agents
            self.agent_0.process_individual_agent_episode_returns_and_advantages(factual_values=None, counterfactual_values=None)
            self.agent_1.process_individual_agent_episode_returns_and_advantages(factual_values=None, counterfactual_values=None)
            self.agent_2.process_individual_agent_episode_returns_and_advantages(factual_values=None, counterfactual_values=None)
            for step in range(self.episode_len):
                # rewards
                self.assertAlmostEqual(self.agent_0.mbi_rewards[step], self.agent_1.mbi_rewards[step], places=5)
                self.assertAlmostEqual(self.agent_0.mbi_rewards[step], self.agent_2.mbi_rewards[step], places=5)
                # returns
                self.assertAlmostEqual(self.agent_0.mbi_returns[step], self.agent_1.mbi_returns[step], places=5)
                self.assertAlmostEqual(self.agent_0.mbi_returns[step], self.agent_2.mbi_returns[step], places=5)

            # reset for next episode (not actually calling training)
            self.agent_0.clear_individual_agent_episode_data()
            self.agent_1.clear_individual_agent_episode_data()
            self.agent_2.clear_individual_agent_episode_data()

    def test_multi_agent_returns_2(self):
        '''mappo: equal returns, rewards, and advantages when values centralized'''
        n_episodes = 10
        for ep in range(n_episodes):

            # Generate true global state of system
            state = np.zeros(len(self.group_trainer.agent_trainer_group))
            central_values = np.zeros(self.episode_len+1)
            for step in range(self.episode_len):
                for ag_ind, ag in enumerate(self.group_trainer.agent_trainer_group):
                    ag.mbi_observations.append(np.random.normal(state[ag_ind], 0.1, 1))
                    ag.mbi_actions.append(np.random.normal(1.0, 0.1, 1))
                    ag.mbi_obs_values.append(np.random.normal(ag.mbi_observations[-1][0], 10.0))
                    ag.mbi_dones.append(False)
                    ag.mbi_neglogp_actions.append(-np.log(np.random.uniform(0,1)))
                    ag.mbi_healths.append(1.0)

                    if step ==self.episode_len-1:
                        ag.mbi_observations.append(np.random.normal(state[ag_ind], 0.1, 1))
                        ag.mbi_dones.append(True)

                    # update state
                    state[ag_ind] += ag.mbi_actions[-1][0]

                # calculate reward:
                reward = np.mean(state)
                for ag_ind, ag in enumerate(self.group_trainer.agent_trainer_group):
                    ag.mbi_rewards.append(reward)

                # calculate centralized values
                central_values[step] = np.mean([ag.mbi_obs_values[step] for ag in self.group_trainer.agent_trainer_group])

                # if step == self.episode_len-1:
                #     for ag_ind, ag in enumerate(self.group_trainer.agent_trainer_group):
                #         ag.mbi_observations.append(np.random.normal(state[ag_ind], 0.1, 1))
                #         ag.mbi_dones.append(True)
                #     central_values[step+1] = np.mean([np.random.normal(ag.mbi_observations[-1][0], 10.0) for ag in self.group_trainer.agent_trainer_group])

            # test that returns, advantages are same for all agents with centralized values
            self.agent_0.process_individual_agent_episode_returns_and_advantages(factual_values=central_values, counterfactual_values=None)
            self.agent_1.process_individual_agent_episode_returns_and_advantages(factual_values=central_values, counterfactual_values=None)
            self.agent_2.process_individual_agent_episode_returns_and_advantages(factual_values=central_values, counterfactual_values=None)
            for step in range(self.episode_len):
                # rewards
                self.assertAlmostEqual(self.agent_0.mbi_rewards[step], self.agent_1.mbi_rewards[step], places=5)
                self.assertAlmostEqual(self.agent_0.mbi_rewards[step], self.agent_2.mbi_rewards[step], places=5)
                # returns
                self.assertAlmostEqual(self.agent_0.mbi_returns[step], self.agent_1.mbi_returns[step], places=5)
                self.assertAlmostEqual(self.agent_0.mbi_returns[step], self.agent_2.mbi_returns[step], places=5)
                # advantages
                self.assertAlmostEqual(self.agent_0.mbi_factual_advantages[step], self.agent_1.mbi_factual_advantages[step], places=5)
                self.assertAlmostEqual(self.agent_0.mbi_factual_advantages[step], self.agent_2.mbi_factual_advantages[step], places=5)
                # values
                self.assertAlmostEqual(self.agent_0.mbi_factual_values[step], self.agent_1.mbi_factual_values[step], places=5)
                self.assertAlmostEqual(self.agent_0.mbi_factual_values[step], self.agent_2.mbi_factual_values[step], places=5)

            # reset for next episode (not actually calling training)
            self.agent_0.clear_individual_agent_episode_data()
            self.agent_1.clear_individual_agent_episode_data()
            self.agent_2.clear_individual_agent_episode_data()

    def test_multi_agent_heuristic_credit_assignment_1(self):
        '''mappo: heuristic credits: all agents receive equal credit if return equals return mean and all actions same probability'''

        # change shared_reward to true for this test
        self.group_trainer.shared_reward = True
        self.group_trainer.crediting_algorithm = 'batch_mean_deviation_heuristic'

        for trial in range(10):

            # generate random reward history that each agent will have for every episode
            common_reward_history = np.random.normal(0,10, self.group_trainer.n_steps_per_episode)
            # generate random action probability that all agent use for given step
            common_neglogp_actions = -np.log(np.random.uniform(0,1, 
                (self.group_trainer.n_episodes_per_batch, self.group_trainer.n_steps_per_episode)))
     

            for ep in range(self.group_trainer.n_episodes_per_batch):
                
                # check size of batch is growing appropriately
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    self.group_trainer.n_agents*self.group_trainer.n_steps_per_episode*ep)
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_factual_values))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_counterfactual_values))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_returns))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_actions))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_neglogp_actions))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_dones))

                for ag in self.group_trainer.agent_trainer_group:
                    for step in range(self.group_trainer.n_steps_per_episode):
                        ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                        ag.mbi_actions.append(np.random.uniform(-1., +1., 1))
                        ag.mbi_rewards.append(common_reward_history[step])
                        ag.mbi_obs_values.append(np.random.uniform(0, +1.))
                        ag.mbi_dones.append(False)
                        ag.mbi_neglogp_actions.append(common_neglogp_actions[ep][step])
                        ag.mbi_healths.append(1.0)
                    ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                    ag.mbi_dones.append(True)

                if ep < self.group_trainer.n_episodes_per_batch - 1:
                    self.group_trainer.update_group_policy(terminal=1)
                    self.assertEqual(len(self.group_trainer.agent_trainer_group[0].mbi_rewards), 0)
                    self.assertEqual(len(self.group_trainer.agent_trainer_group[1].mbi_rewards), 0)
                    self.assertEqual(len(self.group_trainer.agent_trainer_group[2].mbi_rewards), 0)
                
                else:
                    # don't actually run final update call, call batch_credit_assignment instead
                    break

            # format batch and run credit assignment
            episode_factual_values, episode_counterfactual_values = self.group_trainer.process_episode_value_centralization_and_credit_assignment()
            self.group_trainer.process_episode_returns_and_store_group_training_batch(episode_factual_values, episode_counterfactual_values)
            self.group_trainer.process_episode_clear_data()
            crediting_info = self.group_trainer.batch_credit_assignment()
            return_stds = crediting_info[1]
            credit_scale = crediting_info[2]

            # check that every agent is receiving the same credit
            self.assertEqual(len(self.group_trainer.batch_effective_returns), self.group_trainer.n_data_per_batch)
            for ep in range(self.group_trainer.n_episodes_per_batch):
                for step in range(self.group_trainer.n_steps_per_episode):
                    self.assertAlmostEqual(return_stds[step], 0.0, places=5)
                    self.assertAlmostEqual(credit_scale[ep][step], 0.0, places=5)

                    for ag in range(self.group_trainer.n_agents):
                        batch_index = (ep*self.group_trainer.n_agents + ag) * self.group_trainer.n_steps_per_episode + step
                        self.assertAlmostEqual(self.group_trainer.batch_neglogp_actions[batch_index],
                            common_neglogp_actions[ep][step], places=5)
                        self.assertAlmostEqual(self.group_trainer.batch_effective_returns[batch_index], 
                            self.group_trainer.batch_returns[batch_index]/float(self.group_trainer.n_agents),
                            places=5)

            # execute training to refresh batch data
            self.group_trainer.execute_group_training()

    def test_multi_agent_heurisitic_credit_assignment_2(self):
        '''mappo: heurisitc credits: one agent receives all the credit when action prob much larger and returns=mean'''

        # change shared_reward to true for this test
        self.group_trainer.shared_reward = True
        self.group_trainer.crediting_algorithm = 'batch_mean_deviation_heuristic'

        for trial in range(10):

            # generate random reward history that each agent will have for every episode
            common_reward_history = np.random.normal(0,10, self.group_trainer.n_steps_per_episode)

            # generate random action probabilities with one agent recieving high prob and others low
            high_neglogp_actions = -np.log(np.random.uniform(0.999,1, 
                (self.group_trainer.n_episodes_per_batch, self.group_trainer.n_steps_per_episode)))
            low_neglogp_actions = -np.log(np.random.uniform(0,0.001, 
                (self.group_trainer.n_episodes_per_batch, self.group_trainer.n_steps_per_episode)))

            # pick random agent to recieve high probility actions
            lucky_agent = np.random.randint(self.group_trainer.n_agents, 
                size=(self.group_trainer.n_episodes_per_batch,))

            for ep in range(self.group_trainer.n_episodes_per_batch):
                
                # check size of batch is growing appropriately
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    self.group_trainer.n_agents*self.group_trainer.n_steps_per_episode*ep)
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_factual_values))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_counterfactual_values))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_returns))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_actions))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_neglogp_actions))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_dones))

                for ag_ind, ag in enumerate(self.group_trainer.agent_trainer_group):
                    for step in range(self.group_trainer.n_steps_per_episode):
                        ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                        ag.mbi_actions.append(np.random.uniform(-1., +1., 1))
                        ag.mbi_rewards.append(common_reward_history[step])
                        ag.mbi_obs_values.append(np.random.uniform(0, +1.))
                        ag.mbi_dones.append(False)
                        ag.mbi_healths.append(1.0)
                        if ag_ind == lucky_agent[ep]:
                            ag.mbi_neglogp_actions.append(high_neglogp_actions[ep][step])
                        else:
                            ag.mbi_neglogp_actions.append(low_neglogp_actions[ep][step])
                    ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                    ag.mbi_dones.append(True)

                if ep < self.group_trainer.n_episodes_per_batch - 1:
                    self.group_trainer.update_group_policy(terminal=1)
                    self.assertEqual(len(self.group_trainer.agent_trainer_group[0].mbi_rewards), 0)
                    self.assertEqual(len(self.group_trainer.agent_trainer_group[1].mbi_rewards), 0)
                    self.assertEqual(len(self.group_trainer.agent_trainer_group[2].mbi_rewards), 0)
                
                else:
                    # don't actually run final update call, call batch_credit_assignment instead
                    break

            # format batch and run credit assignment
            episode_factual_values, episode_counterfactual_values = self.group_trainer.process_episode_value_centralization_and_credit_assignment()
            self.group_trainer.process_episode_returns_and_store_group_training_batch(episode_factual_values, episode_counterfactual_values)
            self.group_trainer.process_episode_clear_data()
            crediting_info = self.group_trainer.batch_credit_assignment()
            return_stds = crediting_info[1]
            credit_scale = crediting_info[2]

            # check that one agent recieves almost all the credit
            self.assertEqual(len(self.group_trainer.batch_effective_returns), self.group_trainer.n_data_per_batch)
            for ep in range(self.group_trainer.n_episodes_per_batch):
                for step in range(self.group_trainer.n_steps_per_episode):
                    self.assertAlmostEqual(return_stds[step], 0.0, places=5)
                    self.assertAlmostEqual(credit_scale[ep][step], 0.0, places=5)

                    for ag in range(self.group_trainer.n_agents):
                        batch_index = (ep*self.group_trainer.n_agents + ag) * self.group_trainer.n_steps_per_episode + step
                        tol = abs(self.group_trainer.batch_returns[batch_index])/10.0
                        if ag == lucky_agent[ep]:
                            self.assertAlmostEqual(self.group_trainer.batch_effective_returns[batch_index], 
                                self.group_trainer.batch_returns[batch_index], delta=tol)
                        else:
                            self.assertAlmostEqual(self.group_trainer.batch_effective_returns[batch_index], 0.0, delta=tol)


            # execute training to refresh batch data
            self.group_trainer.execute_group_training()

    def test_multi_agent_heuristic_credit_assignment_3(self):
        '''mappo: No crediting: check that returns equal credits when no crediting applied'''

        # change shared_reward to true for this test
        self.group_trainer.crediting_algorithm = None

        for trial in range(10):

            # generate random reward history that each agent will have for every episode
            common_reward_history = np.random.normal(0,10, self.group_trainer.n_steps_per_episode)

            for ep in range(10):

                # for each episode, the group batch data should grow by number of agents
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    self.group_trainer.n_agents*self.group_trainer.n_steps_per_episode*ep)
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_factual_values))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_counterfactual_values))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_returns))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_actions))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_neglogp_actions))
                self.assertEqual(len(self.group_trainer.batch_observations), 
                    len(self.group_trainer.batch_dones))


                for ag in self.group_trainer.agent_trainer_group:
                    for step in range(5):
                        ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                        ag.mbi_actions.append(np.random.uniform(-1., +1., 1))
                        ag.mbi_rewards.append(common_reward_history[step])
                        ag.mbi_obs_values.append(np.random.uniform(0, +1.))
                        ag.mbi_dones.append(False)
                        ag.mbi_neglogp_actions.append(-np.log(np.random.uniform(0,1)))
                        ag.mbi_healths.append(1.0)
                    ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                    ag.mbi_dones.append(True)

                if ep < self.group_trainer.n_episodes_per_batch - 1:
                        self.group_trainer.update_group_policy(terminal=1)
                        self.assertEqual(len(self.group_trainer.agent_trainer_group[0].mbi_rewards), 0)
                        self.assertEqual(len(self.group_trainer.agent_trainer_group[1].mbi_rewards), 0)
                        self.assertEqual(len(self.group_trainer.agent_trainer_group[2].mbi_rewards), 0)
                    
                else:
                    # don't actually run final update call, call batch_credit_assignment instead
                    break

            # format batch and run credit assignment
            episode_factual_values, episode_counterfactual_values = self.group_trainer.process_episode_value_centralization_and_credit_assignment()
            self.group_trainer.process_episode_returns_and_store_group_training_batch(episode_factual_values, episode_counterfactual_values)
            self.group_trainer.process_episode_clear_data()
            self.group_trainer.batch_credit_assignment()
            # credit_scale = crediting_info[2]

            # check that one agent recieves almost all the credit
            self.assertEqual(len(self.group_trainer.batch_effective_returns), self.group_trainer.n_data_per_batch)
            for ep in range(self.group_trainer.n_episodes_per_batch):
                for step in range(self.group_trainer.n_steps_per_episode):
                    # self.assertAlmostEqual(credit_scale[ep][step], 0.0, places=5)
                    expected_credit = self.group_trainer.batch_effective_returns[ep*self.group_trainer.n_agents*self.group_trainer.n_steps_per_episode+step]
                    for ag in range(self.group_trainer.n_agents):
                        batch_index = (ep*self.group_trainer.n_agents + ag) * self.group_trainer.n_steps_per_episode + step
                        # tol = abs(self.group_trainer.batch_returns[batch_index])/10.0
                        self.assertAlmostEqual(
                            self.group_trainer.batch_effective_returns[batch_index], 
                            self.group_trainer.batch_returns[batch_index], places=4)
                        self.assertAlmostEqual(
                            self.group_trainer.batch_effective_returns[batch_index], 
                            expected_credit, places=4)

            # execute training to refresh batch data
            self.group_trainer.execute_group_training()


class TestCentralCriticNetwork1(unittest.TestCase):
    ''' test central_critic_network class from mappo.py
    '''

    def setUp(self):
        ''' the with tf.Graph.as_default()... command allows for multiple calls to setUp
        without causing variable scopes to "clash". See baselines/common/tests/util.py for examples
        '''

        with tf.Graph().as_default() as self.setup_graph, tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default() as self.setup_sess:
           
            self.test_n_training_iterations = 1000
            self.test_n_data_per_batch = 100
            self.test_num_layers = 2
            self.test_num_units = 8
            self.test_activation = 'tanh'
            self.test_learning_rate = 1e-2
            self.test_input_size = 1
            self.test_cliprange = 0.2
            joint_state_stamped_ph = [U.BatchInput((self.test_input_size, ), name="joint_state").get()]
            deep_mlp = DeepMLP(num_layers=self.test_num_layers, activation=self.test_activation)

            self.central_vf_value, self.central_vf_train, self.central_vf_debug = central_critic_network(
                inputs_placeholder_n=joint_state_stamped_ph,
                v_func=deep_mlp.deep_mlp_model,
                optimizer=tf.train.AdamOptimizer(learning_rate=self.test_learning_rate),
                scope = "joint_state_critic",
                num_units=self.test_num_units,
                grad_norm_clipping=0.5
            )

    def tearDown(self):
        '''Don't actually tearDown the tf graph
        Note: it may seem tempting to use tf.reset_default_graph(), but this
        causes an error in subsequent setUp calls with something to do with 
        op: NoOp ... is not an element of this graph
        Instead use the with tf.Graph.as_default()... in setUp
        '''
        pass

    def test_central_critic_network_constant_target(self):
        '''mappo: central critic learning constant target value'''

        # randomly generated but fixed constant target, regardless of input
        const_target = 8.245529015329097

        # in order to make calls to the central value function, we need to operate within the tf session
        # and initialize variables
        with self.setup_sess:
            self.setup_sess.run(tf.global_variables_initializer())

            for train_iter in range(self.test_n_training_iterations):

                # create individual training batch of random input but fixed target 
                training_feed = [[], [], [], []]
                for i in range(self.test_n_data_per_batch):
                    rand_input = np.random.uniform(-1., +1., self.test_input_size)
                    training_feed[0].append(rand_input)
                    training_feed[1].append(const_target)
                    training_feed[2].append(self.central_vf_value(np.expand_dims(rand_input, axis=0))[0])
                training_feed[3] = self.test_cliprange

                # call train and update target network
                central_vf_loss = self.central_vf_train(*training_feed)

            # check that value estimate has converged to const_target
            test_vals = []
            for test_iter in range(1000):
                test_vals.append(self.central_vf_value(np.expand_dims(np.random.uniform(-1., +1., self.test_input_size),axis=0)))
            # print("test mean = {} | test std = {}".format(np.mean(test_vals), np.std(test_vals)))
            self.assertAlmostEqual(np.mean(test_vals), const_target, places=3)


class TestCentralCriticNetwork2(unittest.TestCase):
    ''' test central_critic_network class from mappo.py
    '''

    def setUp(self):
        ''' the with tf.Graph.as_default()... command allows for multiple calls to setUp
        without causing variable scopes to "clash". See baselines/common/tests/util.py for examples
        '''

        with tf.Graph().as_default() as self.setup_graph, tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default() as self.setup_sess:
           
            self.test_n_training_iterations = 1000
            self.test_n_data_per_batch = 128
            self.test_num_layers = 4
            self.test_num_units = 64
            self.test_activation = 'elu'
            self.test_learning_rate = 1e-3
            self.test_input_size = 1
            self.test_test_size = 10000
            self.test_cliprange = 0.2
            joint_state_stamped_ph = [U.BatchInput((self.test_input_size, ), name="joint_state").get()]
            deep_mlp = DeepMLP(num_layers=self.test_num_layers, activation=self.test_activation)

            self.central_vf_value, self.central_vf_train, self.central_vf_debug = central_critic_network(
                inputs_placeholder_n=joint_state_stamped_ph,
                v_func=deep_mlp.deep_mlp_model,
                optimizer=tf.train.AdamOptimizer(learning_rate=self.test_learning_rate),
                scope = "joint_state_critic",
                num_units=self.test_num_units,
                grad_norm_clipping=0.5
            )

    def tearDown(self):
        '''Don't actually tearDown the tf graph
        Note: it may seem tempting to use tf.reset_default_graph(), but this
        causes an error in subsequent setUp calls with something to do with 
        op: NoOp ... is not an element of this graph
        Instead use the with tf.Graph.as_default()... in setUp
        '''
        pass

    def test_central_critic_network_periodic_target(self):
        '''mappo: central critic learning periodic function (this may take a while)'''

        # sinusoidal target function
        periodic_target = lambda x: np.sin(x)

        # in order to make calls to the central value function, we need to operate within the tf session
        # and initialize variables
        with self.setup_sess:
            self.setup_sess.run(tf.global_variables_initializer())

            central_vf_loss = []
            central_vf_expvar = []
            for train_iter in range(self.test_n_training_iterations):

                # create individual training batch of random input but fixed target 
                training_feed = [[], [], [], []]
                for i in range(self.test_n_data_per_batch):
                    rand_input = np.random.uniform(-10., +10., self.test_input_size)
                    training_feed[0].append(rand_input)
                    training_feed[1].append(periodic_target(rand_input)[0])
                    training_feed[2].append(self.central_vf_value(np.expand_dims(rand_input, axis=0))[0])
                training_feed[3] = self.test_cliprange

                # call train and update target network
                central_vf_loss.append(self.central_vf_train(*training_feed))
                central_vf_expvar.append(explained_variance(self.central_vf_value(training_feed[0]), np.asarray(training_feed[1])))


                if _DEBUG:
                    rand_in = np.random.uniform(-10., +10., self.test_input_size)
                    val_est = self.central_vf_value(np.expand_dims(rand_in,axis=0))
                    val_tar = periodic_target(rand_in[0])
                    example_diff = val_est - val_tar
                    print("iter {} | in={:5.2f} | tar={:5.2f} | est={:7.3f} | diff={:7.3f} | loss={:7.3E} | expln var={:7.3E}".format(
                        train_iter, 
                        rand_in[0],
                        val_tar,
                        val_est[0],
                        example_diff[0],
                        central_vf_loss[-1],
                        central_vf_expvar[-1]
                        ))

            if _DEBUG:
                ti = np.arange(self.test_n_training_iterations)
                plt.plot(ti, central_vf_loss, ti, central_vf_expvar)
                plt.xlabel('training iteration')
                plt.ylabel('value loss & explained variance')
                plt.legend(['value loss', 'explained variance'])
                plt.show()

            # check value loss has converged to expected level (based on emperical testing)
            self.assertLessEqual(np.mean(central_vf_loss[-int(self.test_n_training_iterations*.005):]), 5e-3)
            self.assertGreaterEqual(np.mean(central_vf_expvar[-int(self.test_n_training_iterations*.005):]), 0.975)


            # check that value estimate has converged 
            test_vals = [[],[],[],[]]
            for test_iter in range(self.test_test_size):
                test_vals[0].append(np.random.uniform(-10., +10., self.test_input_size))
                test_vals[1].append(self.central_vf_value(np.expand_dims(test_vals[0][-1],axis=0)))
                test_vals[2].append(periodic_target(test_vals[0][-1]))
                test_vals[3].append(test_vals[1][-1] - test_vals[2][-1])


            # print("test mean = {} | test std = {}".format(np.mean(test_vals), np.std(test_vals)))
            self.assertAlmostEqual(np.mean(test_vals[3]), 0.0, places=1)
            self.assertLessEqual(np.std(test_vals[3]), 0.1)

class TestCentralCriticNetwork3(unittest.TestCase):
    ''' test central_critic_network class from mappo.py
    '''

    def setUp(self):
        ''' the with tf.Graph.as_default()... command allows for multiple calls to setUp
        without causing variable scopes to "clash". See baselines/common/tests/util.py for examples
        '''

        with tf.Graph().as_default() as self.setup_graph, tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default() as self.setup_sess:
           
            self.test_n_training_iterations = 1000
            self.test_n_data_per_batch = 128
            self.test_num_layers = 4
            self.test_num_units = 64
            self.test_activation = 'elu'
            self.test_learning_rate = 1e-3
            self.test_agent_state_len = 5
            self.test_n_agents = 4
            self.test_input_size = 1 + self.test_agent_state_len*self.test_n_agents
            self.test_test_size = 10000
            self.test_cliprange = 0.2
            joint_state_stamped_ph = [U.BatchInput((self.test_input_size, ), name="joint_state").get()]
            deep_mlp = DeepMLP(num_layers=self.test_num_layers, activation=self.test_activation)

            self.central_vf_value, self.central_vf_train, self.central_vf_debug = central_critic_network(
                inputs_placeholder_n=joint_state_stamped_ph,
                v_func=deep_mlp.deep_mlp_model,
                optimizer=tf.train.AdamOptimizer(learning_rate=self.test_learning_rate),
                scope = "joint_state_critic",
                num_units=self.test_num_units,
                grad_norm_clipping=0.5
            )

    def tearDown(self):
        '''Don't actually tearDown the tf graph
        Note: it may seem tempting to use tf.reset_default_graph(), but this
        causes an error in subsequent setUp calls with something to do with 
        op: NoOp ... is not an element of this graph
        Instead use the with tf.Graph.as_default()... in setUp
        '''
        pass

    def test_central_critic_network_terminated_target(self):
        '''mappo: central critic learning nonlinear terminated target similar to XOR (this may take a while)'''

        # randomly generated but fixed constant target, regardless of input
        def terminated_target(s):
            # reward if only one agent is terminated
            n_term = sum(s[self.test_agent_state_len::self.test_agent_state_len])
            if np.isclose(n_term, 1.0):
                return s[0]
            else:
                return 0.0

        def gen_rand_input():
            rand_input = [np.random.randint(50)+1]
            for agsi in range(1, self.test_input_size, self.test_agent_state_len):
                rand_input.extend(np.random.uniform(-10., +10., self.test_agent_state_len-1))
                rand_input.extend([np.random.randint(2)])
            return rand_input


        # in order to make calls to the central value function, we need to operate within the tf session
        # and initialize variables
        with self.setup_sess:
            self.setup_sess.run(tf.global_variables_initializer())

            central_vf_loss = []
            central_vf_expvar = []
            for train_iter in range(self.test_n_training_iterations):

                # create individual training batch of random input but fixed target 
                training_feed = [[], [], [], []]
                for i in range(self.test_n_data_per_batch):
                    rand_input = gen_rand_input()
                    training_feed[0].append(rand_input)
                    training_feed[1].append(terminated_target(rand_input))
                    training_feed[2].append(self.central_vf_value(np.expand_dims(rand_input, axis=0))[0])
                training_feed[3] = self.test_cliprange

                # call train and update target network
                central_vf_loss.append(self.central_vf_train(*training_feed))
                central_vf_expvar.append(explained_variance(self.central_vf_value(training_feed[0]), np.asarray(training_feed[1])))

                if _DEBUG:
                    rand_in = gen_rand_input()
                    val_est = self.central_vf_value(np.expand_dims(rand_in,axis=0))
                    val_tar = terminated_target(rand_in)
                    example_diff = val_est - val_tar
                    print("iter {} | in={:5.2f} | tar={:5.2f} | est={:7.3f} | diff={:7.3f} | loss={:7.3E} | expln var={:7.3E}".format(
                        train_iter, 
                        rand_in[0],
                        val_tar,
                        val_est[0],
                        example_diff[0],
                        central_vf_loss[-1],
                        central_vf_expvar[-1]
                        ))

            if _DEBUG:
                ti = np.arange(self.test_n_training_iterations)
                plt.plot(ti, central_vf_loss, ti, central_vf_expvar)
                plt.xlabel('training iteration')
                plt.ylabel('value loss & explained variance')
                plt.legend(['value loss', 'explained variance'])
                plt.show()

            # check value loss and explained variance has converged to expected level (based on emperical testing)
            self.assertLessEqual(np.mean(central_vf_loss[-int(self.test_n_training_iterations*.005):]), 2.0)
            self.assertGreaterEqual(np.mean(central_vf_expvar[-int(self.test_n_training_iterations*.005):]), 0.975)


            # # check that value estimate has converged 
            # test_vals = [[],[],[],[]]
            # for test_iter in range(self.test_test_size):
            #     test_vals[0].append(np.random.uniform(-10., +10., self.test_input_size))
            #     test_vals[1].append(self.central_vf_value(np.expand_dims(test_vals[0][-1],axis=0)))
            #     test_vals[2].append(periodic_target(test_vals[0][-1]))
            #     test_vals[3].append(test_vals[1][-1] - test_vals[2][-1])


            # # print("test mean = {} | test std = {}".format(np.mean(test_vals), np.std(test_vals)))
            # self.assertAlmostEqual(np.mean(test_vals[3]), 0.0, places=2)
            # self.assertLessEqual(np.std(test_vals[3]), 0.1)


class TestPPOGroupTrainer3(unittest.TestCase):
    ''' test PPOGroupTrainer class from mappo.py
    '''

    def setUp(self):
        ''' the with tf.Graph.as_default()... command allows for multiple calls to setUp
        without causing variable scopes to "clash". See baselines/common/tests/util.py for examples
        '''

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default():
           
            self.test_n_training_iterations = 1000
            self.test_episode_len = 5
            self.test_n_episodes_per_batch = 10
            self.test_num_layers = 2
            self.test_activation = 'tanh'
            self.test_n_opt_epochs = 4
            self.test_n_minibatches = 4
            self.test_gamma = 0.99
            self.test_joint_state_space_len = 1
            deep_mlp = DeepMLP(num_layers=self.test_num_layers, activation=self.test_activation)

            # create trainer that would live in a simple 1D environment
            # with 1D continuous observations and actions
            # and single step episodes
            self.group_trainer = PPOGroupTrainer(
                    n_agents=3, 
                    obs_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), 
                    act_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    n_steps_per_episode=self.test_episode_len, ent_coef=0.0, local_actor_learning_rate=3e-4, vf_coef=0.5,
                    num_layers=2, num_units=4, activation=self.test_activation, cliprange=0.2, 
                    n_episodes_per_batch=self.test_n_episodes_per_batch, shared_reward=True,
                    critic_type='central_joint_state', central_critic_model=deep_mlp.deep_mlp_model, 
                    central_critic_learning_rate=3e-4, central_critic_num_units=4, joint_state_space_len=self.test_joint_state_space_len,
                    max_grad_norm = 0.5, n_opt_epochs=self.test_n_opt_epochs, n_minibatches=self.test_n_minibatches)


            # Populate the group with stripped out versions of agents
            Args = namedtuple('Args', ['max_episode_len', 'gamma'])
            args = Args(self.test_episode_len, self.test_gamma)
            self.agent_0 = PPOAgentComputer(
                name="agent_0", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=0, args=args, local_q_func=None, lam=1.0)
            self.agent_1 = PPOAgentComputer(
                name="agent_1", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=1, args=args, local_q_func=None, lam=1.0)
            self.agent_2 = PPOAgentComputer(
                name="agent_1", 
                model=self.group_trainer.local_actor_critic_model, 
                obs_shape_n=None, act_space_n=None, 
                agent_index=2, args=args, local_q_func=None, lam=1.0)
            self.group_trainer.update_agent_trainer_group([self.agent_0, self.agent_1, self.agent_2])

    def tearDown(self):
        '''Don't actually tearDown the tf graph
        Note: it may seem tempting to use tf.reset_default_graph(), but this
        causes an error in subsequent setUp calls with something to do with 
        op: NoOp ... is not an element of this graph
        Instead use the with tf.Graph.as_default()... in setUp
        '''
        pass

    def nontest_execute_group_training_central_joint_state_critic_1(self):
        '''mappo: (this test currently deprecated but not removed yet because using some of the code as a guide)
        integration test of many functions to ensure central joint state critic converges given constant input
        '''

        self.assertTrue(False) # this test currently deprecated but not removed yet because using some of the code as a guide

        const_reward = 8.245529015329097

        # in order to make calls to the central value function, we need to operate within the tf session
        # and initialize variables
        # with self.group_trainer.sess:
        #     tf.global_variables_initializer()
        with self.group_trainer.sess as sess:
            sess.run(tf.global_variables_initializer())

            training_loss_stats = []
            for train_iter in range(self.test_n_training_iterations):
                for ep in range(self.test_n_episodes_per_batch):

                    # for each episode, the group batch data should grow by number of agents*time steps
                    self.assertEqual(len(self.group_trainer.batch_observations), 
                        self.group_trainer.n_agents*self.group_trainer.n_steps_per_episode*ep)
                    self.assertEqual(len(self.group_trainer.batch_observations), 
                        len(self.group_trainer.batch_factual_values))
                    self.assertEqual(len(self.group_trainer.batch_observations), 
                        len(self.group_trainer.batch_counterfactual_values))
                    self.assertEqual(len(self.group_trainer.batch_observations), 
                        len(self.group_trainer.batch_returns))
                    self.assertEqual(len(self.group_trainer.batch_observations), 
                        len(self.group_trainer.batch_actions))
                    self.assertEqual(len(self.group_trainer.batch_observations), 
                        len(self.group_trainer.batch_neglogp_actions))
                    self.assertEqual(len(self.group_trainer.batch_observations), 
                        len(self.group_trainer.batch_dones))

                    # for each episode, joint data should grow by number of time steps
                    self.assertEqual(len(self.group_trainer.batch_joint_state_stamped), 
                        (self.group_trainer.n_steps_per_episode+1)*ep)

                    # populate episode with random data, except rewards, those are constant
                    for ag in self.group_trainer.agent_trainer_group:
                        for step in range(self.test_episode_len):
                            ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                            ag.mbi_actions.append(np.random.uniform(-1., +1., 1))
                            ag.mbi_rewards.append(const_reward)     # only element that is constant, not randomly varying
                            ag.mbi_obs_values.append(np.random.uniform(0, +1.))
                            ag.mbi_dones.append(False)
                            ag.mbi_neglogp_actions.append(-np.log(np.random.uniform(0,1)))
                            ag.mbi_healths.append(1.0)
                        ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                        ag.mbi_dones.append(True)

                    for step in range(self.test_episode_len+1):
                        # self.group_trainer.record_joint_state(np.array([
                        #     np.random.uniform(-1., +1., 4), np.random.uniform(-1., +1., 4), np.random.uniform(-1., +1., 4)]))
                        self.group_trainer.record_joint_state(np.array([np.random.uniform(-1., +1., self.test_joint_state_space_len)]))

                    # self.group_trainer.update_group_policy(terminal=1)
                    episode_factual_values, episode_counterfactual_values = self.group_trainer.process_episode_value_centralization_and_credit_assignment()
                    self.group_trainer.process_episode_returns_and_store_group_training_batch(episode_factual_values, episode_counterfactual_values)
                    self.group_trainer.process_episode_clear_data()

                    # check that returns are always the same sequence, given the constant reward
                    cur_return = const_reward
                    for ep_step in range(self.test_episode_len):
                        self.assertAlmostEqual(self.group_trainer.batch_joint_state_stamped[ep_step][0], self.test_episode_len-ep_step)
                        self.assertAlmostEqual(self.group_trainer.batch_joint_returns[-ep_step-1], cur_return, places=5)
                        cur_return = const_reward + self.test_gamma*cur_return

                    # check that individuals' memories are properly cleared out
                    self.assertEqual(len(self.group_trainer.agent_trainer_group[0].mbi_rewards), 0)
                    self.assertEqual(len(self.group_trainer.agent_trainer_group[1].mbi_rewards), 0)
                    self.assertEqual(len(self.group_trainer.agent_trainer_group[2].mbi_rewards), 0)

                # after episodes per batch, update policy
                self.group_trainer.batch_credit_assignment()
                batch_loss_stats = self.group_trainer.execute_group_training()
                training_loss_stats += [[self.test_episode_len*self.test_n_episodes_per_batch*(train_iter+1)] + L for L in batch_loss_stats]
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

                print("training iter {}: value at t = {}: {} | value at t = {}: {}".format(
                    train_iter, self.test_episode_len, self.group_trainer.central_vf_value(np.expand_dims(np.concatenate(([0], np.random.uniform(-1., +1., self.test_joint_state_space_len))),axis=0)),
                    0, self.group_trainer.central_vf_value(np.expand_dims(np.concatenate(([self.test_episode_len], np.random.uniform(-1., +1., self.test_joint_state_space_len))),axis=0))))

            print(self.group_trainer.central_vf_value(np.expand_dims(np.concatenate(([self.test_episode_len], np.random.uniform(-1., +1., self.test_joint_state_space_len))),axis=0)))
            self.assertTrue(False)


class TestPPOGroupTrainer_LocalCritic_NoCrediting_1(unittest.TestCase):
    '''Unit tests for individual subroutines in PPOGroupTrainer'''

    def setUp(self):
        ''' the with tf.Graph.as_default()... command allows for multiple calls to setUp
        without causing variable scopes to "clash". See baselines/common/tests/util.py for examples
        '''

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default():
            self.group_trainer = PPOGroupTrainer(
                n_agents=3, 
                obs_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), 
                act_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                n_steps_per_episode=50, ent_coef=0.0, local_actor_learning_rate=3e-4, vf_coef=0.5,
                num_layers=2, num_units=4, activation='tanh', cliprange=0.2, 
                n_episodes_per_batch=16, shared_reward=True,
                critic_type='distributed_local_observations', central_critic_model=None, 
                central_critic_learning_rate=None, central_critic_num_units=None, joint_state_space_len=None,
                max_grad_norm = 0.5, n_opt_epochs=4, n_minibatches=4)

    def tearDown(self):
        '''Don't actually tearDown the tf graph
        Note: it may seem tempting to use tf.reset_default_graph(), but this
        causes an error in subsequent setUp calls with something to do with 
        op: NoOp ... is not an element of this graph
        Instead use the with tf.Graph.as_default()... in setUp
        '''
        pass

    def test_process_episode_value_centralization_and_credit_assignment_1(self):
        '''mappo:process_episode_value_centralization_and_credit_assignment: local critic, no crediting'''

        # create trainer that would live in a simple 1D environment
        # with 1D continuous observations and actions
        # and single step episodes

        # Populate the group with generic objects
        self.group_trainer.update_agent_trainer_group([object, object, object])

        # call the centralization and crediting function
        episode_factual_values, episode_counterfactual_values = self.group_trainer.process_episode_value_centralization_and_credit_assignment()

        # check outputs
        self.assertTrue(episode_factual_values is None)
        self.assertTrue(episode_counterfactual_values is None)
        self.assertEqual(len(self.group_trainer.batch_joint_observations_stamped), 51)
        self.assertEqual(len(self.group_trainer.batch_joint_state_stamped), 51)
        for i,_ in enumerate(self.group_trainer.batch_joint_observations_stamped):
            self.assertTrue(self.group_trainer.batch_joint_observations_stamped[i] is None)
            self.assertTrue(self.group_trainer.batch_joint_state_stamped[i] is None)


class TestPPOGroupTrainer_JointObserveCritic_NoCrediting_1(unittest.TestCase):
    '''Unit tests for individual subroutines in PPOGroupTrainer'''

    def setUp(self):
        ''' the with tf.Graph.as_default()... command allows for multiple calls to setUp
        without causing variable scopes to "clash". See baselines/common/tests/util.py for examples
        '''

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default():
            # create trainer that would live in a simple 1D environment
            # with 1D continuous observations and actions
            # with randomized parameterized when they are not important for this test
            self.group_trainer = PPOGroupTrainer(
                n_agents=np.random.randint(9)+2, 
                obs_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), 
                act_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                n_steps_per_episode=50, ent_coef=np.random.rand(), local_actor_learning_rate=np.random.rand(), vf_coef=np.random.rand(),
                num_layers=np.random.randint(15)+2, num_units=np.random.randint(255)+2, activation='tanh', cliprange=np.random.rand(), 
                n_episodes_per_batch=np.random.randint(1024)+1, shared_reward=True,
                critic_type='central_joint_observations', central_critic_model=DeepMLP(num_layers=np.random.randint(16)+1, activation='tanh').deep_mlp_model, 
                central_critic_learning_rate=np.random.rand(), joint_state_space_len=np.random.randint(256)+1,
                central_critic_num_units=np.random.randint(255)+2,
                max_grad_norm = np.random.rand(), n_opt_epochs=np.random.randint(16)+1, n_minibatches=np.random.randint(16)+1)

    def tearDown(self):
        '''Don't actually tearDown the tf graph
        Note: it may seem tempting to use tf.reset_default_graph(), but this
        causes an error in subsequent setUp calls with something to do with 
        op: NoOp ... is not an element of this graph
        Instead use the with tf.Graph.as_default()... in setUp
        '''
        pass

    def test_process_episode_value_centralization_and_credit_assignment_1(self):
        '''mappo:process_episode_value_centralization_and_credit_assignment: joint observations critic, no crediting'''

        n_steps = self.group_trainer.n_steps_per_episode
        n_agents = self.group_trainer.n_agents

        # Overwrite central value function with simple, dummy value function
        self.group_trainer.central_vf_value = lambda jnt_obs: [sum(sum(jnt_obs))]

        # Populate the group with stripped out versions of agents with random observation
        class DummyAgent(object):
            def __init__(self, nsteps):
                # self.mbi_observations = list(np.random.uniform(-1,1,group_trainer.n_steps_per_episode+1))
                self.mbi_observations = [[np.random.uniform(-1,1)] for i in range(nsteps+1)]

        agent_group = []
        for i in range(self.group_trainer.n_agents):
            agent_group.append(DummyAgent(n_steps))
        self.group_trainer.update_agent_trainer_group(agent_group)

        # call the centralization and crediting function
        episode_factual_values, episode_counterfactual_values = self.group_trainer.process_episode_value_centralization_and_credit_assignment()

        # check outputs
        self.assertEqual(n_agents, self.group_trainer.n_agents)
        self.assertEqual(len(episode_factual_values), n_steps+1)
        self.assertEqual(len(episode_counterfactual_values), n_agents)
        self.assertEqual(len(self.group_trainer.batch_joint_observations_stamped), n_steps+1)
        self.assertEqual(len(self.group_trainer.batch_joint_state_stamped), n_steps+1)
        for i in range(n_steps+1):
            self.assertEqual(len(self.group_trainer.batch_joint_observations_stamped[i]), n_agents+1)
            self.assertAlmostEqual(self.group_trainer.batch_joint_observations_stamped[i][0], n_steps+1-i) # check time stamp
            expect_value = n_steps+1 - i + sum([ag.mbi_observations[i][0] for ag in agent_group])
            if i == n_steps: expect_value = 0.0
            self.assertAlmostEqual(episode_factual_values[i], expect_value) # all equal without crediting
            for agi in range(n_agents):
                self.assertTrue(episode_counterfactual_values[agi][i] is None) # No crediting


class TestPPOGroupTrainer_JointStateCritic_NoCrediting_1(unittest.TestCase):
    '''Unit tests for individual subroutines in PPOGroupTrainer'''

    def setUp(self):
        ''' the with tf.Graph.as_default()... command allows for multiple calls to setUp
        without causing variable scopes to "clash". See baselines/common/tests/util.py for examples
        '''

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default():
            # create trainer that would live in a simple 1D environment
            # with 1D continuous observations and actions
            # with randomized parameterized when they are not important for this test
            n_agents=np.random.randint(9)+2
            self.group_trainer = PPOGroupTrainer(
                n_agents=n_agents, 
                obs_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), 
                act_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                n_steps_per_episode=50, ent_coef=np.random.rand(), local_actor_learning_rate=np.random.rand(), vf_coef=np.random.rand(),
                num_layers=np.random.randint(8)+1, num_units=np.random.randint(63)+2, activation='tanh', cliprange=np.random.rand(), 
                n_episodes_per_batch=np.random.randint(63)+2, shared_reward=True,
                critic_type='central_joint_state', central_critic_model=DeepMLP(num_layers=np.random.randint(8)+1, activation='tanh').deep_mlp_model, 
                central_critic_learning_rate=np.random.rand(), central_critic_num_units=np.random.randint(63)+2, joint_state_space_len=2*n_agents,
                max_grad_norm = np.random.rand(), n_opt_epochs=np.random.randint(16)+1, n_minibatches=np.random.randint(16)+1, joint_state_entity_len=2)

    def tearDown(self):
        '''Don't actually tearDown the tf graph
        Note: it may seem tempting to use tf.reset_default_graph(), but this
        causes an error in subsequent setUp calls with something to do with 
        op: NoOp ... is not an element of this graph
        Instead use the with tf.Graph.as_default()... in setUp
        '''
        pass

    def test_process_episode_value_centralization_and_credit_assignment_1(self):
        '''mappo:process_episode_value_centralization_and_credit_assignment: joint state critic, no crediting'''

        n_steps = self.group_trainer.n_steps_per_episode
        n_agents = self.group_trainer.n_agents

        # Overwrite central value function with simple, dummy value function
        self.group_trainer.central_vf_value = lambda jnt_obs: [sum(sum(jnt_obs))]

        # Populate the group with stripped out versions of agents
        class DummyAgent(object):
            def __init__(self):
                pass
        agent_group = []
        for i in range(self.group_trainer.n_agents):
            agent_group.append(DummyAgent())
        self.group_trainer.update_agent_trainer_group(agent_group)

        # create randomized central state generator
        self.group_trainer.episode_joint_state = [np.random.uniform(-1,1,n_agents) for i in range(n_steps+1)]

        # call the centralization and crediting function
        episode_factual_values, episode_counterfactual_values = self.group_trainer.process_episode_value_centralization_and_credit_assignment()

        # check outputs
        self.assertEqual(n_agents, self.group_trainer.n_agents)
        self.assertEqual(len(episode_factual_values), n_steps+1)
        self.assertEqual(len(episode_counterfactual_values), n_agents)
        self.assertEqual(len(self.group_trainer.batch_joint_observations_stamped), n_steps+1)
        self.assertEqual(len(self.group_trainer.batch_joint_state_stamped), n_steps+1)
        for i in range(n_steps+1):
            self.assertEqual(len(self.group_trainer.batch_joint_state_stamped[i]), n_agents+1)
            self.assertAlmostEqual(self.group_trainer.batch_joint_state_stamped[i][0], n_steps+1-i) # check time stamp
            expect_value = n_steps+1 - i + sum(self.group_trainer.episode_joint_state[i])
            if i == n_steps: expect_value = 0.0
            self.assertAlmostEqual(episode_factual_values[i], expect_value) # all equal without crediting
            for agi in range(n_agents):
                self.assertTrue(episode_counterfactual_values[agi][i] is None) # No crediting

    def test_process_episode_subroutines_1(self):
        '''mappo:process_episode_[subroutine]: joint state critic, no crediting'''

        n_steps = self.group_trainer.n_steps_per_episode
        n_agents = self.group_trainer.n_agents
        n_episodes = self.group_trainer.n_episodes_per_batch
        n_trials = 10
        gamma = 0.99
        lam = 1.0

        # Overwrite central value function with simple, simple value function
        self.group_trainer.central_vf_value = lambda jnt_obs: [sum(sum(jnt_obs))]

        # Establish args for stripped out versions of agents
        Args = namedtuple('Args', ['max_episode_len', 'gamma'])
        args = Args(n_steps, gamma)

        for trial in range(n_trials):

            # generate random reward history that each agent will have for every episode
            common_reward_history = np.random.normal(0,10, n_steps)

            # generate new group of agents
            agent_group = []
            for agi in range(n_agents):
                # agent_group.append(DummyAgent(n_steps, gamma, lam))
                agent_group.append(PPOAgentComputer(
                    name="agent_{}".format(agi), 
                    model=self.group_trainer.local_actor_critic_model, 
                    obs_shape_n=None, act_space_n=None, 
                    agent_index=agi, args=args, local_q_func=None, lam=lam))
            self.group_trainer.update_agent_trainer_group(agent_group)

            for ep in range(n_episodes):

                # for each episode, the group batch data should grow by number of agents
                expect_len = n_agents*n_steps*ep
                self.assertEqual(len(self.group_trainer.batch_observations), expect_len)
                self.assertEqual(len(self.group_trainer.batch_factual_values), expect_len)
                self.assertEqual(len(self.group_trainer.batch_counterfactual_values), expect_len)
                self.assertEqual(len(self.group_trainer.batch_actions), expect_len)
                self.assertEqual(len(self.group_trainer.batch_neglogp_actions), expect_len)
                self.assertEqual(len(self.group_trainer.batch_dones), expect_len)
                self.assertEqual(len(self.group_trainer.batch_returns), expect_len)
                self.assertEqual(len(self.group_trainer.batch_joint_observations_stamped), ep*(n_steps+1))
                self.assertEqual(len(self.group_trainer.batch_joint_state_stamped), ep*(n_steps+1))

                # fill agent history with random input, except rewards the same
                for ag in self.group_trainer.agent_trainer_group:
                    for step in range(n_steps):
                        ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                        ag.mbi_actions.append(np.random.uniform(-1., +1., 1))
                        ag.mbi_rewards.append(common_reward_history[step])
                        ag.mbi_obs_values.append(np.random.uniform(0, +1.))
                        ag.mbi_dones.append(False)
                        ag.mbi_neglogp_actions.append(-np.log(np.random.uniform(0,1)))
                        ag.mbi_healths.append(1.0)
                    ag.mbi_observations.append(np.random.uniform(-1., +1., 1))
                    ag.mbi_dones.append(True)

                # create randomized central state generator
                self.group_trainer.episode_joint_state = [np.random.uniform(-1,1,n_agents) for i in range(n_steps+1)]

                # get episode baseline values    
                episode_factual_values, episode_counterfactual_values = self.group_trainer.process_episode_value_centralization_and_credit_assignment()

                # check baseline values
                self.assertEqual(len(episode_factual_values), self.group_trainer.n_steps_per_episode+1)
                self.assertEqual(len(episode_counterfactual_values), self.group_trainer.n_agents)
                for i in range(n_steps+1):
                    self.assertEqual(len(self.group_trainer.batch_joint_state_stamped[i]), n_agents+1)
                    self.assertAlmostEqual(self.group_trainer.batch_joint_state_stamped[i][0], n_steps+1-i) # check time stamp
                    expect_value = n_steps+1 - i + sum(self.group_trainer.episode_joint_state[i])
                    if i == n_steps: expect_value = 0.0
                    self.assertAlmostEqual(episode_factual_values[i], expect_value) # all equal without crediting
                    for agi in range(n_agents):
                        self.assertTrue(episode_counterfactual_values[agi][i] is None) # No crediting

                # calculate returns, advantages and store in batch
                self.group_trainer.process_episode_returns_and_store_group_training_batch(episode_factual_values, episode_counterfactual_values)

                # check episode and batch data
                for agi, ag in enumerate(self.group_trainer.agent_trainer_group):

                    # with no crediting, returns and values should match batch_joint values
                    self.assertTrue(np.allclose(self.group_trainer.batch_joint_returns[-n_steps-1:], ag.mbi_returns))
                    s1 = -n_steps*(n_agents-agi)
                    s2 = -n_steps*(n_agents-agi-1) if -n_steps*(n_agents-agi-1) < 0 else None
                    self.assertTrue(np.allclose(self.group_trainer.batch_factual_values[s1:s2], ag.mbi_factual_values[:-1]))
                    self.assertTrue(np.allclose(self.group_trainer.batch_actions[s1:s2], ag.mbi_actions))
                    self.assertTrue(np.allclose(self.group_trainer.batch_returns[s1:s2], ag.mbi_returns[:-1]))
                    self.assertTrue(np.allclose(self.group_trainer.batch_neglogp_actions[s1:s2], ag.mbi_neglogp_actions))
                    self.assertTrue(np.allclose(self.group_trainer.batch_healths[s1:s2], ag.mbi_healths))

                # clear episode data
                self.group_trainer.process_episode_clear_data()

                # check episode data cleared out
                self.assertEqual(len(self.group_trainer.episode_joint_state), 0)    # episode state cleared out
                for ag in self.group_trainer.agent_trainer_group:
                    # each agent's episode data cleared
                    self.assertEqual(len(ag.mbi_observations), 0)
                    self.assertEqual(len(ag.mbi_actions), 0)
                    self.assertEqual(len(ag.mbi_rewards), 0)
                    self.assertEqual(len(ag.mbi_obs_values), 0)
                    self.assertEqual(len(ag.mbi_dones), 0)
                    self.assertEqual(len(ag.mbi_neglogp_actions), 0)
                    self.assertEqual(len(ag.mbi_healths), 0)

                
                if ep == n_episodes - 1:
                    # clear out batch data (don't actually run any of the training functions,
                    # trying to keep this test more trimmed down)
                    # Clear out group batch
                    self.group_trainer.batch_observations = []
                    self.group_trainer.batch_joint_observations_stamped = []
                    self.group_trainer.batch_joint_state_stamped = []
                    self.group_trainer.batch_returns = []
                    self.group_trainer.batch_joint_returns = []
                    self.group_trainer.batch_effective_returns = []
                    self.group_trainer.batch_dones = []
                    self.group_trainer.batch_actions = [] 
                    self.group_trainer.batch_factual_values = []
                    self.group_trainer.batch_counterfactual_values = []
                    self.group_trainer.batch_effective_values = []
                    self.group_trainer.batch_neglogp_actions = []

class TestPPOGroupTrainer_JointStateCritic_TerminatedBaselineCrediting_1(unittest.TestCase):
    '''Tests for individual subroutines in PPOGroupTrainer with joint-state critic and terminated baseline crediting'''

    def setUp(self):
        ''' the with tf.Graph.as_default()... command allows for multiple calls to setUp
        without causing variable scopes to "clash". See baselines/common/tests/util.py for examples
        '''

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)).as_default():
            # create trainer that would live in a simple 1D environment
            # with 1D continuous observations and actions
            # with randomized parameterized when they are not important for this test
            n_agents = np.random.randint(9)+2
            self.entity_state_len = 5
            self.group_trainer = PPOGroupTrainer(
                n_agents=n_agents, 
                obs_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), 
                act_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                n_steps_per_episode=50, ent_coef=np.random.rand(), local_actor_learning_rate=np.random.rand(), vf_coef=np.random.rand(),
                num_layers=np.random.randint(8)+1, num_units=np.random.randint(63)+2, activation='tanh', cliprange=np.random.rand(), 
                n_episodes_per_batch=np.random.randint(63)+2, shared_reward=True,
                critic_type='central_joint_state', central_critic_model=DeepMLP(num_layers=np.random.randint(8)+1, activation='tanh').deep_mlp_model, 
                central_critic_learning_rate=np.random.rand(), joint_state_space_len=self.entity_state_len*n_agents,
                central_critic_num_units=np.random.randint(63)+2,
                max_grad_norm = np.random.rand(), n_opt_epochs=np.random.randint(16)+1, n_minibatches=np.random.randint(16)+1,
                crediting_algorithm = 'terminated_baseline')

    def tearDown(self):
        '''Don't actually tearDown the tf graph
        Note: it may seem tempting to use tf.reset_default_graph(), but this
        causes an error in subsequent setUp calls with something to do with 
        op: NoOp ... is not an element of this graph
        Instead use the with tf.Graph.as_default()... in setUp
        '''
        pass

    def test_process_episode_value_centralization_and_credit_assignment_1(self):
        '''mappo:process_episode_value_centralization_and_credit_assignment: joint state critic, terminated baseline crediting'''

        n_steps = self.group_trainer.n_steps_per_episode
        n_agents = self.group_trainer.n_agents
        entity_state_len = self.entity_state_len


        # Overwrite central value function with simple function that sums non-terminated states
        # self.group_trainer.central_vf_value = lambda s: [sum([s1*(1-s2) for s1,s2 in zip(s[1::entity_state_len], s[5::entity_state_len])])]
        def value_func(jss):
            jss = jss[0] # strip off additional layer that is added in mappo
            return [sum([s1*(1-s2) for s1,s2 in zip(jss[1::entity_state_len], jss[5::entity_state_len])])]
        self.group_trainer.central_vf_value = value_func

        # Populate the group with stripped out versions of agents
        class DummyAgent(object):
            def __init__(self):
                pass
        agent_group = []
        jsl = []
        for agi in range(self.group_trainer.n_agents):
            agent_group.append(DummyAgent())
            jsl.append("agent_{}".format(agi))
        self.group_trainer.update_agent_trainer_group(agent_group)
        self.group_trainer.joint_state_labels = jsl

        # create randomized central state generator with all agents at full health
        # self.group_trainer.episode_joint_state = [[None]*n_agents]*(n_steps+1)
        self.group_trainer.episode_joint_state = [None]*(n_steps+1)
        for i in range(n_steps+1):
            cur_state = []
            for agi in range(n_agents):
                cur_state.extend(np.append(np.random.uniform(-1,1,entity_state_len-1), 0.0))

            self.group_trainer.episode_joint_state[i] = cur_state

        # call the centralization and crediting function
        episode_factual_values, episode_counterfactual_values = self.group_trainer.process_episode_value_centralization_and_credit_assignment()

        # check outputs
        self.assertEqual(n_agents, self.group_trainer.n_agents)
        self.assertEqual(len(episode_factual_values), n_steps+1)
        self.assertEqual(len(episode_counterfactual_values), n_agents)
        self.assertEqual(len(self.group_trainer.batch_joint_observations_stamped), n_steps+1)
        self.assertEqual(len(self.group_trainer.batch_joint_state_stamped), n_steps+1)
        for i in range(n_steps+1):
            self.assertEqual(len(self.group_trainer.batch_joint_state_stamped[i]), entity_state_len*n_agents+1)
            self.assertAlmostEqual(self.group_trainer.batch_joint_state_stamped[i][0], n_steps+1-i) # check time stamp
            actual_expect_value = sum(self.group_trainer.batch_joint_state_stamped[i][1::entity_state_len]) # expected true value of state is sum over non-terminated states, ignoring stamp, with no agents terminated 
            self.assertAlmostEqual(self.group_trainer.central_vf_value(np.expand_dims(self.group_trainer.batch_joint_state_stamped[i],axis=0))[0], actual_expect_value)
            if i == n_steps:
                self.assertAlmostEqual(episode_factual_values[i], 0.0)
            else:
                self.assertAlmostEqual(episode_factual_values[i], actual_expect_value)
            for agi in range(n_agents):
                self.assertAlmostEqual(self.group_trainer.batch_joint_state_stamped[i][1+(agi+1)*entity_state_len-1], 0) # actual termination values are false
                counterfactual_expect_value = actual_expect_value - self.group_trainer.batch_joint_state_stamped[i][1+agi*entity_state_len]
                self.assertAlmostEqual(episode_counterfactual_values[agi][i], counterfactual_expect_value)   # all equal without crediting

class TestRedistributedSoftmax(unittest.TestCase):
    ''' 
    '''

    def setUp(self):
        pass

    def test_redistributed_softmax_single_value(self):
        '''redistributed_softmax: random single-value'''
        for _ in range(100):
            p_arr = [np.random.normal(0.0, 10.0)]
            scale = np.random.uniform(0.0, 1.0)
            p_scaled = redistributed_softmax(p_arr, scale)
            self.assertAlmostEqual(p_scaled[0], 1.0)

    def test_redistributed_softmax_two_values(self):
        '''redistributed_softmax: random two-values'''
        for _ in range(100):
            p_arr = np.random.normal(0.0, 10.0, 2)
            scale = np.random.uniform(0.0, 1.0)
            p_scaled = redistributed_softmax(p_arr, scale)
            self.assertAlmostEqual(sum(p_scaled), 1.0)
            if scale > 0.5:
                self.assertGreaterEqual(p_scaled[p_arr.argmin()], p_scaled[p_arr.argmax()])
            else:
                self.assertLessEqual(p_scaled[p_arr.argmin()], p_scaled[p_arr.argmax()])

    def test_redistributed_softmax_multi_values(self):
        '''redistributed_softmax: random multi-values'''
        for _ in range(100):
            n = np.random.randint(1,20)
            p_arr = np.random.normal(0.0, 10.0, n)
            scale = np.random.uniform(0.0, 1.0)
            p_scaled = redistributed_softmax(p_arr, scale)
            self.assertAlmostEqual(sum(p_scaled), 1.0)
            if n > 1 and scale > 1.0 - 1.0/float(n):
                self.assertFalse(p_scaled.argmax() == p_arr.argmax())


if __name__ == '__main__':
    unittest.main()
