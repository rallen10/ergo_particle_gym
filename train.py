import argparse
import gym
import numpy as np
import os
import sys
import tensorflow as tf
import time
import pickle
import datetime
import subprocess
import sys
import inspect
from copy import deepcopy
import tensorflow.contrib.layers as layers
# from contextlib import redirect_stderr
from shutil import copyfile

# import environments, training algorithms, and utilities (note rl_algs must be imported first for sys.path definitions)
import particle_environments as pt_envs
import rl_algorithms as rl_algs
import rl_algorithms.maddpg.maddpg.common.tf_util as U
from rl_algorithms.mclearning import ScenarioHeuristicGroupTrainer
from rl_algorithms.baselines_agent_trainer import BaselinesAgentTrainer
from rl_algorithms.mappo import PPOGroupTrainer, PPOAgentComputer

import faulthandler
faulthandler.enable()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(args_feed):
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--environment", type=str, default="MultiAgentEnv", help="name of environment")
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script to run in environment")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--variable-num-agents", type=int, default=4, help="number of agents, if variable scenario")
    parser.add_argument("--variable-num-hazards", type=int, default=0, help="number of hazards, if variable scenario")
    parser.add_argument("--variable-local-rewards", type=str2bool, nargs='?', const=True, default=False, help="false if agents receive identical rewards, if variable scenario")
    parser.add_argument("--variable-observation-type", type=str, default="direct", help="observation function to use direct or histogram, if variable scenario")
    # Core training parameters
    parser.add_argument("--training-algorithm", type=str, default="MADDPGAgentTrainer", help="algorithm to perform learning (e.g. maddpg, trpo, none, etc.)")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--variable-learning-rate", type=str2bool, nargs='?', const=True, default=False, help="PPO: enable variable learning rate")
    parser.add_argument("--learning-rate-min", type=float, default=1e-5, help="PPO: min learning rate used if variable learning rate enabled")
    parser.add_argument("--learning-rate-period", type=int, default=10, help="PPO: period of training cycles for variable learning rate")
    parser.add_argument("--entropy-coef", type=float, default=0.0, help="PPO: entropy coefficient exploration loss weighting")
    parser.add_argument("--value-coef", type=float, default=0.5, help="PPO: value function coefficient loss weighting")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-layers", type=int, default=2, help="number of units in the mlp")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--activation", type=str, default="relu", help="activation function for mlp layers")
    parser.add_argument("--cliprange", type=float, default=0.2, help="advantage clipping factor for PPO")
    parser.add_argument("--num-minibatches", type=int, default=4, help="PPO: number of minibatches to create from each batch")
    parser.add_argument("--num-opt-epochs", type=int, default=4, help="PPO: number of training epochs to run for each batch")
    # Central critic training parameters
    parser.add_argument("--critic-type", type=str, default="distributed_local_observations", help="PPO: set critic to be decentralized or centralized, based on observation or state")
    parser.add_argument("--central-critic-learning-rate", type=float, default=None, help="learning rate for central critic optimizer (if None, use learning_rate)")
    parser.add_argument("--central-critic-num-layers", type=int, default=None, help="number of units in the central critic mlp. If None, defaults to num-layers")
    parser.add_argument("--central-critic-num-units", type=int, default=None, help="number of units in the central critic mlp. If None, defaults to num-units")
    parser.add_argument("--central-critic-activation", type=str, default=None, help="activation function for central critic mlp layers. If None, defaults to activation")
    parser.add_argument("--crediting-algorithm", type=str, default=None, help="algorithm to apply multi-agent crediting when rewards are shared")
    # Checkpointing
    parser.add_argument("--experiment-name", type=str, default='default', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./experiments/default/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--display", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--benchmark", type=str2bool, nargs='?', const=True, default=False,)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./experiments/benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./experiments/default/", help="directory where plot data is saved")
    parser.add_argument("--record-experiment", type=str2bool, nargs='?', const=True, default=False, help="record all relevant info about experiment in one directory")
    return parser.parse_args(args_feed)

class DeepMLP(object):
    ''' deep, fully connected, multi-layer perceptron
    Notes:
     - defined this way so that we can set an arbitrary number of hidden layers (default 2)
     - This is NOT used for PPO or other baselines algorithms. See PolicyWithValue 
        in baselines/common/policies.py
    '''
    def __init__(self, num_layers, activation):
        ''' Note: num_units is handled separately for compatibility with maddpg
        '''
        self.num_layers = num_layers
        # self.num_units = num_units
        self.activation_fn = getattr(tf.nn, activation)

    def deep_mlp_model(self, input, num_outputs, scope, num_units=None, reuse=False, rnn_cell=None):
        ''' This model takes as input an observation and returns values of all actions
        Notes:
         - setting num_units to None to ensure that it must be set by whoever calls this (see maddpg.py)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            out = input
            for i in range(self.num_layers):
                out = layers.fully_connected(out, num_outputs=num_units, activation_fn=self.activation_fn)
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
            return out

def make_env(environment_name, scenario_name, arglist, 
    benchmark=False, discrete_action_space=True, legacy_multidiscrete=True):

    # load environment class
    env_class = pt_envs.load_environment_class(environment_name)

    # load scenario class, using modifiable scenario params if appropriate
    if 'variable' in scenario_name:
        scenario = pt_envs.load_scenario_module(environment_name, scenario_name).Scenario(
            num_agents = arglist.variable_num_agents, 
            num_hazards = arglist.variable_num_hazards,
            identical_rewards = not arglist.variable_local_rewards,
            observation_type = arglist.variable_observation_type)
    else:
        scenario = pt_envs.load_scenario_module(environment_name, scenario_name).Scenario()

    # create world
    world = scenario.make_world()

    # establish done callback function in case scenario doesn't include one 
    done_callback = None
    if hasattr(scenario, 'done_callback') and callable(scenario.done_callback):
        done_callback = scenario.done_callback

    # create environment
    if benchmark:
        env = env_class(world=world, 
                        reset_callback=scenario.reset_world, 
                        reward_callback=scenario.reward, 
                        observation_callback=scenario.observation, 
                        info_callback=scenario.benchmark_data,
                        done_callback=done_callback,
                        discrete_action_space=discrete_action_space,
                        legacy_multidiscrete=legacy_multidiscrete)
    else:
        env = env_class(world=world, 
                        reset_callback=scenario.reset_world, 
                        reward_callback=scenario.reward, 
                        observation_callback=scenario.observation, 
                        done_callback=done_callback,
                        discrete_action_space=discrete_action_space,
                        legacy_multidiscrete=legacy_multidiscrete)

    return env

def get_agent_trainers(env, num_adversaries, obs_shape_n, arglist):
    '''create list of agent trainers'''
    trainers = []
    group_trainer = None
    model = None

    # create trainer objects by mapping input arg string to class: e.g. NonLearningAgentTrainer, MADDPGAgentTrainer, etc
    if arglist.training_algorithm == 'ScenarioHeuristicAgentTrainer' or arglist.training_algorithm == 'ScenarioHeuristicGroupTrainer':
        # get trainer class specificly tailored to a scenario
        trainer_class = pt_envs.load_scenario_module(arglist.environment, arglist.scenario).ScenarioHeuristicComputer

    elif arglist.training_algorithm == 'PPOGroupTrainer':
        # get a group trainer based on ppo
        # NOTE: the model deep_mlp is not actually passed 
        # unless a centralized critic is used,
        # instead one is generated internally based on the 
        # baselines code base

        # set central critic model, if central critic used
        central_critic_model = None
        joint_state_space_len = None
        central_critic_num_layers = arglist.central_critic_num_layers
        central_critic_learning_rate = arglist.central_critic_learning_rate
        central_critic_activation = arglist.central_critic_activation
        central_critic_num_units = arglist.central_critic_num_units
        if arglist.critic_type == "distributed_local_observations":
            pass

        elif (  arglist.critic_type == "central_joint_observations" or 
                arglist.critic_type == "central_joint_state"):

            # select default or user defined paremeters for central critic
            central_critic_num_layers = central_critic_num_layers if central_critic_num_layers is not None else arglist.num_layers
            central_critic_activation = central_critic_activation if central_critic_activation is not None else arglist.activation
            central_critic_learning_rate = central_critic_learning_rate if central_critic_learning_rate is not None else arglist.learning_rate
            central_critic_num_units = central_critic_num_units if central_critic_num_units is not None else arglist.num_units

            # create central critic MLP model and determine joint state space
            deep_mlp = DeepMLP(central_critic_num_layers, central_critic_activation)
            central_critic_model = deep_mlp.deep_mlp_model
            joint_state_space = env.get_joint_state()
            joint_state_space_len = 0
            for s in joint_state_space['state']:
                joint_state_space_len += len(s)

        else:
            raise Exception('Unrecognized critic type description: {}'.format(self.critic_type))

        # set learning rate options
        if arglist.variable_learning_rate:
            max_lr = arglist.learning_rate
            min_lr = arglist.learning_rate_min
            period_lr = arglist.learning_rate_period
            assert max_lr >= min_lr > 0.0
            assert period_lr > 0
            learning_rate = lambda t: 2.*abs(t/period_lr - np.floor(t/period_lr + 0.5))*(max_lr-min_lr)+min_lr

        else:
            learning_rate = arglist.learning_rate

        # check for consistent observation and action spaces
        # Note that the term "group" (as opposed to "multi-agent")
        # implies that agents are homogenuous
        # with equivalent state and action spaces
        for i in range(env.n):
            if not (env.observation_space[i] == env.observation_space[0] and
                env.action_space[i] == env.action_space[0]):
                raise Exception("PPOGroupTrainer can only be used for groups of homogenuous agents.")

        # create single group trainer for all agents
        group_trainer = PPOGroupTrainer(
            n_agents=env.n, 
            obs_space = env.observation_space[0], 
            act_space = env.action_space[0],
            n_steps_per_episode = arglist.max_episode_len, 
            ent_coef = arglist.entropy_coef, 
            local_actor_learning_rate = learning_rate,
            vf_coef = arglist.value_coef,
            num_layers = arglist.num_layers, 
            num_units = arglist.num_units, 
            activation = arglist.activation,
            cliprange = arglist.cliprange,
            n_episodes_per_batch = arglist.batch_size, 
            shared_reward = env.shared_reward,
            critic_type = arglist.critic_type,
            central_critic_model = central_critic_model,
            central_critic_learning_rate = central_critic_learning_rate,
            central_critic_num_units = central_critic_num_units,
            joint_state_space_len = joint_state_space_len,
            max_grad_norm=0.5, 
            n_opt_epochs=arglist.num_opt_epochs, 
            n_minibatches=arglist.num_minibatches,
            crediting_algorithm=arglist.crediting_algorithm)

        # set trainer class as a ppo agent computer that is trained base of the group ppo model
        trainer_class = PPOAgentComputer
        model = group_trainer.local_actor_critic_model

    else:
        deep_mlp = DeepMLP(arglist.num_layers, arglist.activation)
        model = deep_mlp.deep_mlp_model
        trainer_class = rl_algs.load_trainer_class(arglist.training_algorithm)


    # create and append trainers for adversary agents
    # use init format for maddpg, monte-carlo, etc
    for i in range(num_adversaries):
        trainers.append(trainer_class(
            name="agent_%d" % i, 
            model=model, 
            obs_shape_n=obs_shape_n, 
            act_space_n=env.action_space, 
            agent_index=i, 
            args=arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))

    # create and append trainers for good agents
    for i in range(num_adversaries, env.n):
        trainers.append(trainer_class(
            name="agent_%d" % i, 
            model=model, 
            obs_shape_n=obs_shape_n, 
            act_space_n=env.action_space, 
            agent_index=i, 
            args=arglist,
            local_q_func=(arglist.good_policy=='ddpg')))

    # create scenario heuristic group trainer if applicable
    if arglist.training_algorithm == 'ScenarioHeuristicGroupTrainer':
        group_trainer = ScenarioHeuristicGroupTrainer(
                agent_trainer_group=trainers, 
                init_group_policy=None, 
                n_episodes_per_batch=arglist.batch_size, 
                n_elite=25)
    elif arglist.training_algorithm == 'PPOGroupTrainer':
        group_trainer.update_agent_trainer_group(trainers)

    return trainers, group_trainer

def get_trainer_actions(agents, trainers, observations, combined_action_value=False):
    '''return list of actions from each agent's trainer
    Args:
     - agents: list of agents in the environment; i.e. "physical" robots
     - trainers: list of decision makers for each agent; i.e. policy and learning algorithm
     - observations: list of each observation for each trainer
     - combined_action_value: boolean, does trainer.action output value est and probability
        along with action
    Returns:
     - action_n: list of actions for each of the n agents at a single time step
     - value_n: list of value estimate for each of the n agents actions 
     - neglogpact_n: list of negative log probability for each of the n agents actions 
    '''

    # check agents, observations, and trainers are synced as agents may be terminated
    # NOTE: this is a not an exhaustive check of syncing, just checks they are of the 
    # same size lists
    n_trainers = len(trainers)
    assert len(observations) == n_trainers
    assert len(agents) == n_trainers

    # initialize list of actions
    action_n = [None]*len(trainers)
    value_n = [None]*len(trainers)
    neglogpact_n = [None]*len(trainers)
    health_n = [1.0]*len(trainers)

    # get action from each trainer
    for i, trainer in enumerate(trainers):

        # ensure order of trainers has not been modified
        if int(trainer.name.split('_')[1]) != i: raise OrderingException('trainer list out of order')

        # get action for trainer-agent i
        avn = trainer.action(observations[i])
        if combined_action_value:
            assert len(avn) == 3
            assert avn[1].shape == (1,) # ensure that value is coming in expected format
            assert avn[2].shape == (1,) # ensure that value is coming in expected format
            action_n[i] = avn[0]
            value_n[i] = avn[1][0]
            neglogpact_n[i] = avn[2][0]
        else:
            action_n[i] = avn

        # handle terminated agent
        if hasattr(agents[i], 'terminated') and agents[i].terminated:
            # return all zeros for action. Can't use None since it messes up the ReplayBuffer
            # to have inconsistent action formatting
            action_n[i] = action_n[i]*0.0
            # if neglogpact_n[i] is not None:
            #     neglogpact_n[i] *= 0.0
            health_n[i] = 0.0

    return action_n, value_n, neglogpact_n, health_n

def setup_experiment_record(arglist):
    ''' Setup file structure for a new experiment and record call info

    Notes:
     - this is used for recording complete experiments to be used for further analysis. This
     is distinct from just "testing things out" and iterating designs.
     - It captures the git commit hash, command line call used to kick off experiment in
     a markdown file. In this way the experiment can be completely reproduceable.
     - It also stores the policy and learning curves in the same directory
    '''

    # copy input arglist
    input_arglist = deepcopy(arglist)

    # check if changes committed and get commit hash
    git_st = subprocess.check_output(['git', 'status', '--porcelain'])
    if len(git_st) > 0:
        raise Exception('Uncommitted changes present. Please commit all changes before running experiment')

    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])


    # enforce no display or benchmark
    arglist.display  = arglist.benchmark = False

    # check the save directory and adjust if default
    if arglist.save_dir == '/tmp/policy/':
        arglist.save_dir = '/tmp/gym_experiments/'

    # check experiment name and adjust if default
    if arglist.experiment_name == 'default':
        arglist.experiment_name = arglist.training_algorithm + '.' + arglist.scenario

    # make expdata directory within save directory
    count = 0
    while True:
        exp_dir_name = ('expdata.' + 
               datetime.datetime.today().strftime('%Y-%m-%d') + 
               '.' + arglist.experiment_name + '.' + str(count) + '/')
        exp_dir = os.path.join(arglist.save_dir, exp_dir_name)
        if os.path.exists(exp_dir):
            count += 1
        else:
            os.makedirs(exp_dir)
            break

    # set save_dir and plots_dir
    arglist.save_dir = exp_dir + 'policy/'
    arglist.plots_dir = exp_dir

    # make notes.md
    with open(exp_dir+'notes.md', 'w+') as notes:
        # record command and commit of experiment
        notes.write('Call: python ')
        for arg in sys.argv:
            notes.write(arg + ' ')
        notes.write('\n')
        notes.write('Commit: {}'.format(git_hash))

    # redefine stderr to capture errors during experiment and also output to terminal
    sys.stderr = Logger(exp_dir+'error.log')


def train(arglist):
    with U.single_threaded_session():

        training_start_time = time.perf_counter()

        # Create environment
        env = make_env(arglist.environment, arglist.scenario, arglist, 
                        benchmark=arglist.benchmark,
                        discrete_action_space=rl_algs.use_discrete_action_space(arglist.training_algorithm),
                        legacy_multidiscrete=rl_algs.use_legacy_multidiscrete(arglist.training_algorithm))

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers, group_trainer = get_agent_trainers(env, num_adversaries, obs_shape_n, arglist)

        # Create storage for training loss data useful for post processing and debugging
        training_loss_names = ['episode_count']
        if hasattr(group_trainer, 'mb_loss_names'):
            training_loss_names = training_loss_names + group_trainer.mb_loss_names

        # Initialize
        U.initialize()

        # Setup learning experiment recording
        if arglist.record_experiment:
            setup_experiment_record(arglist)

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if (arglist.display or arglist.restore or arglist.benchmark):
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        episode_agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        episode_rewards_stats = []  # sum of rewards for training curve
        episode_agent_rewards_stats = []  # agent rewards for training curve
        training_loss_stats = []
        arglist_filename = arglist.plots_dir + arglist.experiment_name + '.arglist.pkl'
        episode_rewards_filename = arglist.plots_dir + arglist.experiment_name + '.rewards.pkl'
        training_loss_filename = arglist.plots_dir + arglist.experiment_name + '.losses.pkl'
        episode_rewards_stats_filename = arglist.plots_dir + arglist.experiment_name + '.rewards_stats.pkl'.format(arglist.save_rate)
        episode_agent_rewards_stats_filename = arglist.plots_dir + arglist.experiment_name + '.rewards_per_agent_stats.pkl'.format(arglist.save_rate)
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        # capture joint state of all entities in world in case of use of centralized critic
        if "central_joint_" in arglist.critic_type:
            joint_state = env.get_joint_state()

        print('Starting iterations...')
        while True:

            # record joint state of system for (training purposes only, not accessible by agents at runtime)
            if "central_joint_" in arglist.critic_type:
                group_trainer.record_joint_state(joint_state)

            # get action, value estimate, and action probability from each agent's trainer
            action_n, value_n, neglogp_action_n, health_n = get_trainer_actions(env.agents, trainers, obs_n,
                combined_action_value=rl_algs.use_combined_action_value(arglist.training_algorithm))

            # ensure actions are valid
            if any([any(np.isnan(aa)) for aa in action_n]):
                raise Exception("NaN actions returned by get_trainer_actions: obs_n={}\naction_n={}".format(obs_n, action_n))

            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            # capture new joint state to be recorded at next time step
            if "central_joint_" in arglist.critic_type:
                joint_state = env.get_joint_state()


            # collect experience and store in the replay buffer
            for i, agent in enumerate(trainers):
                if rl_algs.use_combined_action_value(arglist.training_algorithm):
                    agent.experience(   obs=obs_n[i], 
                                        act=action_n[i], 
                                        rew=rew_n[i], 
                                        new_obs=new_obs_n[i], 
                                        val=value_n[i], 
                                        neglogpact=neglogp_action_n[i], 
                                        done=done_n[i], 
                                        health=health_n[i],
                                        terminal=terminal)

                else:
                    agent.experience(   obs=obs_n[i], 
                                        act=action_n[i], 
                                        rew=rew_n[i], 
                                        new_obs=new_obs_n[i], 
                                        done=done_n[i], 
                                        terminal=terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                episode_agent_rewards[i][-1] += rew

            if done or terminal:

                # before reseting environment, capture last joint state
                if "central_joint_" in arglist.critic_type:
                    group_trainer.record_joint_state(env.get_joint_state())

                # reset environment and appropriate variables
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in episode_agent_rewards:
                    a.append(0)
                agent_info.append([[]])

                # capture new joint state after reset to be recorded at next iteration
                if "central_joint_" in arglist.critic_type:
                    joint_state = env.get_joint_state()

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    filename = arglist.benchmark_dir + arglist.experiment_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(filename, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.05)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            if group_trainer is None:
                loss_stats = None
                # single trainer per agent, individualized learning
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss_stats = agent.update(trainers, train_step)
            
            else:
                # group-wide learning
                loss_stats = group_trainer.update_group_policy(terminal)

            if loss_stats is not None:
                ep_num = len(episode_rewards)
                training_loss_stats += [[ep_num] + L for L in loss_stats]


            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):

                # tensorize group policy
                # NOTE: this is not done at every time step because it appears 
                # to be very slow is very slow
                # NOTE: This is a bit of a hack since ScenarioHeuristicGroupTrainer doesn't 
                # inherently store policy as tf Tensors. This will likely be removed
                # later assuming ScenarioHeuristicGroupTrainer moves to tf-centric format
                if group_trainer is not None:
                    group_trainer.tensorize_group_policy()
                
                # save policy (tensorflow variables)
                U.save_state(arglist.save_dir, saver=saver)

                # record learning
                recent_reward_stats=(np.mean(episode_rewards[-arglist.save_rate:]), np.std(episode_rewards[-arglist.save_rate:]), time.perf_counter()-training_start_time)
                episode_rewards_stats.append(recent_reward_stats)
                for rew in episode_agent_rewards:
                    episode_agent_rewards_stats.append((np.mean(rew[-arglist.save_rate:]), np.std(rew[-arglist.save_rate:])))
                pickle_learning_curves(arglist_filename, episode_rewards_filename, episode_rewards_stats_filename, episode_agent_rewards_stats_filename, training_loss_filename, 
                                        arglist, episode_rewards, episode_rewards_stats, episode_agent_rewards_stats, (training_loss_names, training_loss_stats))

                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, std episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), recent_reward_stats[0], round(recent_reward_stats[1], 3), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {},  mean episode reward: {}, std episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), recent_reward_stats[0], round(recent_reward_stats[1], 3),
                        [np.mean(rew[-arglist.save_rate:]) for rew in episode_agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                pickle_learning_curves(arglist_filename, episode_rewards_filename, episode_rewards_stats_filename, episode_agent_rewards_stats_filename, training_loss_filename, 
                                        arglist, episode_rewards, episode_rewards_stats, episode_agent_rewards_stats, (training_loss_names, training_loss_stats))
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

def pickle_learning_curves(arglist_filename, episode_rewards_filename, episode_rewards_stats_filename, episode_agent_rewards_stats_filename, training_loss_filename, 
                            arglist, episode_rewards, episode_rewards_stats, episode_agent_rewards_stats, training_loss_names_and_stats):
    ''' record learning data into pickle file
    Args:
     - episode_rewards: net rewards per episode
     - episode_rewards_stats: net rewards per episode averaged over batch of episodes
     - episode_agent_rewards_stats: per-agent rewards averaged over batch of episodes
    '''
    with open(arglist_filename, 'wb') as fp:
        pickle.dump(arglist, fp)
    with open(episode_rewards_filename, 'wb') as fp:
        pickle.dump(episode_rewards, fp)
    with open(episode_rewards_stats_filename, 'wb') as fp:
        pickle.dump(episode_rewards_stats, fp)
    with open(episode_agent_rewards_stats_filename, 'wb') as fp:
        pickle.dump(episode_agent_rewards_stats, fp)
    with open(training_loss_filename, 'wb') as fp:
        pickle.dump(training_loss_names_and_stats, fp)

class OrderingException(Exception):
    ''' Custom exception to test if things are ordered as desired
    Notes:
        empty class still useful for unit testing
    '''
    pass

class Logger(object):
    def __init__(self, errlog_filename):
        self.terminal = sys.stderr
        # self.errlog_filename = "/tmp/ergo_error_log.log"
        self.errlog = open(errlog_filename, "w+")

    def write(self, message):
        self.terminal.write(message)
        self.errlog.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

if __name__ == '__main__':

    arglist = parse_args(sys.argv[1:])
    train(arglist)


