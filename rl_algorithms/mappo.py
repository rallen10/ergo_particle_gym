import numpy as np
from rl_algorithms.baselines_agent_trainer import BaselinesAgentTrainer
from rl_algorithms.group_trainer import GroupTrainer
import random
import tensorflow as tf
from copy import deepcopy
from rl_algorithms.mclearning import softmax
import maddpg.common.tf_util as U

# we need to hack the sys.path for imports
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'baselines'))
from baselines.ppo2.model import Model
from baselines.common.policies import build_policy
from baselines.common import explained_variance

_EPS = 1e-5
_CREDIT_SCALE_CONST = np.arctanh(0.5)


_VALID_CRITIC_TYPES = ( "distributed_local_observations",
                        "central_joint_observations",
                        "central_joint_state")

_VALID_CREDITING_ALGORITHMS = (
    "terminated_baseline",
    "batch_mean_deviation_heuristic")

# shared_reward, use_centralized_critic, critic_type, crediting_algorithm
_VALID_OPTION_COMBINATIONS = (
    (False, False,  "distributed_local_observations",   None),
    (True,  False,  "distributed_local_observations",   None),
    (True,  False,  "distributed_local_observations",   "batch_mean_deviation_heuristic"),
    (True,  True,   "central_joint_observations",       None),
    (True,  True,   "central_joint_state",              None),
    (True,  True,   "central_joint_observations",       "batch_mean_deviation_heuristic"),
    (True,  True,   "central_joint_state",              "batch_mean_deviation_heuristic"),
    (True,  True,   "central_joint_observations",       "terminated_baseline"),
    (True,  True,   "central_joint_state",              "terminated_baseline"),
)

# length of subvector of each entity in joint state
# see mager/environment.py:get_joint_state()
_DEFAULT_JOINT_STATE_ENTITY_LEN = 5

def constfn(val):
    def f(_):
        return val
    return f

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

class UpdateException(Exception):
    ''' Custom exception to test if things are being updated properly
    Notes:
        empty class still useful for unit testing
    '''
    pass

class HealthException(Exception):
    ''' Custom exception to test if things are being updated properly
    Notes:
        empty class still useful for unit testing
    '''
    pass

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def central_critic_network(*, inputs_placeholder_n, v_func, optimizer, scope, num_units, grad_norm_clipping=None, reuse=None):
    ''' value function network that estimates value based on joint state of system
    Args:
     - inputs_placeholder_n     TensorFlow placeholders for the input variables (either states or observations)
     - v_func                   computational graph for the value network
    '''
    with tf.variable_scope(scope, reuse=reuse):

        # set up placeholders
        in_ph_n = inputs_placeholder_n
        returns_target_ph = tf.placeholder(tf.float32, [None], name="target")
        old_v_pred_ph = tf.placeholder(tf.float32, [None], name="old_value_predictions")
        cliprange_ph = tf.placeholder(tf.float32, [])

        v_input = tf.concat(in_ph_n, 1)
        v_func_scope = scope + "_v_func"
        v_pred = v_func(v_input, 1, scope=v_func_scope, num_units=num_units)[:,0]
        v_func_vars = U.scope_vars(U.absolute_scope_name(v_func_scope))

        # Use cliped loss function to prevent wild variations
        v_pred_clipped = old_v_pred_ph + tf.clip_by_value(v_pred - old_v_pred_ph, -cliprange_ph, cliprange_ph)
        v_losses1 = tf.square(v_pred - returns_target_ph)
        v_losses2 = tf.square(v_pred_clipped - returns_target_ph)
        v_loss = .5 * tf.reduce_mean(tf.maximum(v_losses1, v_losses2))
        # v_loss = tf.reduce_mean(tf.square(v - returns_target_ph))

        optimize_expr = U.minimize_and_clip(optimizer, v_loss, v_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=in_ph_n + [returns_target_ph] + [old_v_pred_ph] + [cliprange_ph], outputs=v_loss, updates=[optimize_expr])
        v_values = U.function(inputs=in_ph_n, outputs=v_pred)

        return v_values, train, {'v_values': v_values}

class PPOGroupTrainer(GroupTrainer):
    ''' 
    Learn a group-wide policy using proximal policy optimization

    Notes:
     - Enforces that all agents have the same individual policy, but actions are
        selected in a decentralized fashion using each agents private observations
     '''
    def __init__(self, *, n_agents, obs_space, act_space, n_steps_per_episode, 
        ent_coef, local_actor_learning_rate, vf_coef, num_layers, num_units, activation,
        cliprange, n_episodes_per_batch, shared_reward, critic_type,
        central_critic_model, central_critic_learning_rate, central_critic_num_units, 
        joint_state_space_len, max_grad_norm, 
        n_opt_epochs, n_minibatches,
        crediting_algorithm=None, joint_state_entity_len=None):
        '''
        obs_space:      gym Space describing a single agent's observation space (assumes homogeneous agents in group)
        act_space:      gym Space describing a single agent's action space (assumes homogeneous agents in group)
        n_steps_per_episode: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                          nenv is number of environment copies simulated in parallel)

        ent_coef: float                   policy entropy coefficient in the optimization objective

        local_actor_learning_rate: float or function    learning rate for policy trainging, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                          training and 0 is the end of the training.

        vf_coef: float                    value function loss coefficient in the optimization objective

        max_grad_norm: float or None      gradient norm clipping coefficient
        
        n_opt_epochs: int                   number of training epochs within an update cycle

        cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

        shared_reward: boolean          All agents receive identical rewards at each time step

        critic_type: string             description of critic ['distributed_local_observations', 'central_joint_observations', 'central_joint_state']

        central_critic_model: TensorFlow network    model for constructing central value function network (i.e. critic)

        central_critic_learning_rate: float         learning rate used for training central value network (i.e. critic)

        central_critic_num_units: float             number of units in each layer of central critic

        joint_state_space_len: int      total length of joint state space. e.g. (pos+vel+comm)*n_agents + (pos+vel)*n_landmarks

        n_episodes_per_batch: int       number of complete episodes to collect a training batch run before executing a training cycle

        n_minibatches: int              number of minibatches to be drawn from a training batch

        crediting_algorithm: string     desciption of crediting algorithm to be used

        joint_state_entity_len: int     length of subvector a single entity occupies in the joint state vector
        '''

        # check learning rate are callable even though we may
        # mostly use them as constants
        if isinstance(local_actor_learning_rate, float): local_actor_learning_rate = constfn(local_actor_learning_rate)
        else: assert callable(local_actor_learning_rate)
        if isinstance(cliprange, float): cliprange = constfn(cliprange)
        else: assert callable(cliprange)

        # check integer inputs
        assert isinstance(n_opt_epochs, int)
        assert isinstance(n_steps_per_episode, int)
        assert isinstance(n_episodes_per_batch, int)
        assert isinstance(n_minibatches, int)
        assert isinstance(n_agents, int)

        self.n_opt_epochs = n_opt_epochs
        # self.mb_loss_vals = []
        self.local_actor_learning_rate = local_actor_learning_rate
        self.cliprange = cliprange
        self.n_agents = n_agents
        self.n_steps_per_episode = n_steps_per_episode
        self.n_episodes_per_batch = n_episodes_per_batch
        self.n_minibatches = n_minibatches 
        self.debug_count = 0
        self.shared_reward = shared_reward


        # format batch data containers
        self.n_data_per_batch = n_agents * n_steps_per_episode * n_episodes_per_batch
        self.n_joint_data_per_batch = (n_steps_per_episode+1) * n_episodes_per_batch    # Joint observations, states, and returns all have terminal state (thus the +1) but returns enforced to be 0
        self.n_data_per_minibatch = self.n_data_per_batch // n_minibatches
        self.n_joint_data_per_minibatch = self.n_joint_data_per_batch // n_minibatches
        assert self.n_data_per_minibatch <= self.n_data_per_batch
        assert self.n_data_per_minibatch > 0

        # most recent batch: per agent
        self.batch_observations = []
        self.batch_returns = []         # running list of returns from each agent's trajectory: len= n_data_per_batch
        self.batch_effective_returns = [] # effective_returns==returns if no crediting algorithm used
        self.batch_dones = []
        self.batch_actions = [] 
        self.batch_factual_values = []
        self.batch_counterfactual_values = []
        self.batch_counterfactual_advantages = []
        self.batch_effective_values = [] # effective_values==values if no crediting algorithm used
        self.batch_neglogp_actions = []
        self.batch_healths = []
        self.batch_effective_advantages = []

        # most recent batch: joint
        self.batch_joint_observations_stamped = []
        self.batch_joint_state_stamped = []
        self.batch_joint_returns = []   # running list of returns from group w shared_reward: len = n_joint_data_per_batch
        self.batch_joint_factual_values = []

        # data from most recent episode
        self.episode_joint_state = []   # running list of joint state from a given episode
        self.joint_state_labels = None

        # replay buffer
        # NOTE: I'm not sure that replay buffers are at all appropriate for PPO algorithm, but testing this out
        self.replay_buffer_enabled = False
        self.replay_buffer_size = self.n_joint_data_per_batch
        self.replay_buffer_joint_observations_stamped = [None]*self.replay_buffer_size
        self.replay_buffer_joint_state_stamped = [None]*self.replay_buffer_size
        self.replay_buffer_joint_returns = [None]*self.replay_buffer_size
        self.replay_buffer_index = 0
        self.replay_buffer_filled = False
        self.replay_buffer_add_probability = 0.1

        # keep count of number of times training updates are performed 
        self.training_update_count = 0

        # create dummy environment object to pass obs and act info to build_policy
        # i.e. I don't want to modify build_policy
        dummy_env = type('EmptyObj', (), {})()
        dummy_env.action_space = act_space
        dummy_env.observation_space = obs_space

        # create policy_fn that returns a PolicyWithValue object
        policy_fn = build_policy(env=dummy_env, policy_network='mlp', value_network='copy', 
            activation=getattr(tf.nn, activation), num_layers=num_layers, num_hidden=num_units)

        # create policy model from baselines/ppo2/model.py
        self.local_actor_critic_model = Model( policy=policy_fn, 
                        ob_space=obs_space, 
                        ac_space=act_space, 
                        nbatch_act=1, 
                        nbatch_train=self.n_data_per_minibatch,
                        nsteps=self.n_data_per_batch, 
                        ent_coef=ent_coef, 
                        vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm)

        # set options for critic and credit assignment and create necessary objects to call and train the value function
        self.crediting_algorithm = crediting_algorithm
        self.critic_type = critic_type
        self.joint_state_entity_len =_DEFAULT_JOINT_STATE_ENTITY_LEN if joint_state_entity_len is None else joint_state_entity_len

        self.use_centralized_critic = False
        if self.critic_type == "distributed_local_observations":
            pass

        elif self.critic_type == "central_joint_observations":
            assert self.shared_reward, "Invalid use of critic_type {}. Cannot be used when shared_reward is {}".format(self.critic_type, self.shared_reward)
            self.use_centralized_critic = True

            # Create joint observation placeholder that is stamped with the time to handle the 
            # tight temporal dependence of reward
            assert len(obs_space.shape) == 1 # check that there isn't some unexpected observation space
            joint_observation_stamped_shape = (self.n_agents*obs_space.shape[0]+1,)
            joint_observation_stamped_ph = [U.BatchInput(joint_observation_stamped_shape, name="joint_observation").get()]

            # create central critic network with functions for estimating value, training, updating network, and debugging
            self.central_vf_value, self.central_vf_train, self.central_vf_debug = central_critic_network(
                inputs_placeholder_n=joint_observation_stamped_ph,
                v_func=central_critic_model,
                optimizer=tf.train.AdamOptimizer(learning_rate=central_critic_learning_rate),
                scope = "joint_observation_critic",
                num_units=central_critic_num_units,
                grad_norm_clipping=max_grad_norm
            )

        elif self.critic_type == "central_joint_state":
            assert self.shared_reward, "Invalid use of critic_type {}. Cannot be used when shared_reward is {}".format(self.critic_type, self.shared_reward)
            self.use_centralized_critic = True
            assert joint_state_space_len%self.joint_state_entity_len == 0, "Something is wrong. The joint state space length should be dividable into length of individual entity states"

            # Create joint state placeholder that is stamped with the time to handle the 
            # tight temporal dependence of reward
            joint_state_stamped_ph = [U.BatchInput((1+joint_state_space_len, ), name="joint_state").get()]

            # create central critic network with functions for estimating value, training, updating network, and debugging
            self.central_vf_value, self.central_vf_train, self.central_vf_debug = central_critic_network(
                inputs_placeholder_n=joint_state_stamped_ph,
                v_func=central_critic_model,
                optimizer=tf.train.AdamOptimizer(learning_rate=central_critic_learning_rate),
                scope = "joint_state_critic",
                num_units=central_critic_num_units,
                grad_norm_clipping=max_grad_norm
            )

        else:
            raise Exception('Unrecognized critic type description: {}'.format(self.critic_type))
        

        self.mb_loss_names = ['training_update_count', 'epoch'] + self.local_actor_critic_model.loss_names + ['central_value_loss', 'central_value_explained_variance']
        self.sess = U.get_session()
        self.check_option_combinations()

    def check_option_combinations(self):
        ''' ensure consistency between options (e.g. central critic, crediting, shared rewards)'''

        # check and critic type and credition algorithms are valid 
        assert self.crediting_algorithm in _VALID_CREDITING_ALGORITHMS or self.crediting_algorithm is None
        assert self.critic_type in _VALID_CRITIC_TYPES
        
        # check that combination of shared_reward, use_centralized_critic, critic_type, and crediting_algorithm is valid
        opt_combo = (self.shared_reward, self.use_centralized_critic, self.critic_type, self.crediting_algorithm)
        if opt_combo not in _VALID_OPTION_COMBINATIONS:
            raise Exception("Invalid option combination:\n" + 
                "shared_reward={}, use_centralized_critic={}, critic_type={}, crediting_algorithm={}".format(
                    opt_combo[0], opt_combo[1], opt_combo[2], opt_combo[3]))


    def update_agent_trainer_group(self, trainers):
        '''
        Since individual agents depend on group trainer existing first to provide a model
        we must update the group agent trainer list after they've been created
        '''
        self.agent_trainer_group = trainers
        assert len(self.agent_trainer_group) == self.n_agents

    def record_joint_state(self, joint_state):
        ''' record joint/global/centralized state of all entities in the world
        Notes:
         - this cannot be used at execution time because it assumes a perfect knowledge all agent and landmark states
         - however it can be used for training since it is assumed that we are always training offline
        '''
        self.episode_joint_state.append(np.concatenate(joint_state["state"], axis=0))

        if self.joint_state_labels is None:
            self.joint_state_labels = joint_state["labels"]
        else:
            if self.joint_state_labels != joint_state["labels"]:
                raise Exception("Something has gone wrong, joint state labels should not change")
            # assert self.joint_state_labels == joint_state["labels"], "Something has gone wrong, joint state labels should not change"

    def update_group_policy(self, terminal):
        ''' decide if policy update to be executed and execute it
        Notes:
         - Assumes that policy rollout timesteps is equal to an episode
            length sinc we do not assume polices/experiences can be
            communicated between agents during a run
         - Assumes a minibatch is composed of all agents' experiences
            over an episode, i.e. M = N*T
        '''

        # no policy updates occur on non-terminal time steps
        # because currently it is assumed that agents cannot
        # communicate thier experiences and policies. Therefore
        # learning only occurs between episodes
        if not terminal:
            return

        # Perform multi-agent credit assignment on single episode
        episode_factual_values, episode_counterfactual_values = self.process_episode_value_centralization_and_credit_assignment()

        # consolidate individual private batches into group batch at end of episode
        self.process_episode_returns_and_store_group_training_batch(
            episode_factual_values=episode_factual_values, 
            episode_counterfactual_values=episode_counterfactual_values)

        # clear episode data from group and each agent in prep for next episode
        self.process_episode_clear_data()


        # check if batch is full an whether or not to update
        if len(self.batch_observations) < self.n_data_per_batch:
            return None

        # Perform multi-agent credit assignment on entire batch
        self.batch_credit_assignment()

        # Perform training by calculating and applying gradients
        return self.execute_group_training()

    def process_episode_value_centralization_and_credit_assignment(self):
        ''' Computer centralized values, assigning multi-agent credit if necessary
        '''
        # Centralize/aggregate value estimates if rewards are shared
        # NOTE: THIS IS EXPERIMENTAL!
        # The logic is that the values of each observation
        # of the same time step should essentially be the same
        # since they are just taking partial observations of the same state
        assert self.n_agents == len(self.agent_trainer_group)
        factual_values = None
        counterfactual_values = None
        joint_observations_stamped = [None]*(self.n_steps_per_episode+1)
        joint_state_stamped = [None]*(self.n_steps_per_episode+1)


        # calculate and record centralized values if rewards are shared
        if self.use_centralized_critic:
            assert self.shared_reward

            # establish central value data structures
            factual_values = [None]*(self.n_steps_per_episode+1)
            counterfactual_values = [None]*self.n_agents
            for agi in range(self.n_agents):
                counterfactual_values[agi] = [None]*(self.n_steps_per_episode+1)

            # Walkthrough most recent episode data, record as need, and clear 
            for i in range(self.n_steps_per_episode+1):


                if self.critic_type == "central_joint_observations":

                    # iterate over all agents observations to generate joint observation, stamping with time step
                    jso = [self.n_steps_per_episode+1-i]
                    for ag in self.agent_trainer_group:
                        jso.extend(ag.mbi_observations[i])
                    joint_observations_stamped[i] = np.array(jso)
                    factual_values[i] = self.central_vf_value(np.expand_dims(joint_observations_stamped[i],axis=0))[0]

                    # estimate value of joint observation, applying crediting if applicable
                    if self.crediting_algorithm is None:
                        pass

                    elif self.crediting_algorithm in ['batch_mean_devation_heuristic']:
                        # handle this at the batch level, not the episodic level
                        pass

                    else:
                        raise NotImplementedError("No current implementation of critic_type={} and crediting_algorithm={} combination".format(
                            self.critic_type, self.crediting_algorithm)) # TODO: implement crediting for joint observations

                elif self.critic_type == "central_joint_state":

                    # check that appropriate number of joint states have been recorded
                    assert len(self.episode_joint_state) == self.n_steps_per_episode+1

                    # stamp joint state
                    # rational: states are stamped with the time the state was achieved
                    # because it has a very strong influence on the RETURNS (not necessarily the immediate rewards)
                    joint_state_stamped[i] = np.concatenate(([self.n_steps_per_episode+1-i], self.episode_joint_state[i]))
                    factual_values[i] = self.central_vf_value(np.expand_dims(joint_state_stamped[i],axis=0))[0]

                    # estimate value of joint state, applying crediting if applicable
                    if self.crediting_algorithm is None:
                        pass

                    elif self.crediting_algorithm in ['batch_mean_devation_heuristic']:
                        # handle this at the batch level, not the episodic level
                        pass

                    elif self.crediting_algorithm in ["terminated_baseline"]:

                        # ensure that episode_joint_state stored as single array per timestep of expected size
                        # assumed length 5: [pos_x, pos_y, vel_x, vel_y, terminated/hazard]
                        assert len(joint_state_stamped[i]) == 1 + self.joint_state_entity_len*len(self.joint_state_labels)

                        # for each agent, compute counterfactual state and baseline
                        for ag_ind, ag in enumerate(self.agent_trainer_group):

                            # find agent in joint state and create counterfactual state
                            ag_jss_term_ind = 1 + (self.joint_state_labels.index("agent_{}".format(ag_ind))+1)*self.joint_state_entity_len - 1
                            assert 0.0 <= joint_state_stamped[i][ag_jss_term_ind] <= 1.0, "Something went wrong. Termination state must be between 0 and 1"
                            counterfactual_joint_state_stamped = deepcopy(joint_state_stamped[i])
                            counterfactual_joint_state_stamped[ag_jss_term_ind] = 1 # set terminated field to true

                            # estimate counterfactual value
                            counterfactual_values[ag_ind][i] = self.central_vf_value(np.expand_dims(counterfactual_joint_state_stamped,axis=0))[0]

                    else:
                        raise NotImplementedError("No current implementation of critic_type={} and crediting_algorithm={} combination".format(
                            self.critic_type, self.crediting_algorithm))

                else:
                    raise Exception("Invalid critic_type {} when use_centralized_critic is {}".format(self.critic_type, self.use_centralized_critic))

            # enforce that values are zero at final, terminal state (see Sutton 2nd ed. Eqn 3.9)
            factual_values[-1] = 0.0
            # for agi in range(self.n_agents):
            #     counterfactual_values[agi][-1] = 0.0

        # record episode's stamped observations and state in longer term batch
        self.batch_joint_observations_stamped.extend(joint_observations_stamped)
        self.batch_joint_state_stamped.extend(joint_state_stamped)

        return factual_values, counterfactual_values

    def process_episode_returns_and_store_group_training_batch(self, episode_factual_values, episode_counterfactual_values):
        ''' Calculate returns on each agent's episode data and store in group training batch
        Args:
         - episode_factual_values: None, [n_agents, n_steps_per_episode+1] baseline values to be used for calculating return, if applicable
         - episode_counterfactual_values: None, [n_agents, n_steps_per_episode+1] baseline values to be used for credit assignment, if applicable
        '''

        # Calculate advantages and returns, store in group minibatch, and
        # clear individual minibatches
        # see baselines/ppo2/runner.py: delta, mb_advs, mb_returns
        # batch_episode_returns = np.zeros(self.n_steps_per_episode)
        for trainer_ind, trainer in enumerate(self.agent_trainer_group):

            # extract individual agent's baseline values for advantage calculation, if applicable
            agent_factual_values = None
            agent_counterfactual_values = None
            if episode_factual_values is None:
                assert not self.use_centralized_critic, "centralized critic must provide baseline values"
            else:
                assert len(episode_factual_values) == self.n_steps_per_episode+1
                agent_factual_values = episode_factual_values
                assert len(episode_counterfactual_values[trainer_ind]) == self.n_steps_per_episode+1
                agent_counterfactual_values = episode_counterfactual_values[trainer_ind]

            # perform return and advantage calculation for a single agent over a single episode
            trainer.process_individual_agent_episode_returns_and_advantages(factual_values=agent_factual_values, counterfactual_values=agent_counterfactual_values)

            # Store individual agent's episode data in group training batch
            batch_data = trainer.format_individual_agent_episode_data_for_training_batch()
            self.batch_observations.extend(batch_data[0])
            self.batch_returns.extend(batch_data[1]) 
            self.batch_dones.extend(batch_data[2])
            self.batch_actions.extend(batch_data[3]) 
            self.batch_factual_values.extend(batch_data[4]) 
            self.batch_neglogp_actions.extend(batch_data[5])
            self.batch_counterfactual_values.extend(batch_data[6])
            self.batch_counterfactual_advantages.extend(batch_data[7])
            self.batch_healths.extend(batch_data[8])

            # Record episode returns to form statistics for credit assignment
            if self.shared_reward:
                if trainer_ind == 0:
                    self.batch_joint_returns.extend(deepcopy(batch_data[1]))
                    self.batch_joint_returns.extend([0.0])  # Enforce return at final step is defined to be zero (See Sutton 2nd Ed. Eqn 3.9)
                    self.batch_joint_factual_values.extend(deepcopy(batch_data[4]))
                    self.batch_joint_factual_values.extend([0.0])  # Enforce return at final step is defined to be zero (See Sutton 2nd Ed. Eqn 3.9)
                else:
                    if self.use_centralized_critic:
                        # make sure return match between agents when reward shared and critic centralized
                        if not np.allclose(self.batch_joint_returns[-self.n_steps_per_episode-1:-1], batch_data[1], rtol=1e-4):
                            raise Exception("batch_joint_returns do not match across agents, even though reward shared\n\n"+
                                "batch_joint_returns of current episode:\n{}\n\n".format(self.batch_joint_returns[-self.n_steps_per_episode:]) + 
                                "batch_data from individual agent return calculation:\n{}\n\n".format(batch_data[1]))
                        if not np.allclose(self.batch_joint_factual_values[-self.n_steps_per_episode-1:-1], batch_data[4], rtol=1e-4):
                            raise Exception("batch_joint_factual_values do not match across agents, even though reward shared\n\n"+
                                "batch_joint_factual_values of current episode:\n{}\n\n".format(self.batch_joint_factual_values[-self.n_steps_per_episode:]) + 
                                "batch_data from individual agent return calculation:\n{}\n\n".format(batch_data[4]))
                    else:
                        # rewards shared but critic not centralized (i.e. different values), average of agent's returns
                        for jj in range(-self.n_steps_per_episode, -1):
                            self.batch_joint_returns[jj-1]=(self.batch_joint_returns[jj-1]*trainer_ind+batch_data[1][jj])/(float(trainer_ind+1))

                if episode_factual_values is not None:
                    # if using crediting, the batch values should match the episode_baseline_values in order for the
                    # advantage estimates to come out as intended
                    assert self.use_centralized_critic
                    assert np.allclose(self.batch_factual_values[-self.n_steps_per_episode:], agent_factual_values[:-1], rtol=1e-5)

                if episode_counterfactual_values is not None:
                    if episode_counterfactual_values[trainer_ind] is None or episode_counterfactual_values[trainer_ind][0] is None:
                        # skip this check since counterfactual baseline value not intended to be used
                        pass
                    else:
                        # if using crediting, the batch values should match the episode_baseline_values in order for the
                        # advantage estimates to come out as intended
                        assert self.use_centralized_critic
                        assert np.allclose(self.batch_counterfactual_values[-self.n_steps_per_episode:], agent_counterfactual_values[:-1], rtol=1e-5)


    def process_episode_clear_data(self):
        ''' clear episode-specific data from group and agent to prep for next episode '''

        for trainer in self.agent_trainer_group:
            # Clear individual agent's episode records
            trainer.clear_individual_agent_episode_data()

        # clear out groups episode data
        self.episode_joint_state = []
        

    def execute_group_training(self):
        ''' use calculate and apply graidients to policy and value
        Notes:
         - also distributes new policy to agents and clears batch data
        '''

        # ensure batch of data is of expected size
        assert len(self.batch_observations) == self.n_data_per_batch
        assert len(self.batch_effective_returns) == self.n_data_per_batch
        assert len(self.batch_dones) == self.n_data_per_batch
        assert len(self.batch_actions) == self.n_data_per_batch
        assert len(self.batch_effective_values) == self.n_data_per_batch
        assert len(self.batch_neglogp_actions) == self.n_data_per_batch
        assert len(self.batch_healths) == self.n_data_per_batch
        assert self.batch_effective_advantages is None or len(self.batch_effective_advantages) == self.n_data_per_batch
        assert len(self.batch_joint_observations_stamped) == self.n_joint_data_per_batch
        assert len(self.batch_joint_state_stamped) == self.n_joint_data_per_batch
        if self.shared_reward:
            assert len(self.batch_joint_returns) == self.n_joint_data_per_batch
            assert len(self.batch_joint_factual_values) == self.n_joint_data_per_batch

        # store information in replay buffer (Note: I don't think replay buffers are appropriate in this context, just testing)
        if self.use_centralized_critic and self.replay_buffer_enabled:
            raise NotImplementedError("Replay buffers not appropriate for PPO. If you do want to test them, need to implement replay_buffer_factual_values")
            if self.critic_type == "central_joint_observations" or self.critic_type == "central_joint_state":
                # add to replay buffer with given probability
                for jbi in range(self.n_joint_data_per_batch):
                    if np.random.rand() < self.replay_buffer_add_probability:
                        self.replay_buffer_joint_observations_stamped[self.replay_buffer_index] = self.batch_joint_observations_stamped[jbi]
                        self.replay_buffer_joint_state_stamped[self.replay_buffer_index] = self.batch_joint_state_stamped[jbi]
                        self.replay_buffer_joint_returns[self.replay_buffer_index] = self.batch_joint_returns[jbi]
                        self.replay_buffer_index += 1
                        if self.replay_buffer_index == self.replay_buffer_size:
                            # enable buffer now that it has filled
                            self.replay_buffer_filled = True
                            self.replay_buffer_index = 0
                    
            else:
                raise Exception("Invalid critic_type {} when use_centralized_critic is {}".format(self.critic_type, self.use_centralized_critic))


        # Calculate learning rate and cliprange
        # Note: for now these are constants but could be updated
        # based on training iteration
        # See baselines/ppo2/ppo2.py
        pol_lrnow = self.local_actor_learning_rate(self.training_update_count)
        cliprangenow = self.cliprange(1.0)

        # Establish structures and accounting vars for training info/results
        # Index of each element of batch_size
        # Create the indices array corresponding to each agent's individual
        # minibatch
        mb_loss_stats = []
        self.training_update_count += 1 # different than train.py train_step
        inds = np.arange(self.n_data_per_batch)
        if self.replay_buffer_enabled and self.replay_buffer_filled:
            joint_inds = np.arange(self.n_joint_data_per_batch + self.replay_buffer_size)
        else:
            joint_inds = np.arange(self.n_joint_data_per_batch)
        n_data_per_batch_clipped = self.n_data_per_minibatch * self.n_minibatches
        n_joint_data_per_batch_clipped = self.n_joint_data_per_minibatch * self.n_minibatches

        # Call training operation for K epochs
        # see baselines/ppo2/ppo2.py: model.train
        for epch in range(self.n_opt_epochs):

            # run training on policy and private "local" value function (distributed_local_observations)
            # NOTE: this currently somewhat redudant/useless/inefficient if only concerned
            # with central critic. Left in for now because it provides a bit more info
            # and doesn't conflic with central critic trainingn (just inefficient)
            np.random.shuffle(inds)
            loss_stats = []
            for start in range(0, n_data_per_batch_clipped, self.n_data_per_minibatch):
                end = start + self.n_data_per_minibatch
                mbinds = inds[start:end]
                assert len(mbinds) == self.n_data_per_minibatch

                # run training on policy and private value function
                loss_stats.append(self.local_actor_critic_model.train(pol_lrnow, cliprangenow, 
                    np.asarray(self.batch_observations)[mbinds],
                    np.asarray(self.batch_effective_returns)[mbinds], # effective_returns==returns if no crediting algorithm used
                    np.asarray(self.batch_dones)[mbinds], 
                    np.asarray(self.batch_actions)[mbinds], 
                    np.asarray(self.batch_effective_values)[mbinds], # effective_values==values if no crediting algorithm used 
                    np.asarray(self.batch_neglogp_actions)[mbinds],
                    # np.asarray(self.batch_healths)[mbinds],
                    advantages = None if self.batch_effective_advantages is None else np.asarray(self.batch_effective_advantages)[mbinds]))


            # run training on central value function (if applicable)
            central_vf_loss_stats = [(None, None)]*self.n_minibatches
            if self.use_centralized_critic:

                # randomize minibatch data (i.e. make non-time-sequential) by shuffling indices
                np.random.shuffle(joint_inds)
                joint_minibatch_count = 0
                for start in range(0, n_joint_data_per_batch_clipped, self.n_joint_data_per_minibatch):

                    # establish randomized inidices of data for minibatch
                    end = start + self.n_joint_data_per_minibatch
                    joint_mbinds = joint_inds[start:end]
                    central_vf_loss = [None]*self.n_joint_data_per_minibatch
                    assert len(joint_mbinds) == self.n_joint_data_per_minibatch
                    central_critic_training_feed = [[], [], [], []]

                    # Run central value training on minibatch
                    for ji_count, ji in enumerate(joint_mbinds):

                        if self.critic_type=="central_joint_observations":
                            assert len(self.batch_joint_returns) == len(self.batch_joint_observations_stamped)
                            assert len(self.batch_joint_factual_values) == len(self.batch_joint_observations_stamped)
                            if ji >= self.n_joint_data_per_batch:
                                # pull from replay buffer
                                # NOTE: I'm not sure that replay buffers are at all appropriate for PPO algorithm, but 
                                # testing this out
                                raise NotImplementedError("Replay buffers not appropriate for PPO. If you do want to test them, need to implement replay_buffer_factual_values")
                                assert self.replay_buffer_enabled, "Something went wrong. This condition should not be entered if replay buffer is not used"
                                assert len(self.replay_buffer_joint_returns) == len(self.replay_buffer_joint_observation_stamped)
                                rpi = ji - self.n_joint_data_per_batch
                                # joint_return_feed.append(np.expand_dims(self.replay_buffer_joint_observations_stamped[rpi],axis=0), [self.replay_buffer_joint_returns[rpi]])
                                central_critic_training_feed[0].append(self.replay_buffer_joint_observations_stamped[rpi])
                                central_critic_training_feed[1].append(self.replay_buffer_joint_returns[rpi])
                            else:
                                # joint_return_feed.append(np.expand_dims(self.batch_joint_observations_stamped[ji],axis=0), [self.batch_joint_returns[ji]])
                                central_critic_training_feed[0].append(self.batch_joint_observations_stamped[ji])
                                central_critic_training_feed[1].append(self.batch_joint_returns[ji])
                                central_critic_training_feed[2].append(self.batch_joint_factual_values[ji])

                        elif self.critic_type == "central_joint_state":
                            assert len(self.batch_joint_returns) == len(self.batch_joint_state_stamped)
                            assert len(self.batch_joint_factual_values) == len(self.batch_joint_state_stamped)
                            if ji >= self.n_joint_data_per_batch:
                                # pull from replay buffer
                                # NOTE: I'm not sure that replay buffers are at all appropriate for PPO algorithm, but 
                                # testing this out
                                raise NotImplementedError("Replay buffers not appropriate for PPO. If you do want to test them, need to implement replay_buffer_factual_values")
                                assert self.replay_buffer_enabled, "Something went wrong. This condition should not be entered if replay buffer is not used"
                                assert len(self.replay_buffer_joint_returns) == len(self.replay_buffer_joint_state_stamped)
                                rpi = ji - self.n_joint_data_per_batch
                                # joint_return_feed.append(np.expand_dims(self.replay_buffer_joint_state_stamped[rpi],axis=0), [self.replay_buffer_joint_returns[rpi]])
                                central_critic_training_feed[0].append(self.replay_buffer_joint_state_stamped[rpi])
                                central_critic_training_feed[1].append(self.replay_buffer_joint_returns[rpi])
                            else:
                                # joint_return_feed.append(np.expand_dims(self.batch_joint_state_stamped[ji],axis=0), [self.batch_joint_returns[ji]])
                                central_critic_training_feed[0].append(self.batch_joint_state_stamped[ji])
                                central_critic_training_feed[1].append(self.batch_joint_returns[ji])
                                central_critic_training_feed[2].append(self.batch_joint_factual_values[ji])

                        else:
                            raise Exception("Invalid critic_type {} when use_centralized_critic is {}".format(self.critic_type, self.use_centralized_critic))


                    # Call training function and return losses
                    central_critic_training_feed[3] = cliprangenow
                    central_vf_loss = self.central_vf_train(*central_critic_training_feed)

                    # Take average of loss values over minibatch and calc explained variance
                    central_vf_expvar = explained_variance(self.central_vf_value(central_critic_training_feed[0]), np.asarray(central_critic_training_feed[1]))
                    central_vf_loss_stats[joint_minibatch_count] = (np.mean(central_vf_loss), central_vf_expvar)
                    joint_minibatch_count += 1
                assert joint_minibatch_count == self.n_minibatches

            # record training loss data for the epoch
            assert len(central_vf_loss_stats) == len(loss_stats) == self.n_minibatches
            for i in range(self.n_minibatches):
                mb_loss_stats.append([self.training_update_count, epch, *loss_stats[i], *central_vf_loss_stats[i]])


        # Distribute new policy to all agents and reset experiences (minibatchs)
        # see mclearning.py: group_policy_update
        for agent in self.agent_trainer_group:
            agent.group_policy_update(None)

        # Clear out group batch
        self.batch_observations = []
        self.batch_returns = []
        self.batch_effective_returns = []
        self.batch_dones = []
        self.batch_actions = [] 
        self.batch_factual_values = []
        self.batch_counterfactual_values = []
        self.batch_counterfactual_advantages = []
        self.batch_effective_values = []
        self.batch_neglogp_actions = []
        self.batch_healths = []
        self.batch_effective_advantages = []

        self.batch_joint_observations_stamped = []
        self.batch_joint_state_stamped = []
        self.batch_joint_returns = []
        self.batch_joint_factual_values = []

        return mb_loss_stats

    def batch_credit_assignment(self):
        ''' Calculate effective_values and effective_returns as a form of multi-agent credit assignment
        Args:
        Notes:
         - Since this work is experimental, the default is to NOT attempt to assign credits, instead
         just using the returns as credits thus all agents receive the same credit
        '''

        # if rewards aren't shared and each agent generates their own separate reward, 
        # then credit assignment is automatic, it is simply the returns in the batch
        crediting_info = None
        if self.crediting_algorithm is None or not self.shared_reward:
            self.batch_effective_returns = deepcopy(self.batch_returns)
            self.batch_effective_values = deepcopy(self.batch_factual_values)
            self.batch_effective_advantages = None

        elif self.crediting_algorithm in ['terminated_baseline']:
            # returns and values are multiplied by system health (assumed to be 0 or 1) to 
            # account for discrepancy between chosen actions and executed actions. If the 
            # agent is terminated (health=0) then the action chosen by agent is almost
            # guaranted to NOT equal the action executed, which is zero-action. Thus
            # we need to ensure that the policy does not learn anything based on this 
            # erroneous assumption that the chosen action affected the state of the system
            # and thus the returns. We can prevent any policy learning by ensuring the 
            # advantage is zero.
            # NOTE: it is vital that critic learning is NOT altered based on the health since
            # coming up with a counterfactual baseline relies on the critic being able to 
            # estimate the true value of a terminated state
            self.batch_effective_returns = deepcopy(self.batch_returns)
            self.batch_effective_values = deepcopy(self.batch_factual_values)
            advs = np.zeros_like(self.batch_effective_values)
            for i, h in enumerate(self.batch_healths):
                if not (np.isclose(h,0.0) or np.isclose(h,1.0)):
                    raise NotImplementedError("Current credit assignment only valid for binary health values")
                advs[i] = h*self.batch_counterfactual_advantages[i]

            # normalize advantages
            self.batch_effective_advantages = self.batch_healths * (advs - advs.mean()) / (advs.std() + 1e-8)

        elif self.crediting_algorithm in ['batch_mean_deviation_heuristic']:
            crediting_info = self.batch_mean_deviation_heuristic_credit_assignment()

        else:
            raise Exception('Unrecognized crediting assignment algorithm: {}'.format(self.crediting_algorithm))


        return crediting_info


    def batch_mean_deviation_heuristic_credit_assignment(self):
        ''' multi-agetn credit assignment heuristic that rewards credit based on deviation from mean rewards
        Notes:
         - From a large set of episodes, the mean and std return is calculated for each timestep.
         The further a given steps return in a given episode is from the mean, the more it weights
         the credit toward unlikely actions
         - effective_values=values, only effective_returns are modified
        '''

        # copy over values and leave unmodified
        self.batch_effective_values = deepcopy(self.batch_factual_values)
        self.batch_effective_advantages = None

        # Assign individual return credits based on returns relative to mean
        # raise NotImplementedError("batch_effective_returns re-interpretted and no longer used in training.\nInstead batch_factual_values -> batch_effective_values.\nNeeds to be updated here")
        self.batch_effective_returns = []
        batch_episode_returns = np.reshape(self.batch_joint_returns, (self.n_episodes_per_batch, self.n_steps_per_episode+1))
        batch_episode_returns = batch_episode_returns[:, :-1] # truncate to align with agent mbi which did not store the zero value
        returns_means = np.mean(batch_episode_returns, axis=0)
        returns_stds = np.std(batch_episode_returns, axis=0)
        batch_index = 0
        credit_scale = np.zeros((self.n_episodes_per_batch, self.n_steps_per_episode))
        for ep in range(self.n_episodes_per_batch):
            
            # For a given episode, form the credit scale at each time step using the
            # statistics of returns over all episodes
            cur_episode_credits = np.empty((0,self.n_agents))
            for step in range(self.n_steps_per_episode):
                
                # find number of standard devations return is from mean return
                if returns_stds[step] < _EPS:
                    return_delta = 0.0
                else:
                    return_delta = abs(batch_episode_returns[ep][step] - returns_means[step])/returns_stds[step]
                credit_scale[ep][step] = np.tanh(_CREDIT_SCALE_CONST * return_delta)

                # For each step in each episode, extract the probabilities of each agent's actions
                # to be used to credit assignment
                logp_actions = np.zeros(self.n_agents)  
                for ag_ind in range(self.n_agents):

                    # Since we've cleared out individual private batches from agent computers, we no longer
                    # have an explicit tie between batch data and the agent that generated it
                    # Use assertions as a loose check that ordering through batch has been maintained
                    batch_index = (ep*self.n_agents + ag_ind) * self.n_steps_per_episode + step
                    assert np.isclose(self.batch_returns[batch_index], batch_episode_returns[ep][step])
                    logp_actions[ag_ind] = -self.batch_neglogp_actions[batch_index]

                # Assign credit and check that it adds to episode rewards
                credit_weights = redistributed_softmax(logp_actions, credit_scale[ep][step])
                cur_step_credits = batch_episode_returns[ep][step] * credit_weights
                assert np.isclose(sum(cur_step_credits), batch_episode_returns[ep][step])

                # record credits for all agents in given step 
                cur_episode_credits = np.concatenate((cur_episode_credits, cur_step_credits[np.newaxis,:]))

            # record credits for given episode in batch
            self.batch_effective_returns.extend(cur_episode_credits.reshape(self.n_steps_per_episode*self.n_agents,order='F'))

        # Loose check that ordering of batch has been maintained
        assert batch_index == self.n_data_per_batch - 1

        return (returns_means, returns_stds, credit_scale)

    def tensorize_group_policy(self):
        '''Do nothing since policy is already a tf graph'''
        pass

class PPOAgentComputer(object):
    '''
    Computes actions and records experience for each individual agent. PPOGroupTrainer that is assigned as trainer for each individual agent
    '''
    def __init__(self, *, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func, lam=1.0):
        '''
        Args:
         - model: encompasses the action model (policy) and value function approximator
         - gamma: float                      discounting factor
         - lam: float                        advantage estimation discounting factor (lambda in the paper)

        Notes:
         - local_q_func is meaningless in this context but is kept to maintain uniformity between
            trainer instantiations
        '''
        self.name = name
        self.local_actor_critic_model = model
        self.agent_index = agent_index
        self.nsteps = args.max_episode_len
        self.gamma = args.gamma
        self.lam = lam

        # mbi = minibatch data for agent "i"
        self.mbi_observations = []
        self.mbi_actions = []
        self.mbi_rewards = []
        self.mbi_obs_values = []    # value estimates of private observation
        self.mbi_neglogp_actions = []
        self.mbi_dones = []
        self.mbi_returns = []
        self.mbi_credits = []
        self.mbi_factual_advantages = []
        self.mbi_counterfactual_advantages = []
        self.mbi_healths = []

    def action(self, obs):
        ''' return agent's action based on observation
        '''

        # actions, values, self.states, neglogpacs = self.local_actor_critic_model.step(self.obs, S=self.states, M=self.dones)
        # return actions
        act, val, _, neglogpact = self.local_actor_critic_model.step(obs)
        # return np.array([0] + self.local_actor_critic_model.step(obs) + self.local_actor_critic_model.step(obs))
        return (act.flatten(), val.flatten(), neglogpact.flatten())

    def experience(self, *, obs, act, rew, new_obs, val, neglogpact, done, health, terminal):
        ''' Store individual agent's experiences to be used to form a group minibatch at training
        Args:
         obs: observation
         act: action
         rew: reward recieved by agent
         new_obs: next observation received after action
         val: observation value estimate V(s)
         neglogpact: negative log probability of action
         done: boolean if agent is done in episode (other agents may not be done)
         health: health of agent
         terminal: boolean if episode is a terminal step
        Notes:
         obs and dones longer than other lists, 
         thus new_obs and done are at time=t+1
         where as obs, act, rew, val are at time=t
        '''

        # initilize observations and dones if experience is empty 
        # (i.e. after env reset)
        if len(self.mbi_observations) == 0:
            self.mbi_observations.append(obs)
            self.mbi_dones.append(0)

        # check size of agent's current minibatch is growing properly
        assert (len(self.mbi_observations) == len(self.mbi_dones) == 
                len(self.mbi_rewards) + 1)
        assert (len(self.mbi_rewards) == len(self.mbi_actions) == 
                len(self.mbi_obs_values) == len(self.mbi_neglogp_actions) ==
                len(self.mbi_healths))

        # record new experience
        self.mbi_observations.append(new_obs)       # record t+1
        self.mbi_actions.append(act)                # record t
        self.mbi_rewards.append(rew)                # record t
        self.mbi_obs_values.append(val)             # record t
        self.mbi_neglogp_actions.append(neglogpact) # record t
        self.mbi_dones.append(done)                 # record t+1
        self.mbi_healths.append(health)             # record t
        if terminal:
            # terminal implies done, but not vice versa
            # terminal state when final time in episode is reached
            # sets done to true so that advantage and return calculations are correct
            self.mbi_dones[-1] = True

    def process_individual_agent_episode_returns_and_advantages(self, factual_values, counterfactual_values):
        ''' Calculate returns and advantages from individual agent's episode
        Args:
         - factual_values [nsteps+1] Values used to compute returns. If None, mbi_obs_values used
         - counterfactual_values [nsteps+1] Values used as counterfactual baseline for credit assignment
        Notes:
         - This could be moved to "preupdate" if helps with compatibility train.py, but 
            currently uses different function for clarity of when functions are called
            and disambiguation
        '''
        
        # check that number of steps has been reached for individual agent's episode
        if not (self.nsteps 
            == len(self.mbi_rewards) 
            == len(self.mbi_actions) 
            == len(self.mbi_neglogp_actions)
            == len(self.mbi_healths)):
            raise UpdateException("Inproperly sized episode data: nsteps={} | mbi_rewards:{} | mbi_actions:{} | mbi_neglogp_actions:{} | mbi_healths".format(
                self.nsteps, len(self.mbi_rewards), len(self.mbi_actions), len(self.mbi_neglogp_actions), len(self.mbi_healths)))
        if not (self.nsteps + 1 
            == len(self.mbi_observations) 
            == len(self.mbi_dones)):
            raise UpdateException("Inproperly sized episode data: nsteps+1={} | mbi_observations:{} | mbi_dones:{} | mbi_healths:{}".format(
                self.nsteps+1, len(self.mbi_observations), len(self.mbi_dones)))

        # check that once done is reached in an episode, all subsequent steps are also done
        # It may be possible to "wrap" through multiple episode intervals, but 
        # currently that is not implemented
        # NOTE: I'm not sure this is necessary. If I am getting this error, I might want
        # to reconsider if I should have this here. If anything it acts like a warning
        # that the specific situation needs more investigation and thought than was
        # originally given
        # first_done = np.argmax(np.array(self.mbi_dones)>0)
        # if any(self.mbi_dones) and any([d < 1 for d in self.mbi_dones[first_done:]]):
        if not self.mbi_dones == sorted(self.mbi_dones):
            raise UpdateException("Invalid reversal on done state. mbi_dones={}".format(self.mbi_dones))

        if not self.mbi_healths == sorted(self.mbi_healths, reverse=True):
            raise HealthException("Health vector must be monotonically decreasing. mbi_health={}".format(self.mbi_health))

        
        # To maintain the assumption that experience between agents can only be shared
        # between episodes (i.e. no instaneous communication of experience during an episode)
        # that agent must be done a the last time step
        # Also to enforce assumption that returns from final time step are defined to be zero
        # (see Sutton Ch3, near eqn 3.9)
        # NOTE: this assumption should already be maintained at the group level enforcement
        # of terminal
        if not self.mbi_dones[-1]:
            raise UpdateException("Invalid episode termination. Current assumptions require that episode end with done=True. Only purely episodic (Monte Carlo) currently allowed")

        # convert episode buffers to arrays for later passing to training function
        self.mbi_observations = np.asarray(self.mbi_observations)
        self.mbi_rewards = np.asarray(self.mbi_rewards)
        self.mbi_actions = np.asarray(self.mbi_actions)
        self.mbi_obs_values = np.asarray(self.mbi_obs_values)
        self.mbi_neglogp_actions = np.asarray(self.mbi_neglogp_actions)
        self.mbi_dones = np.asarray(self.mbi_dones, dtype=np.bool)
        self.mbi_healths = np.asarray(self.mbi_healths)

        # use decentralized or centralized values
        if factual_values is None:
            assert len(self.mbi_obs_values) == self.nsteps
            self.mbi_factual_values = np.asarray(deepcopy(self.mbi_obs_values))
            self.mbi_factual_values = np.concatenate((self.mbi_factual_values, [0.0]))
            # last_values = self.local_actor_critic_model.value(self.mbi_observations[-1], M=self.mbi_dones[-1]) # I'm not sure about M, see baselines/ppo2/runner.py last_values
        else:
            assert len(factual_values) == self.nsteps + 1
            self.mbi_factual_values = np.asarray(deepcopy(factual_values))
            # last_values = baseline_values[-1]


        # For consistency, enforce that values, returns and advantages are of length n+1, but the final
        # value is 0 for each
        if not len(self.mbi_factual_values) == self.nsteps+1:
            raise UpdateException("Unexpected length of mbi_factual_values: {} for nsteps={}".format(len(self.mbi_factual_values), self.nsteps))
        if not np.isclose(self.mbi_factual_values[-1], 0.0):
            raise UpdateException("Current implementation enforces assumption that final state of episode is terminal with 0 value and return. Invalid final value: {}".format(self.mbi_factual_values[-1]))
        self.mbi_returns = np.zeros_like(self.mbi_factual_values)
        self.mbi_factual_advantages = np.zeros_like(self.mbi_factual_values)
        # self.mbi_counterfactual_advantages = np.zeros_like(self.mbi_counterfactual_values)

        # TODO: determine if rewards are type int, returns and advantages will be cast to ints at end
        # of calculation

        lastgaelam_factual = 0.0
        lastgaelam_counterfactual = 0.0

        for t in reversed(range(self.nsteps)):
            nextnonterminal = 1.0 - self.mbi_dones[t+1]
            nextvalues = self.mbi_factual_values[t+1]

            # compute factual advantages
            delta_factual = self.mbi_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.mbi_factual_values[t]
            self.mbi_factual_advantages[t] = lastgaelam_factual = delta_factual + self.gamma * self.lam * nextnonterminal * lastgaelam_factual

            # compute counterfactual advantages
            # if self.mbi_counterfactual_values[t] is not None:
            #     delta_counterfactual = self.mbi_healths[t] * (self.mbi_rewards[t] + self.gamma * nextvalues * nextnonterminal - self.mbi_counterfactual_values[t])
            #     self.mbi_counterfactual_advantages[t] = lastgaelam_counterfactual = delta_counterfactual + self.gamma * self.lam * nextnonterminal * lastgaelam_counterfactual
            # else:
            #     self.mbi_counterfactual_advantages[t] = None

        # compute returns from each state in individual agent's episode
        assert self.mbi_factual_advantages.shape == self.mbi_factual_values.shape
        self.mbi_returns = self.mbi_factual_advantages + self.mbi_factual_values
        # store counterfactual values and calc counterfactual advantages for batch training later
        if counterfactual_values is None or any(cv is None for cv in counterfactual_values):
            self.mbi_counterfactual_values = np.asarray([None]*(self.nsteps + 1))
            self.mbi_counterfactual_advantages = np.asarray([None]*(self.nsteps + 1))
        else:
            assert len(counterfactual_values) == self.nsteps + 1
            self.mbi_counterfactual_values = np.asarray(deepcopy(counterfactual_values))
            self.mbi_counterfactual_advantages = self.mbi_returns - self.mbi_counterfactual_values
        assert self.mbi_counterfactual_advantages.shape == self.mbi_counterfactual_values.shape  
        assert np.isclose(self.mbi_returns[-1], 0.0)
        assert np.isclose(self.mbi_factual_advantages[-1], 0.0)

    def format_individual_agent_episode_data_for_training_batch(self):
        ''' format mbi lists such that it can be passed to model.train
        Notes:
         - since dones and observations have an extra element, just chopping of the
            end. Need to think more/test to make sure this is the right move...
        '''

        # ensure that individual episode has already been formatted as
        # arrays of appropriate size
        assert isinstance(self.mbi_observations, np.ndarray)
        assert len(self.mbi_observations.shape) == 2
        assert isinstance(self.mbi_observations, np.ndarray)

        return (self.mbi_observations[:-1], 
                self.mbi_returns[:-1], 
                self.mbi_dones[:-1], 
                self.mbi_actions, 
                self.mbi_factual_values[:-1], 
                self.mbi_neglogp_actions,
                self.mbi_counterfactual_values[:-1],
                self.mbi_counterfactual_advantages[:-1],
                self.mbi_healths)

    def group_policy_update(self, group_policy):
        '''update policy and value function based on group policy and clear experience
        Notes:
         - since each agent's model points to the group model object
         no explicit assignment of the model to an agent should be
         necessary after a training step. TODO: verify this via testing
        '''
        pass

    def clear_individual_agent_episode_data(self):
        self.mbi_observations = []
        self.mbi_actions = []
        self.mbi_rewards = []
        self.mbi_obs_values = []
        self.mbi_factual_values = []
        self.mbi_counterfactual_values = []
        self.mbi_neglogp_actions = []
        self.mbi_dones = []
        self.mbi_returns = []
        self.mbi_credits = []
        self.mbi_factual_advantages = []
        self.mbi_counterfactual_advantages = []
        self.mbi_healths = []

        
    def preupdate(self):
        ''' Unused function for compatibility with train.py
        '''
        raise NotImplementedError

    def update(self, agents, t):
        ''' Unused function for compatibility with train.py
        '''
        raise NotImplementedError


class MAPPOAgentTrainer(BaselinesAgentTrainer):
    '''
    Learn policy using multi-agent variant of PPO algorithm

    Notes:
     - based on OpenAI's baseline of PPO: https://github.com/openai/baselines/tree/master/baselines/ppo2
     - there is some "name mashing" between policies and models due to combining multiagent-particle-envs
        with baselines libraries
    '''

    # def __init__(self, *, name, agent_index, vec_env, args):
    def __init__(self, *, name, agent_index, obs_space, act_space, args, 
            nsteps=2048, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5):

        raise NotImplementedError   # this code is not complete, but instead of commenting or removing,
                                    # just throwing error if initiated
        
        self.name = name
        self.agent_index = agent_index
        self.args = args

        # if not isinstance(vec_env, VecEnv):
        #     raise Exception('vec_env must be vectorized gym environment')

        # create dummy environment object to pass obs and act info to build_policy
        # i.e. I don't want to modify build_policy
        dummy_env = type('EmptyObj', (), {})()
        dummy_env.action_space = act_space
        dummy_env.observation_space = obs_space

        # create dummy tensor flow variables to avoid Saver error
        # TODO: remove this or turn into act function
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            # create policy_fn that returns a PolicyWithValue object
            policy_fn = build_policy(env=dummy_env, policy_network='mlp')
            dummy_var = tf.Variable(0.0)
            # create policy model from baselines/ppo2/model.py
            self.model = Model( policy=policy_fn, 
                            ob_space=obs_space, 
                            ac_space=act_space, 
                            nbatch_act=1, 
                            nbatch_train=nsteps,
                            nsteps=nsteps, 
                            ent_coef=ent_coef, 
                            vf_coef=vf_coef,
                            max_grad_norm=max_grad_norm)
            self.act = act_space.sample


    def action(self, obs):
        ''' return agent's action based on observation
        '''

        # actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
        # return actions

        return np.array([0] + self.act() + self.act())

    def experience(self, obs, act, rew, new_obs, done, terminal):
        pass

    def preupdate(self):
        pass

    def update(self, agents, t):
        return None

def redistributed_softmax(p_arr, scale):
    ''' find softmax of array of values and then re-distribute based on scale param 
    Notes:
     - at scale=0, no redistribution
     - at scale=inf, completely redistributed to lowest softmax value
    '''

    # map probabilities to weights that sum to 1
    p_soft = softmax(p_arr)
    n = len(p_arr)
    assert scale >= 0.0
    assert scale <= 1.0
    scrape_size = scale*(max(p_soft) - min(p_soft))
    scrape_height = max(p_soft) - scrape_size
    scrape_mass = 0.0
    n_fills = 0
    for p in p_soft:
        if p > scrape_height:
            scrape_mass += p - scrape_height
        else:
            n_fills += 1

    p_scaled = np.zeros(n)
    for i,p in enumerate(p_soft):
        if p > scrape_height:
            p_scaled[i] = scrape_height
        else:
            p_scaled[i] = p + scrape_mass/float(n_fills)

    assert abs(sum(p_scaled) - 1.0) < 1e-6
    return p_scaled


