import numpy as np
import tensorflow as tf
from copy import deepcopy
from bisect import bisect
from rl_algorithms.group_trainer import GroupTrainer
import rl_algorithms.maddpg.maddpg.common.tf_util as U


def softmax(x):
    '''softmax function for non tensorflow Tensors
    '''
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

class ScenarioHeuristicGroupTrainer(GroupTrainer):
    ''' Train group of agents in a monte carlo fashion using cross entropy method
    Notes:
     - Enforces that all agents have the same individual policy
     - clambda - policy behavioral parameters 
     - ctheta - policy distribution parameters (distribution of clambda)
    '''
    def __init__(self, agent_trainer_group, init_group_policy, n_episodes_per_batch, n_elite):

        self.agent_trainer_group = agent_trainer_group
        self.group_policy = init_group_policy
        self.n_elite = n_elite
        self.n_episodes_per_batch = n_episodes_per_batch
        self.policy_batch = []

        # if not init_group_policy is given, query from first agent
        if self.group_policy is None:
            self.group_policy = dict()
            pol = self.agent_trainer_group[0].get_initial_policy_distribution()
            for k in pol:
                self.group_policy[k] = {'clambda':pol[k][0], 'ctheta':pol[k]}

        # create tensorflow variables for compatibility with policy saving
        self.tensorize_group_policy()

        # check number of trials strictly greater than n_elite
        assert(self.n_episodes_per_batch > n_elite)

        # check that group policy format matches policy of each agent
        for agent in self.agent_trainer_group:
            assert(len(agent.behavior_params)==len(self.group_policy))
            for k in agent.behavior_params:
                assert(k in self.group_policy)

    def update_group_policy(self, terminal):
        ''' decide type of group policy update and execute it
        Notes:
         - Decides between sampling a new policy, updating policy 
         distribution and sample a new policy, or neither
        '''

        # no policy updates occur on non-terminal time steps
        if not terminal:
            return None

        # record value of current policy
        group_policy_value = self.evaluate_group_policy()

        # store policy in current policy batch
        self.policy_batch.append((group_policy_value, deepcopy(self.group_policy)))

        # check if current batch of policies has filled up
        if len(self.policy_batch) >= self.n_episodes_per_batch:
            # update distribution and sample new plicy
            self.update_group_policy_distribution()
            # resent policy batch
            self.policy_batch = []

        # generate new policy
        self.sample_group_policy()

        # distribute new group policy to agents and reset experiences
        for agent in self.agent_trainer_group:
            agent.group_policy_update(self.group_policy)
            assert(agent.cumulative_reward == 0.0)

        # No loss values to return
        return None

    def tensorize_group_policy(self):
        ''' store group policy as tensorflow variables for Saver compatability
        '''
        with tf.variable_scope("group_policy", reuse=tf.AUTO_REUSE):
            for k in self.group_policy:
                clambda_val = np.float32(self.group_policy[k]['clambda'])
                ctheta_mean_val = np.float32(self.group_policy[k]['ctheta'][0])
                ctheta_std_val = np.float32(self.group_policy[k]['ctheta'][1])
                clambda = tf.get_variable("{}_clambda".format(k), dtype=tf.float32,
                                initializer=tf.constant(clambda_val))
                ctheta_mean = tf.get_variable("{}_ctheta_mean".format(k), dtype=tf.float32,
                                initializer=tf.constant(ctheta_mean_val))
                ctheta_std = tf.get_variable("{}_ctheta_std".format(k), dtype=tf.float32,
                                initializer=tf.constant(ctheta_std_val))
                assign_clamba = tf.assign(clambda, clambda_val)
                assign_ctheta_mean = tf.assign(ctheta_mean, ctheta_mean_val)
                assign_ctheta_std = tf.assign(ctheta_std, ctheta_std_val)
                U.get_session().run(assign_clamba)
                U.get_session().run(assign_ctheta_mean)
                U.get_session().run(assign_ctheta_std)

        # with U.get_session() as sess:
        # for k in self.group_policy:
        #     assign_clamba = tf.assign("{}_clambda".format(k), self.group_policy[k]['clambda'])
        #     assign_ctheta_mean = tf.assign("{}_ctheta_mean".format(k), self.group_policy[k]['ctheta'][0])
        #     assign_ctheta_std = tf.assign("{}_ctheta_std".format(k), self.group_policy[k]['ctheta'][1])
        #     U.get_session().run(assign_clamba)
        #     U.get_session().run(assign_ctheta_mean)
        #     U.get_session().run(assign_ctheta_std)

    def evaluate_group_policy(self):
        ''' estimate the value of a group policy: v <- clambda
        '''
        group_policy_value = 0.0
        for agent in self.agent_trainer_group:
            group_policy_value += agent.cumulative_reward

        return group_policy_value

    def sample_group_policy(self):
        ''' generate a new policy from existing policy distribution
        Notes:
         - Updates clambda values based on fixed ctheta values
        '''

        # sample new clambda values based on gaussian ctheta values
        for k in self.group_policy:
            self.group_policy[k]['clambda'] = np.random.normal(
                                        self.group_policy[k]['ctheta'][0],
                                        self.group_policy[k]['ctheta'][1])


    def update_group_policy_distribution(self):
        ''' generate new policy distribution; i.e. 'learns' clabmda values
        Notes:
         - does not update lambda values
        '''

        # generate new distribution
        self.group_policy = self.minimum_cross_entropy_gaussian_policy_distribution()

    def minimum_cross_entropy_gaussian_policy_distribution(self):
        ''' argument that minimizes cross entropy of gaussian distributions
        Notes:
         - See Kochenderfer 4.7.3, but variation that probabilistically
            draws elite samples based on value instead of deterministically
         - Updates ctheta values (distribution parameters) and subsequently 
            clambda values
        '''

        zero_threshold = 1e-10

        # select elite policies from observed policies
        elite_clambda = self.select_elite_policies()

        # calculate argmax ctheta of cross entropy and set it as policy
        # sample new policy clambda values
        new_group_policy_dist = dict()
        for k in self.group_policy:
            new_group_policy_dist[k] = {
                'ctheta':(np.mean(elite_clambda[k]), np.std(elite_clambda[k])),
                'clambda':None}

        return new_group_policy_dist
        

    def select_elite_policies(self):
        ''' sample observed policies in order to select elite policies
        Notes:
            - perhaps should re-weight elite after being selected to discourage
            but not preclude re-selection
        '''

        # get candidate policies
        candidate_policies = deepcopy(self.policy_batch)

        # create dictionary to hold elite clambda values
        elite_clambda = dict()
        for k in self.group_policy:
            elite_clambda[k] = []

        for j in range(self.n_elite):
            # create sampling probabilities of policies based on values
            # v = []
            # for p in candidate_policies:
            #   v.append(p[0])
            # vcs = np.cumsum(softmax(v))
            vcs = np.cumsum(softmax([cpol[0] for cpol in candidate_policies]))

            # randomly sample elites based on value-based probabilities
            ep_index = bisect(vcs, np.random.rand())
            elite = candidate_policies[ep_index]
            for k in elite_clambda:
                elite_clambda[k] += [elite[1][k]['clambda']]

            # remove elite from candidate (sample without replacement)
            # break if candidate policies becomes empty
            del candidate_policies[ep_index]
            if len(candidate_policies) == 0:
                break

        return elite_clambda
        
        
