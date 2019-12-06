'''
    General parent class for defining trainers specific to their scenarios, i.e. heuristic trainters
'''

import numpy as np
from maddpg import AgentTrainer
import tensorflow as tf
import maddpg.common.tf_util as U


class ScenarioHeuristicAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args

        # create dummy tensor flow variables to avoid Saver error
        # TODO: remove this or turn into act function
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
        with tf.variable_scope(self.name, reuse=None):
            self.dummy_var = U.function(obs_ph_n, outputs=tf.Variable(0))

    def action(self, obs):
        raise NotImplementedError()

    def experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplementedError()

    def preupdate(self):
        raise NotImplementedError()

    def update(self, agents, t):
        raise NotImplementedError()