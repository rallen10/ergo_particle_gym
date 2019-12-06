import sys
import os


class GroupTrainer(object):
    ''' 
    in contrast to AgentTrainer, GroupTrainer assume all agents are getting
    identicaly copies of group-trained policy
    '''
    def __init__(self, *, agent_trainer_group):
        raise NotImplementedError

    def update_group_policy(self, train_step, terminal):
        ''' decide type of group policy update and execute it
        Notes:
         - Decides between sampling a new policy, updating policy 
         distribution and sample a new policy, or neither
        '''
        raise NotImplementedError

