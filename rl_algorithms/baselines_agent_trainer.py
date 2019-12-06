import sys
import os

# Due to how maddpg package is configured with references to 'maddpg', we need to hack the sys.path for imports
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'maddpg'))
from maddpg import AgentTrainer

class BaselinesAgentTrainer(AgentTrainer):
    ''' 
    class with same (similar) functions as AgentTrainer class but with 
    different init arguments in order to handle the format from the baselines library
    '''
    def __init__(self, *, name, model, obs_shape, act_space, args):
        raise NotImplementedError

    def action(self, obs):
        raise NotImplementedError

    def experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplementedError

    def preupdate(self):
        raise NotImplementedError

    def update(self, agents):
        raise NotImplementedError
