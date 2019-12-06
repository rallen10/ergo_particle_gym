"""
module for handling reinforcement learning algorithms or "trainers" (e.g. maddpg, nonlearning, trpo, etc)
"""
import sys
import os

# Due to how maddpg package is configured with references to 'maddpg', we need to hack the sys.path for imports
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/maddpg')
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg import AgentTrainer
from rl_algorithms.nonlearning import NonLearningAgentTrainer
from rl_algorithms.scenariolearning import ScenarioHeuristicAgentTrainer
from rl_algorithms.mclearning import ScenarioHeuristicGroupTrainer
from rl_algorithms.mappo import MAPPOAgentTrainer, PPOGroupTrainer 

_TRAINERS = {
    "MADDPGAgentTrainer": { "class": MADDPGAgentTrainer, 
                            "legacy_multidiscrete": True, 
                            "discrete_action_space": True,
                            "combined_action_value": False},
    "MAPPOAgentTrainer": {  "class": MAPPOAgentTrainer, 
                            "legacy_multidiscrete": False, 
                            "discrete_action_space": False,
                            "combined_action_value": True},
    "PPOGroupTrainer": {    "class": PPOGroupTrainer, 
                            "legacy_multidiscrete": False, 
                            "discrete_action_space": False,
                            "combined_action_value": True},
    "NonLearningAgentTrainer": {        "class": NonLearningAgentTrainer,   
                                        "legacy_multidiscrete": True, 
                                        "discrete_action_space": True,
                                        "combined_action_value": False},
    "ScenarioHeuristicAgentTrainer": {  "class": ScenarioHeuristicAgentTrainer, 
                                        "legacy_multidiscrete": True, 
                                        "discrete_action_space": True, # confusingly, ScenarioHeuristicAgentTrainer is not necessarily a discrete action space but it is labeled this way to stay consistent with the MADDPG algorithm it was developed alongside
                                        "combined_action_value": False}, 
    "ScenarioHeuristicGroupTrainer": {  "class": ScenarioHeuristicGroupTrainer, 
                                        "legacy_multidiscrete": True, 
                                        "discrete_action_space": True, # confusingly, ScenarioHeuristicGroupTrainer is not necessarily a discrete action space but it is labeled this way to stay consistent with the MADDPG algorithm it was developed alongside
                                        "combined_action_value": False}  
}

def load_trainer_class(trainer_name):
    '''
    returns a class definition for a trainer defined in the subpackages
    '''
    if trainer_name in _TRAINERS:
        return _TRAINERS[trainer_name]["class"]
    else:
        raise Exception('Invalid trainer name: {}'.format(trainer_name))

def use_legacy_multidiscrete(trainer_name):
    '''
    Due to discrepancy between maddpg and baselines, different algorithms
    use different definitions of MultiDiscrete class
    See: https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/multi_discrete.py
    '''
    if trainer_name in _TRAINERS:
        return _TRAINERS[trainer_name]["legacy_multidiscrete"]
    else:
        raise Exception('Invalid trainer name: {}'.format(trainer_name))

def use_discrete_action_space(trainer_name):
    '''
    Due to discrepancy between maddpg and other algorithms,
    certain algs need discrete action spaces
    '''
    if trainer_name in _TRAINERS:
        return _TRAINERS[trainer_name]["discrete_action_space"]
    else:
        raise Exception('Invalid trainer name: {}'.format(trainer_name))

def use_combined_action_value(trainer_name):
    '''
    Boolean if trainer.action function returns action, value estimate, and 
    negative log probability of action
    '''
    if trainer_name in _TRAINERS:
        return _TRAINERS[trainer_name]["combined_action_value"]
    else:
        raise Exception('Invalid trainer name: {}'.format(trainer_name))

