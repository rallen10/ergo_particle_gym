# module for handling environments and scenarios from multiple packages (e.g. mager, multiagent-particle-envs, etc)

import importlib
import sys
import os

# Due to how multiagent-particle-envs package is configured with references to 'multiagent', we need to hack the sys.path for imports
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/multiagent_particle_envs')
from multiagent.environment import MultiAgentEnv
from particle_environments.mager.environment import MultiAgentRiskEnv

_ENVIRONMENTS = {
    "MultiAgentEnv": {"class": MultiAgentEnv, "scenarios": "multiagent.scenarios"},
    "MultiAgentRiskEnv": {"class": MultiAgentRiskEnv, "scenarios": "particle_environments.mager.scenarios"}
}

def load_environment_class(env_name):
    '''
    returns a class definition for an environment defined in the subpackages
    '''
    if env_name in _ENVIRONMENTS:
        return _ENVIRONMENTS[env_name]["class"]
    else:
        raise Exception('Invalid environment name: {}'.format(env_name))

def load_scenario_module(env_name, scenario_name):
    '''
    load a module that defines the specific scenario of a given environment
    '''
    if env_name in _ENVIRONMENTS:
        scenarios = _ENVIRONMENTS[env_name]["scenarios"]
        scenario_import_str = scenarios+"."+scenario_name
        if importlib.util.find_spec(scenario_import_str) is not None:
            scenario = importlib.import_module(scenario_import_str)
        else:
            raise Exception('Invalid scenario. Environment {} has no scenario named {}'.format(env_name, scenario_name))

        return scenario
    else:
        raise Exception('Invalid environment. No environment found named {}'.format(env_name))


    