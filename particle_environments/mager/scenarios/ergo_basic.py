""" Module for risk-exploiting landmark identification scenario
- This is meant to be the most basic extension the simple environment to multi-agents
- rewards are completely independent between agents
- agents don't even observe each other since their rewards don't depend on each other
- no collisions
- Landmarks which can give reward and/or can cause an agent to fail with a probability based on distance (each has own function)
- know location location of landmarks but cannont directly observe the rewards/risk of landmarks except through indirect interaction
"""

import numpy as np
from random import shuffle
from multiagent.scenario import BaseScenario
from particle_environments.mager.world import MortalAgent, HazardousWorld, RiskRewardLandmark
from particle_environments.mager.observation import format_observation
from particle_environments.common import is_collision, distance, delta_pos, delta_vel
from particle_environments.common import RadialPolynomialRewardFunction2D as RadialReward
from particle_environments.common import RadialBernoulliRiskFunction2D as RadialRisk
from rl_algorithms.scenariolearning import ScenarioHeuristicAgentTrainer


# Scenario Parameters
_MAX_COMMUNICATION_DISTANCE = np.inf
_AGENT_SIZE = 0.15
_LANDMARK_SIZE = 0.05
_AGENT_OBSERVATION_LEN = 5
_LANDMARK_OBSERVATION_LEN = 3
_NUM_AGENTS = 3


_LANDMARKS = []
_LANDMARKS.append(
    RiskRewardLandmark( risk_fn=RadialRisk(_LANDMARK_SIZE, 0.5), reward_fn=RadialReward(1.0, 0.0)))
_LANDMARKS.append(
    RiskRewardLandmark( risk_fn=None, reward_fn=RadialReward(1.0, 0.0)))
_LANDMARKS.append(
    RiskRewardLandmark( risk_fn=None, reward_fn=RadialReward(1.0, 0.0)))
_LANDMARKS.append(
    RiskRewardLandmark( risk_fn=None, reward_fn=RadialReward(1.0, 0.0)))
_N_LANDMARKS = len(_LANDMARKS)

class Scenario(BaseScenario):
    # static class
    num_agents = _NUM_AGENTS

    def make_world(self):
        world = HazardousWorld(collision_termination_probability=0.0)
        
        # observation-based communication
        world.dim_c = 0
        world.max_communication_distance = _MAX_COMMUNICATION_DISTANCE

        # collaborative rewards
        world.collaborative = True
        world.systemic_rewards = False
        world.identical_rewards = False

        # add landmarks
        world.landmarks = []
        for lm in _LANDMARKS:
            world.landmarks.append(lm)

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = _LANDMARK_SIZE
            # properties for landmarks
            if isinstance(landmark, RiskRewardLandmark) and landmark.is_hazard:
                #TODO: make colors heatmap of risk probability over all bounds
                landmark.color = np.array([landmark.risk_fn.get_failure_probability(0,0) + .1, 0, 0])
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        # add agents
        world.agents = [MortalAgent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.terminated = False
            agent.size = _AGENT_SIZE
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.previous_observation = None

        # shuffle landmarks to make sure hazard is not in same index
        shuffle(world.landmarks)
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            # properties for landmarks
            if isinstance(landmark, RiskRewardLandmark) and landmark.is_hazard:
                #TODO: make colors heatmap of risk probability over all bounds
                landmark.color = np.array([landmark.risk_fn.get_failure_probability(0,0) + .1, 0, 0])
                landmark.hazard_tag = 1.0
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.hazard_tag = 0.0

    def benchmark_data(self, agent, world):
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.linalg.norm(a.state.p_pos - l.state.p_pos) for a in world.agents]
            min_dists += min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if is_collision(a, agent):
                    collisions += 1
        return (self.reward(agent, world), collisions, min_dists, occupied_landmarks)

    def done_callback(self, agent, world):
        return False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        assert world.identical_rewards == False
        rew = 0
        rew -= min([distance(agent,lm) for lm in world.landmarks if not lm.is_hazard])
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame


        def observe_landmarks(landmark):
            ''' fill in information observed about landmarks
            '''
            # ld_obs = delta_pos(landmark, agent).tolist()
            ld_obs = landmark.state.p_pos.tolist()

            # check if landmark is giving reward or hazard warning
            d = distance(landmark, agent)
            if d < landmark.size:
                if landmark.is_hazard:
                    landmark.hazard_tag = 1.0
                    landmark.color = np.array([1.1, 0, 0])
            ld_obs += [landmark.hazard_tag]

            assert(len(ld_obs) == _LANDMARK_OBSERVATION_LEN)
            return ld_obs

        landmark_positions = format_observation(observe = observe_landmarks,
                                                objects = world.landmarks, 
                                                num_observations = len(world.landmarks), 
                                                observation_size = _LANDMARK_OBSERVATION_LEN)


        # new_obs = np.asarray(agent.state.p_vel.tolist() + agent.state.p_pos.tolist() + landmark_positions + agent_observations)
        new_obs = np.asarray(agent.state.p_vel.tolist() + agent.state.p_pos.tolist() + landmark_positions)

        return new_obs


class ScenarioHeuristicComputer(ScenarioHeuristicAgentTrainer):
    ''' representation of an individual agent's embedded processor and memory tailor
    Notes:
     - This is meant to be used as a scenario-specific alternative to
     the more general purpose, scenario-agnostic "trainers". It can hold an agents model
     of the world (transition and reward functions), policy, and learning process, if any.
    '''
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, **kwargs):
        raise NotImplementedError()
