""" Module for risk-exploiting landmark identification scenario
Environment will contain:
- Landmarks which can give reward and/or can cause an agent to fail with a probability based on distance (each has own function)
- A variable number of agents that can drop as agents are destroyed (at least 25 at start)
Agents will:
- know location but not effects of landmarks
- be able to communicate with others within a certain distance (can communicate state, action, reward, and policy)
- have a high chance of failure on collision
"""

import numpy as np
from multiagent.scenario import BaseScenario
from particle_environments.mager.world import MortalAgent, HazardousWorld, RiskRewardLandmark
from particle_environments.mager.world import TemporarilyObservableRiskRewardLandmark as TORRLandmark
from particle_environments.mager.observation import format_observation
from particle_environments.common import is_collision, distance, delta_pos
from particle_environments.common import RadialPolynomialRewardFunction2D as RadialReward
from particle_environments.common import RadialBernoulliRiskFunction2D as RadialRisk
from particle_environments.common import DefaultParameters as DP


# Scenario Parameters
_MAX_COMMUNICATION_DISTANCE = 0.5
_AGENT_SIZE = 0.01
_LANDMARK_SIZE = 0.025


_LANDMARKS = []
_LANDMARKS.append(
    TORRLandmark( risk_fn=RadialRisk(0.1), reward_fn=RadialReward(0.15, 10.0), observe_duration=1.0))
_LANDMARKS.append(
    TORRLandmark( risk_fn=RadialRisk(0.1), reward_fn=RadialReward(0.15, 10.0), observe_duration=1.0))
_LANDMARKS.append(
    TORRLandmark( risk_fn=RadialRisk(0.1), reward_fn=RadialReward(0.15, 10.0), observe_duration=1.0))

class Scenario(BaseScenario):
    # static class
    num_agents = 10

    def make_world(self):
        world = HazardousWorld()
        
        # observation-based communication
        world.dim_c = 0
        world.identical_rewards = False
        world.max_communication_distance = _MAX_COMMUNICATION_DISTANCE

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
            agent.collide = True
            agent.silent = True
            agent.terminated = False
            agent.size = _AGENT_SIZE
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.color = np.array([0.35, 0.35, 0.85])

        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.reset_landmark()

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

    # def _individual_reward(self, agent, world):
    #     # TODO cache calculated rewards to avoid re-calculation (this should be done in a generalized scenario class)

    #     if agent.terminated:
    #         # terminated agents produce no reward
    #         return 0.0

    #     rew = 0
    #     for l in world.landmarks:
    #         if isinstance(l, RiskRewardLandmark):
    #             xrel, yrel = delta_pos(agent, l)
    #             rew += l.reward_fn.get_value(xrel, yrel)
    #     return rew

    def done_callback(self, agent, world):
        ''' indicate a terminated agent as done '''
        if agent.terminated: 
            return True
        else:
            return False

    def reward(self, agent, world):
        ''' reward value received by each agent
        '''
        # rew = sum([self._individual_reward(a, world) for a in world.agents if not a.terminated])

        if agent.terminated:
            # terminated agents produce no reward
            return 0.0

        total_reward = 0.0
        for ldmrk in world.landmarks:
            if isinstance(ldmrk, RiskRewardLandmark):

                # check if landmark is observable and skip if not
                if not ldmrk.is_observable():
                    continue

                # if observable, evaluate reward
                xrel, yrel = delta_pos(agent, ldmrk)
                base_reward = ldmrk.reward_fn.get_value(xrel, yrel)

                # adjust reward based on cumulative reward distributed (diminishing returns)
                adjusted_reward = base_reward * np.exp(-ldmrk.cumulative_distributed_reward)
                ldmrk.update_cumulative_distributed_reward(adjusted_reward)

                # sum up rewards from all landmarsk
                total_reward += adjusted_reward

        return total_reward

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame

        def observe_agents(other_agent):
            ''' fill in information communicated/observed between agents
            '''
            ag_obs = delta_pos(other_agent, agent).tolist()

            # check if within observation range
            if distance(agent, other_agent) > world.max_communication_distance:
                ag_obs = [0] * len(ag_obs)

            return ag_obs

        def observe_landmarks(landmark):
            ''' fill in information observed about landmarks
            '''
            ld_obs = delta_pos(landmark, agent).tolist()

            # check if within observation range and is observable
            if (distance(landmark, agent) > world.max_communication_distance or
                not landmark.is_observable()):
                ld_obs = [0]*len(ld_obs)

            return ld_obs

        landmark_positions = format_observation(observe = observe_landmarks,
                                                objects = world.landmarks, 
                                                num_observations = len(world.landmarks), 
                                                observation_size = world.dim_p)

        communications = format_observation(observe = observe_agents, 
                                            objects = [a for a in world.agents if (a is not agent and not a.terminated)],
                                            num_observations = self.num_agents, 
                                            observation_size = world.dim_p,
                                            sort_key = lambda o: distance(agent, o))

        obs = np.asarray(agent.state.p_pos.tolist() + landmark_positions + communications)

        if agent.terminated:
            # if agent is terminated, return all zeros for observation
            # TODO: make this more efficient. Right now it does a lot of unnecessary calcs which are all
            #   then set to zero. Done this way to ensure consistant array size
            obs = 0.0*obs

        return obs

