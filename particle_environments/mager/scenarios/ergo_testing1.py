"""Module for the integration and system testing

- Single agent and single landmark
- Agent has complete knowledge of the world, must navigate to max reward or min cost
- Purpose is to test that learning functions are behaving as expected
"""

import numpy as np
from multiagent.scenario import BaseScenario
from particle_environments.mager.world import MortalAgent, HazardousWorld, RiskRewardLandmark
from particle_environments.mager.observation import format_observation
from particle_environments.common import is_collision, distance, delta_pos, PolynomialRewardFunction2D
from particle_environments.common import DefaultParameters as DP


class Scenario(BaseScenario):
    # static class
    num_agents = 1

    def make_world(self):
        world = HazardousWorld()
        world.dim_c = 2

        # add landmarks
        # TODO: decide on desired landmark properties instead of using a 'default' collection
        world.landmarks = []
        world.landmarks.append(RiskRewardLandmark(risk_fn=None, 
            reward_fn=PolynomialRewardFunction2D(coefs=[10.0, 0.0, 0.0, -1.0, 0.0, -1.0], 
                                                 bounds={'xmin':-2.0, 'xmax':2.0, 'ymin':-2.0, 'ymax':2.0})))

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = DP.landmark_size
            # properties for landmarks
            if isinstance(landmark, RiskRewardLandmark) and landmark.is_hazard:
                landmark.color = np.array([landmark.risk(landmark.size) + .1, 0, 0])
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
            agent.size = DP.agent_size
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.color = np.array([0.35, 0.35, 0.85])

        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

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

    def _individual_reward(self, agent, world):
        # TODO cache calculated rewards to avoid re-calculation (this should be done in a generalized scenario class)
        rew = 0
        for l in world.landmarks:
            if isinstance(l, RiskRewardLandmark):
                rel_pos = delta_pos(agent, l)
                rew += l.reward_fn.get_value(rel_pos[0], rel_pos[1])
        return rew

    def reward(self, agent, world):
        # TODO cache calculated rewards to avoid re-calculation (this should be done in a generalized scenario class)
        rew = sum([self._individual_reward(a, world) for a in world.agents])

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        landmark_positions =  format_observation(observe = lambda landmark: delta_pos(agent, landmark).tolist(),
                                                 objects = world.landmarks, 
                                                 num_observations = len(world.landmarks), 
                                                 observation_size = world.dim_p)
        obs = np.asarray(agent.state.p_pos.tolist() + landmark_positions)
        return obs


