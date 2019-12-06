import numpy as np
from multiagent.core import World, Landmark
from multiagent.scenario import BaseScenario
from particle_environments.mager.world import MortalAgent, HazardousWorld
from particle_environments.mager.observation import format_observation
from particle_environments.common import is_collision, distance, delta_pos
from particle_environments.common import DefaultParameters as DP

class Obstacle(Landmark):
    def __init__(self):
        super().__init__()
        self.known = False

class ObstacleWorld(World):
    def __init__(self):
        super().__init__()
        self.obstacles = []

    @property
    def entities(self):
        return self.agents + self.landmarks + self.obstacles

class Scenario(BaseScenario):
    num_agents = 10
    num_landmarks = 3
    num_obstacles = 1

    def make_world(self):
        world = HazardousWorld()
        
        # observation-based communication
        world.dim_c = 0
        world.max_communication_distance = DP.max_communication_distance

        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = DP.landmark_size
        # add obstacles
        world.obstacles = [Obstacle() for i in range(self.num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = True
            obstacle.size = 0.05
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # add agents with random properties
        world.agents = [MortalAgent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.terminated = False
            agent.collide = True
            agent.silent = True
            agent.size = DP.agent_size
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        goal = np.random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        for i, obstacle in enumerate(world.obstacles):
            obstacle.color = np.array([0.90, 0.40, 0.40])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for obstacle in world.obstacles:
            obstacle.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            obstacle.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.linalg.norm(a.state.p_pos - l.state.p_pos) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.linalg.norm(a.state.p_pos - l.state.p_pos) for a in world.agents]
            rew -= min(dists)

        for o in world.obstacles:
            if o.known:
                dists = [np.linalg.norm(a.state.p_pos - o.state.p_pos) for a in world.agents]
                rew += min(dists)
        if agent.collide:
            for a in world.agents:
                if is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):

        def communications_observed(other_agent):
            ''' fill in information communicated between agents
            '''
            comms = delta_pos(other_agent, agent).tolist()
            comms += [other_agent.state.c]
            # will only work with zero-padding formatting
            # TODO: I think non-communication should send None instead of zero, because zero has real meaning
            #   however this causes a problem with action function
            if distance(agent, other_agent) > world.max_communication_distance:
                comms = [0] * len(comms)
            return comms

        landmark_positions =  format_observation(observe = lambda landmark: delta_pos(landmark, agent).tolist(),
                                                 objects = world.landmarks, 
                                                 num_observations = len(world.landmarks), 
                                                 observation_size = world.dim_p)

        communications = format_observation(observe = communications_observed, 
                                            objects = [a for a in world.agents if (a is not agent and not a.terminated)],
                                            num_observations = self.num_agents, 
                                            observation_size = world.dim_p + 1,
                                            sort_key = lambda o: distance(agent, o))

        return np.asarray(agent.state.p_pos.tolist() + landmark_positions + communications)
