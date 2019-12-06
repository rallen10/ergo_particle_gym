"""Module for the Risk-Exploiting Circuit Scenario (i.e. Simplified Comm Relay)
 
- Network of agents are to act as a connecition between origin and destination terminals of 
known (observed).
- This is a simplified, surrogate scenario for the more complex communication relay scenario 
since would be expected to produce similar physical behavior of the network without adding
the complexity of message passing to the action space 
- The aggregate reward is a function of the connection quality of the network between the terminals.
- The connection quality is modeled like an electrical circuit where resistance is a function
of distance between agents in the network. Connections can be serial and parallel.
Beyond a certain distance threshold, connections are not made between agents
- Landmarks can act to boost or degrade the connection quality of agents within a certain
proximity.
- Landmarks can also have a risk associated with them, i.e. probability of causing a nearby
agent to fail and eliminating if from the network.
- Agents actions are their movements
- Most general case: landmarks are at unknown locations and unknown nature (i.e. risk,
signal degredation) and part of the problem is to explore for landmarks and learn their nature
- Simplified case: to accelerate testing and learning, a simplified case has the landmarks
at known locations with known nature. 
- Interesting behavior to be investigate: Landmarks of large boosting quality but high risk
of causing agent failures. How does the network leverage such double edged swords?

SIMPLIFIED VERSION:

- Only 3 agents as opposed to 25
- No intermediate landmarks that can boost signal or cause failurs
- Terminals are at the same position in every episode
"""

import numpy as np
from multiagent.scenario import BaseScenario
from particle_environments.mager.world import MortalAgent, HazardousWorld, RiskRewardLandmark
from particle_environments.mager.observation import format_observation
from particle_environments.common import is_collision, distance, delta_pos
from particle_environments.common import RadialPolynomialRewardFunction2D as RadialReward
from particle_environments.common import RadialBernoulliRiskFunction2D as RadialRisk
from particle_environments.common import DefaultParameters as DP
from particle_environments.common import linear_index_to_lower_triangular, ResistanceNetwork


# Scenario Parameters
_DISTANCE_RESISTANCE_GAIN = 1.0
_MAX_COMMUNICATION_DISTANCE = 0.7
_AGENT_SIZE = 0.025
_LANDMARK_SIZE = 0.025

_NON_TERMINAL_LANDMARKS = []

class Scenario(BaseScenario):
    # static class
    num_agents = 3

    def make_world(self):
        world = HazardousWorld()

        # set scenario-specific world parameters
        world.collaborative = True
        world.systemic_rewards = True
        world.identical_rewards = False
        world.dim_c = 0 # observation-based communication
        world.max_communication_distance = _MAX_COMMUNICATION_DISTANCE
        world.distance_resistance_gain = _DISTANCE_RESISTANCE_GAIN

        # create and add terminal landmarks 
        # no intermediate landmarks in this scenario
        world.origin_terminal_landmark = RiskRewardLandmark( risk_fn=None, reward_fn=RadialReward(1.0, 10.0))
        world.origin_terminal_landmark.name = 'origin'
        world.origin_terminal_landmark.state.p_pos = np.array([-0.75, -0.75])
        world.destination_terminal_landmark = RiskRewardLandmark( risk_fn=None, reward_fn=RadialReward(1.0, 10.0))
        world.destination_terminal_landmark.name = 'destination'
        world.destination_terminal_landmark.state.p_pos = np.array([0.75, 0.75])

        # create landmark list and set properties
        world.landmarks = [world.origin_terminal_landmark, world.destination_terminal_landmark]
        for i, landmark in enumerate(world.landmarks):
            landmark.p_vel = np.zeros(world.dim_p)
            landmark.collide = False
            landmark.movable = False
            landmark.size = _LANDMARK_SIZE
            landmark.color = np.array([landmark.risk_fn.get_failure_probability(0,0) + .1, 0, 0])

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

        # terminal landmarks at static locations

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

    def reward(self, agent, world, systemic_call=False):
        ''' individual or system-wide rewards per agent 
        Args
         - systemic_call (bool): determine if this is a system-wide calc, or individual agent
        Notes:
         - this function returns zero for all individual agents, but can return nonzero
         for the entire system
        '''

        if systemic_call:
            return self.systemic_reward(world)
        else:
            if agent.terminated:
                # TODO: perhaps add a negative reware
                return 0.0
            else:
                return 0.0

    def systemic_reward(self, world):
        ''' Singular reward for entire system based on communication "conductance" through network
        Notes:
         - for each agent connections are established to all other agents within a proximity
         threshold
         - for each connection, the "resistance" is calculated based on length of distance, longer
         distances have greater resistance.
         - if an agent is within proximity of a Landmarks then the agent's connection resistances are
         amplified or attenuated equally based on the reward function of the landmark. Positive
         reward functions attenuate resistances equally, negative reward functions amplify
         resistances equally
         - Once all connections have been determined and resistances calculated, the "current"
         is calculated between the origin and destination terminals using a normalized, fixed
         "voltage difference" across the terminals.
         - The higher the "current" the greater the systemic rewards
        '''

        # define nodes in resistance network
        # by construction, node 0 is origin landmark, node 1 is destination landmark
        # terminated agents are not part of network
        nodes = [world.origin_terminal_landmark, world.destination_terminal_landmark]
        nodes.extend([a for a in world.agents if not a.terminated])
        n_nodes = len(nodes)

        # init list to hold direct communication resistance values between agents
        # there is no direct communication between origin and destination
        n_pairs = int(n_nodes*(n_nodes+1)/2)
        resistance_array = [None]*n_pairs
        resistance_array[0] = 0.0
        resistance_array[1] = np.inf
        resistance_array[2] = 0.0
        
        # calculate direct communication resistance between agents
        for k in range(3,n_pairs):
            i,j = linear_index_to_lower_triangular(k)
            if i == j:
                resistance_array[k] = 0.0
            else:
                resistance_array[k] = self._calculate_resistance(nodes[i], nodes[j], world)

        # create resistance network
        resnet = ResistanceNetwork(n_nodes, resistance_array)

        # calculate resistance between origin and destination 
        comm_resistance = resnet.get_two_point_resistance(0,1)
        assert not isinstance(comm_resistance, complex)

        # systemic reward is inverse of resistance (conductance)
        return [1.0/comm_resistance]*self.num_agents


    def _calculate_resistance(self, agent1, agent2, world):
        ''' calculate communication resistance as a function of distance between agents
        TODO:
         - perhaps normalize gain based on max communication distance
        '''
        d = distance(agent1, agent2)
        if d > world.max_communication_distance:
            res = np.inf
        else:
            landmark_gain = self._calculate_landmark_resistance_gain(agent1, agent2, world)
            res =  world.distance_resistance_gain*landmark_gain*d

        return res

    def _calculate_landmark_resistance_gain(self, agent1, agent2, world):
        ''' calculate amplification/attenuation of comm resistance based on proximity to landmarks
        '''
        # TODO: complete this based on landmark reward functions
        return 1.0

    def done_callback(self, agent, world):
        ''' indicate a terminated agent as done '''
        if agent.terminated: 
            return True
        else:
            return False

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame

        def communications_observed(other_agent):
            ''' fill in information communicated between agents
            '''
            comms = delta_pos(other_agent, agent).tolist()
            comms += [self._calculate_landmark_resistance_gain(agent, other_agent, world)]
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

        obs = np.asarray(agent.state.p_pos.tolist() + landmark_positions + communications)

        if agent.terminated:
            # if agent is terminated, return all zeros for observation
            # TODO: make this more efficient. Right now it does a lot of unnecessary calcs which are all
            #   then set to zero. Done this way to ensure consistant array size
            obs = 0.0*obs

        return obs

