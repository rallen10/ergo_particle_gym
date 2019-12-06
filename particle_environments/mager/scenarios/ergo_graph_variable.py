"""Module for the Risk-Exploiting Graph Scenario with variable input parameters
 
- Network of agents are to act as a connecition between origin and destination terminals of 
known (observed).
- This is a simplified, surrogate scenario for the more complex communication relay scenario 
since would be expected to produce similar physical behavior of the network without adding
the complexity of message passing to the action space 
- Agents can form links to other agents over small distances
- Reward is received by all agents if the two fixed terminals are connected (identical_systemic_reward case)
- An unobserved landmark in the environment is a hazard and can cause agents in its vicinity to be terminated
with some probability. Surviving agents may observed terminated agents in order to deduce location of the hazard
and avoid
- Agents actions are their movements


- Scenario is derived from simple_graph_small, simple_graph_large, and ergo_graph_large, making the following 
    settable parameters instead of hard coded
    - number of agents
    - number of hazards (0,1)
    - whether rewards are shared or "local"
    - whether observations are direct per entity or histogram based
    - Hazard failure risk
    - Collision failure risk
"""

import numpy as np
from bisect import bisect, insort
from shapely.geometry import LineString, Point
from multiagent.scenario import BaseScenario
from multiagent.core import Landmark
from particle_environments.mager.observation import format_observation, agent_histogram_observation
from particle_environments.mager.world import SensingLimitedMortalAgent, MortalAgent, HazardousWorld, RiskRewardLandmark
from particle_environments.common import is_collision, distance, delta_pos, delta_vel, nearest_point_on_line_segment_2d, check_2way_communicability
from particle_environments.common import RadialPolynomialRewardFunction2D as RadialReward
from particle_environments.common import RadialBernoulliRiskFunction2D as RadialRisk
from particle_environments.common import linear_index_to_lower_triangular, SimpleNetwork, truncate_or_pad
from rl_algorithms.scenariolearning import ScenarioHeuristicAgentTrainer


# Scenario Parameters
#_MAX_CONNECTION_DISTANCE = 0.60 # 4 agents
_TERMINAL_DISTANCE = 2.0
_CONNECTION_MARGIN = 1.2
_MAX_OBSERVATION_DISTANCE = 1.0
_CONNECTION_REWARD = 1.0
_TERMINATION_REWARD = -0.0
_AGENT_SIZE = 0.01
_LANDMARK_SIZE = 0.025
_HAZARD_SIZE = 0.1
_N_RADIAL_BINS = 4
_N_ANGULAR_BINS = 8
_N_OBSERVED_TERMINATIONS = 5
_N_TERMINALS = 2
_ZERO_THRESHOLD = 1e-6



class Scenario(BaseScenario):
    

    def __init__(self, *, num_agents, num_hazards, identical_rewards, observation_type, hazard_risk=0.5, collision_risk=0.0):
        '''
        Args:
         - num_agents [int] number of agents in scenario
         - num_hazards [int] number of hazards landmarks in the scenario
         - identical_rewards [bool] true if all agents receieve exact same reward, false if rewards are "local" to agents
         - observation_type [str] "direct" if observation directly of each entity, "histogram" if bin entities in spacial grid
         - hazard_risk [float] max probability of failure caused by hazard landmark
         - collision_risk [float] probability of failure caused by collision
        '''

        # check inputs
        assert isinstance(num_agents, int); assert num_agents >= 1
        assert isinstance(num_hazards, int); assert (num_hazards == 0 or num_hazards == 1)
        assert isinstance(identical_rewards, bool)
        assert (observation_type == "direct" or observation_type == "histogram")
        assert (hazard_risk >= 0.0 and hazard_risk <= 1.0)
        assert (collision_risk >= 0.0 and collision_risk <= 1.0)

        # set max connection distance such that there is a 5% probability of connection
        # between terminals given random placement of n agents
        # equation found emperically, see connection_probability_measurement.py
        if not np.isclose(_TERMINAL_DISTANCE, 2.0):
            raise Warning('Connection distance formula assumes distance between terminals of 2.0, received {}'.format(_TERMINAL_DISTANCE))
        # c_1 = 1.6838; c_2 = 0.18367; c_3 = -0.5316 # 5% probability of connection with random placement
        # c_1 = 1.29428202;  c_2 = 0.24156174; c_3 = -0.23681555 # 1% probability of connection with random placement
        c_1 = 0.9834973;  c_2 = 0.34086771; c_3 = -0.01181418 # 0.1% probability of connection with random placement
        self.max_connection_distance = c_1 * num_agents**(-c_2) + c_3
        assert self.max_connection_distance > 0.0

        # set member vars
        self.num_agents = num_agents
        self.num_hazards = num_hazards
        self.identical_rewards = identical_rewards
        self.observation_type = observation_type
        self.hazard_risk = hazard_risk
        self.collision_risk = collision_risk

    def make_world(self):
        world = HazardousWorld()

        # set scenario-specific world parameters
        world.collaborative = True
        world.systemic_rewards = True
        world.identical_rewards = self.identical_rewards
        world.dim_c = 0 # observation-based communication
        world.connection_reward = _CONNECTION_REWARD
        world.termination_reward = _TERMINATION_REWARD
        world.render_connections = True

        # add landmarks. terminals first then hazards (if any)
        # world.origin_terminal_landmark = RiskRewardLandmark( risk_fn=None, reward_fn=RadialReward(1.0, 0.0))
        # world.destination_terminal_landmark = RiskRewardLandmark( risk_fn=None, reward_fn=RadialReward(1.0, 0.0))
        world.origin_terminal_landmark = Landmark()
        world.destination_terminal_landmark = Landmark()
        world.landmarks = [world.origin_terminal_landmark, world.destination_terminal_landmark]

        world.hazard_landmarks = []
        for i in range(self.num_hazards):
            lm = RiskRewardLandmark( risk_fn=RadialRisk(_HAZARD_SIZE, self.hazard_risk), reward_fn=RadialReward(_HAZARD_SIZE, 0.0))
            lm.silent = True
            lm.deaf = True
            lm.ignore_connection_rendering = True
            world.hazard_landmarks.append(lm)
            world.landmarks.append(lm)

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark_%d' % i
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
        world.agents = [SensingLimitedMortalAgent(_MAX_OBSERVATION_DISTANCE, self.max_connection_distance) for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_%d' % i
            agent.collide = True
            agent.blind = False
            agent.silent = False
            agent.deaf = False
            agent.terminated = False
            agent.size = _AGENT_SIZE
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.previous_observation = None

        # place landmarks
        for landmark in world.landmarks:
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_pos = np.zeros(world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # randomize terminal locations, but regularize to ensure conistent distances
        origin_state, destination_state, hazard_states = self.spawn_landmarks(world)
        world.origin_terminal_landmark.state.p_pos = origin_state
        world.destination_terminal_landmark.state.p_pos = destination_state
        for i in range(self.num_hazards):
            world.hazard_landmarks[i].state.p_pos = hazard_states[i]


    def spawn_landmarks(self, world):
        ''' create communication terminals at random positions but regularized distance
        Notes:
         - regularizing the distance between terminals is important to ensure consistency in 
         max rewards possible between different episodes
        '''

        # angle of line connecting terminals
        th = np.random.uniform(0, 2.0*np.pi)

        # distance between terminals
        # d = np.random.normal(2.0, 0.1)
        d = _TERMINAL_DISTANCE
        dx = d/2.0*np.cos(th)
        dy = d/2.0*np.sin(th)

        # center of line connecting terminals
        xc = yc = 0.0

        # hazard state position along connecting line
        hazard_states = []
        for i in range(self.num_hazards):
            dh = np.random.uniform(-0.9*d/2.0, 0.9*d/2.0)
            dhx = dh*dx
            dhy = dh*dy
            hazard_states.append(np.array([xc+dhx, yc+dhy]))

        return  (np.array([xc-dx, yc-dy]), np.array([xc+dx, yc+dy]), hazard_states)

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
        ''' indicate a terminated agent as done (still being decided)
        Notes:
            - Even though terminated agent cannot take actions, it may be more appropriate
            to NOT set the agent is done in order to keep collecting data for training
            purposes
        '''
        # if agent.terminated: 
        #     return True
        # else:
        #     return False
        return False

    def reward(self, agent, world, systemic_call=False):
        if systemic_call:
            if self.identical_rewards and world.identical_rewards:
                return self._identical_systemic_reward(world)
            elif not self.identical_rewards and not world.identical_rewards:
                return self._local_systemic_reward(world)
            else:
                raise Exception(
                    "Inconsistent reward options: self.identical_rewards={} world.identical_rewards={}".format(
                    self.identical_rewards, world.identical_rewards))

        else:
            return 0.0


    def _local_systemic_reward(self, world):
        ''' reward agent if they are part of complete connection between terminals
        Notes:
        '''
        assert self.identical_rewards == False
        assert world.identical_rewards == False
        comm_net = self._create_network(world)

        reward_n = [0.0]*self.num_agents
        node_count = 2
        for i, a in enumerate(world.agents):
            if a.terminated:
                reward_n[i] = world.termination_reward
            else:
                # check ordering has stayed consistent in node list
                assert(a==comm_net.nodes[node_count])
                reward_n[i] = world.connection_reward*(
                    comm_net.breadth_first_connectivity_search(node_count, 0) and
                    comm_net.breadth_first_connectivity_search(node_count, 1)
                    )
                node_count += 1

        return reward_n

    def _identical_systemic_reward(self, world):
        ''' reward all agents the same if there is a complete connection between terminals
        Notes:
        '''
        assert self.identical_rewards == True
        assert world.identical_rewards == True
        comm_net = self._create_network(world)
        reward_n = [comm_net.breadth_first_connectivity_search(0,1)]*self.num_agents
        return reward_n


    def _create_network(self, world):
        ''' Establish connectivity network at every time step
        '''

        # define nodes in simple connectivity network
        # by construction, node 0 is origin landmark, node 1 is destination landmark
        # terminated agents are not part of network
        nodes = [world.origin_terminal_landmark, world.destination_terminal_landmark]
        nodes.extend([a for a in world.agents if not a.terminated])
        n_nodes = len(nodes)
        comm_net = SimpleNetwork(nodes)

        # init list to hold direct communication distance values between agents
        # there is no direct communication between origin and destination
        n_pairs = int(n_nodes*(n_nodes+1)/2)
        
        # calculate direct communication resistance between agents
        for k in range(n_pairs):
            i,j = linear_index_to_lower_triangular(k)
            if i==1 and j==0: continue # enforce that origin & destination don't directly connect
            if check_2way_communicability(nodes[i], nodes[j]):
                comm_net.add_edge(i, j)

        # systemic reward is inverse of resistance (conductance)
        return comm_net


    def observation(self, agent, world):
        ''' call observation function based on type of observation function '''
        if self.observation_type == "direct":
            return self._direct_observation(agent, world)
        elif self.observation_type == "histogram":
            return self._histogram_observation(agent, world)
        else:
            raise Exception("Unrecognized observation type: {}".format(self.observation_type))


    def _direct_observation(self, agent, world):
        # get positions of all entities in this agent's reference frame

        def communications_observed(other_comm_node):
            ''' Communication between agents is just the conductance
            Notes:
             - inverse of comm resistance (i.e. conductance) used so that we can use
             zero for out of range comms
             - noisy measurement of heading
             - TODO: observation of failures
            '''

            # check if node is terminated
            is_terminated = 0
            if isinstance(other_comm_node, MortalAgent) and other_comm_node.terminated:
                is_terminated = 1

            dx = dy = dvx = dvy = 0.
            if not is_terminated:
                dx, dy = delta_pos(other_comm_node, agent)
                dvx, dvy = delta_vel(other_comm_node, agent)

            comms = [is_terminated, dx, dy, dvx, dvy]

            # set comms to zero if out for range
            # if distance(agent, other_comm_node) > agent.max_observation_distance:
            if not check_2way_communicability(agent, other_comm_node):
                comms = [0] * len(comms)

            return comms

        # Observe communication terminals
        terminals = (world.origin_terminal_landmark.state.p_pos.tolist() + 
                    world.destination_terminal_landmark.state.p_pos.tolist())

        # comm_nodes are origin and destination terminals and unterminated agents
        comm_nodes = []
        comm_nodes.extend([a for a in world.agents if a is not agent])
        communications = format_observation(observe = communications_observed, 
                                            objects = comm_nodes,
                                            num_observations = self.num_agents-1, 
                                            observation_size = 2*world.dim_p + 1)


        # package observation
        obs = np.asarray([agent.terminated] + agent.state.p_vel.tolist()  + agent.state.p_pos.tolist() + terminals + communications)

        return obs

    def _histogram_observation(self, agent, world):
        # get positions of all entities in this agent's reference frame

        # Observe communication terminals
        terminals = (world.origin_terminal_landmark.state.p_pos.tolist() + 
                    world.destination_terminal_landmark.state.p_pos.tolist())

        # get histogram of agent observations
        agent_histogram_2d, observed_terminations_2d = agent_histogram_observation(
            agent, world.agents, _MAX_OBSERVATION_DISTANCE, _N_RADIAL_BINS, _N_ANGULAR_BINS)

        # flatten histogram to 1d list
        agent_histogram = [val for sublist in agent_histogram_2d for val in sublist]

        # flatten, truncate/pad observed terminations to fixed length
        observed_terminations = [val for sublist in observed_terminations_2d for val in sublist]
        observed_terminations = truncate_or_pad(observed_terminations, 2*_N_OBSERVED_TERMINATIONS)

        # package new observation
        new_obs = np.asarray([agent.terminated] + agent.state.p_vel.tolist() + agent.state.p_pos.tolist() + terminals + agent_histogram + observed_terminations)

        # append previous observation for velocity estimation
        if agent.previous_observation is None:
            agent.previous_observation = 0.0*new_obs
        obs = np.append(new_obs, agent.previous_observation)
        agent.previous_observation = new_obs

        return obs


class ScenarioHeuristicComputer(ScenarioHeuristicAgentTrainer):
    ''' representation of an individual agent's embedded processor and memory tailor
    Notes:
     - This is meant to be used as a scenario-specific alternative to
     the more general purpose, scenario-agnostic "trainers". It can hold an agents model
     of the world (transition and reward functions), policy, and learning process, if any.
    '''
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, **kwargs):
        ScenarioHeuristicAgentTrainer.__init__(self, name, model, obs_shape_n, act_space_n, agent_index, args)

        raise NotImplementedError()


    def get_initial_policy_distribution(self):
        ''' method for "jumpstarting" monte carlo group distribution
        '''

        raise NotImplementedError()


    def action(self, obs):
        ''' maps observation array to action forces in x,y directions
        Notes:
         - Assumes observation array formated as:
            [0:2] = agent.state.p_vel.tolist() 
            [2:4] = agent.state.p_pos.tolist() 
            [4:8] = terminals
            [8:40] = agent_histogram + 
            failures)
        '''

        raise NotImplementedError()

    def experience(self, obs, act, rew, new_obs, done, terminal):
        ''' Monte Carlo learning only record cumulative reward
        '''
        # record cumulative reward
        raise NotImplementedError()


    def preupdate(self):
        '''unused function handle compatibility with train.py
        '''
        raise NotImplementedError()

    def update(self, agents, t):
        '''unused function handle compatibility with train.py
        '''
        raise NotImplementedError()

    def group_policy_update(self, group_policy):
        '''update behavior parameters based on group policy
        '''
        raise NotImplementedError()
