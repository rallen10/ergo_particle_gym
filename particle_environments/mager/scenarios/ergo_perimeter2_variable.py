""" Module for risk-exploiting reward perimeter scenario
Environment will contain:
- A 2D reward distribution. The objective of the multi-agent system is to form a "perimeter" around the reward distribution
    such that is maximize the surface integral of the reward distribution across the surface of the largest convex region defined by the agents
- Rewards will be normalized by the total, true integral of the reward function
- The reward distribution goes negative toward the outskirts of the domain, thus the optimal policy would distribute agents along the curve
    of where the reward function crosses zero
- The reward function has a corresponding risk function; i.e. when the reward becomes non-zero, so does the risk of agent failure
- If an agent fails, it is still used as part of the reward calculation, but it is not able to move for the rest of the episode
Agents will:
- observe the position of other agents as well as thier local measurement of the risk/reward function


- Scenario is derived from ergo_perimeter_small making the following 
    settable parameters instead of hard coded
    - number of agents
    - whether rewards are shared or "local"
    - whether observations are direct per entity or histogram based
    - Hazard failure risk
    - Collision failure risk
"""

import numpy as np
from bisect import bisect
from shapely.geometry import Polygon, Point, MultiPoint
from multiagent.scenario import BaseScenario
from particle_environments.mager.world import MortalAgent, HazardousWorld, RiskRewardLandmark
from particle_environments.mager.observation import format_observation
from particle_environments.common import is_collision, distance, delta_pos, delta_vel
from particle_environments.common import ExtendedRadialPolynomialRewardFunction2D as ExtendedRadialReward
from particle_environments.common import RadialBernoulliRiskFunction2D as RadialRisk
from particle_environments.common import truncate_or_pad
from rl_algorithms.scenariolearning import ScenarioHeuristicAgentTrainer


# Scenario Parameters
_MAX_COMMUNICATION_DISTANCE = np.inf
_MAX_OBSERVATION_DISTANCE = 2.0
_AGENT_SIZE = 0.01
_LANDMARK_SIZE = 0.2
_AGENT_OBSERVATION_LEN = 6
_LANDMARK_OBSERVATION_LEN = 1
_PEAK_REWARD = 1.0
_TERMINATION_REWARD = -0.0
_N_RADIAL_BINS = 4
_N_ANGULAR_BINS = 8
_N_OBSERVED_TERMINATIONS = 3
_PRECISION_WEIGHT = 1.0
_COVERAGE_WEIGHT = 0.01


# _LANDMARKS = []
# _LANDMARKS.append(
#     RiskRewardLandmark( risk_fn=RadialRisk(_LANDMARK_SIZE), reward_fn=ExtendedRadialReward(_LANDMARK_SIZE, 1.0)))
# _N_LANDMARKS = len(_LANDMARKS)
# _POSITIVE_REGION_INTEGRAL = _LANDMARKS[0].reward_fn.get_radial_integral(_LANDMARK_SIZE)

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

        # set member vars
        self.num_agents = num_agents
        self.num_hazards = num_hazards
        self.identical_rewards = identical_rewards
        self.observation_type = observation_type
        self.hazard_risk = hazard_risk
        self.collision_risk = collision_risk

        # Create landmark
        landmarks = []
        if self.num_hazards == 0:
            landmarks.append(
                RiskRewardLandmark(risk_fn=None, 
                reward_fn=ExtendedRadialReward(_LANDMARK_SIZE, _PEAK_REWARD)))
        else:
            landmarks.append(
                RiskRewardLandmark(risk_fn=RadialRisk(_LANDMARK_SIZE, self.hazard_risk), 
                reward_fn=ExtendedRadialReward(_LANDMARK_SIZE, _PEAK_REWARD)))
        self.scenario_landmarks = landmarks
        self.n_landmarks = len(self.scenario_landmarks)

    def make_world(self):
        world = HazardousWorld(
                    collision_termination_probability=self.collision_risk, 
                    flyoff_termination_radius = 10.0,
                    flyoff_termination_speed = 50.0,
                    spontaneous_termination_probability = 1.0/(8.0*50.0*self.num_agents))

        # set scenario-specific world parameters
        world.collaborative = True
        world.systemic_rewards = True
        world.identical_rewards = self.identical_rewards
        world.dim_c = 0 # observation-based communication
        world.termination_reward = _TERMINATION_REWARD
        world.render_connections = True

        # add landmarks to world
        world.landmarks = []
        for lm in self.scenario_landmarks:
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
        world.agents = [MortalAgent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_%d' % i
            agent.collide = False
            agent.blind = False
            agent.silent = True
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
        raise NotImplementedError()

    def _identical_systemic_reward(self, world):
        ''' reward all agents the same for perimeter forming around reward function
        Notes:
         - computes surface integral over surface defined by convex hull of agents
        '''
        assert self.identical_rewards == True
        assert world.identical_rewards == True

        # reward function is designed for a single landmark case
        assert len(world.landmarks) == 1
        assert len(self.scenario_landmarks) == 1

        # get convex hull created by agents
        landmark_circle = Point(world.landmarks[0].state.p_pos).buffer(world.landmarks[0].size)
        agent_positions = [ag.state.p_pos for ag in world.agents]
        agent_hull = MultiPoint(agent_positions).convex_hull
        if agent_hull.geom_type == 'LineString':
            # if hull is degenerate (e.g. all points collinear), then set integral to zero
            return [0.0]*self.num_agents

        assert agent_hull.geom_type == 'Polygon', "Unexpected geometry type for convex hull: {}".format(agent_hull.geom_type)
        assert agent_hull.is_valid, "Invalid Polygon for convex hull"
        assert landmark_circle.is_valid, "Invalid Polygon for convex hull"

        # find intersection of convex hull and landmark
        landmark_coverage = agent_hull.intersection(landmark_circle)

        # compute reward
        A = agent_hull.area
        B = landmark_circle.area
        AB = landmark_coverage.area
        w_p = _PRECISION_WEIGHT;    assert w_p >= 0.0
        w_c = _COVERAGE_WEIGHT;     assert w_c >= 0.0
        reward_signal = w_p*AB/A - w_c*(B-AB)/B
        assert (reward_signal >= -w_c and reward_signal <= w_p)

        # normalize by number of agents
        reward_signal /= float(self.num_agents)

        # assign to all agents
        reward_n = [reward_signal]*self.num_agents

        return reward_n

    def observation(self, agent, world):
        ''' call observation function based on type of observation function '''
        if self.observation_type == "direct":
            return self._direct_observation(agent, world)
        elif self.observation_type == "histogram":
            return self._histogram_observation(agent, world)
        else:
            raise Exception("Unrecognized observation type: {}".format(self.observation_type))

    def _histogram_observation(self, agent, world):
        # get positions of all entities in this agent's reference frame

        # check reward signal received from landmark
        landmark_sensor_reading = 0.0
        for lm in world.landmarks:
            landmark_sensor_reading += lm.reward_fn.get_value(*agent.state.p_pos)

        # Format agent histograms
        bin_depth = _MAX_OBSERVATION_DISTANCE/10.0
        radial_bins = np.logspace(np.log10(bin_depth), np.log10(_MAX_OBSERVATION_DISTANCE), num=_N_RADIAL_BINS)
        bin_angle = 2.0*np.pi/float(_N_ANGULAR_BINS)
        angular_bins = np.linspace(bin_angle/2.0, 2*np.pi - bin_angle/2.0, num=_N_ANGULAR_BINS)
        agent_histogram_2d = np.array([[0]*_N_ANGULAR_BINS]*_N_RADIAL_BINS)
        reward_histogram_2d = np.array([[0.0]*_N_ANGULAR_BINS]*_N_RADIAL_BINS)

        # establish observation of failures
        observed_terminations_2d = []
        observed_terminations_dists = []

        # count agents in each bin
        for a in world.agents:
            dist = distance(a, agent)

            # skip if agent is agent
            if a == agent:
                continue

            # record observed termination
            if a.terminated:
                insert_index = bisect(observed_terminations_dists, dist)
                observed_terminations_dists.insert(insert_index, dist)
                observed_terminations_2d.insert(insert_index, delta_pos(a, agent))
                # don't "continue", record terminated agent in histogram like live agent

            # find radial bin
            rad_bin = np.searchsorted(radial_bins, dist)
            if rad_bin == _N_RADIAL_BINS:
                # agent is too far away and observation is not stored
                continue

            # calculate angle
            dx, dy = delta_pos(a, agent)
            ang = np.arctan2(dy, dx)
            if ang < 0:
                ang += 2*np.pi

            # find angular bin
            ang_bin = np.searchsorted(angular_bins, ang)
            if ang_bin == _N_ANGULAR_BINS:
                ang_bin = 0

            # add count to histogram
            agent_histogram_2d[rad_bin][ang_bin] = agent_histogram_2d[rad_bin][ang_bin] + 1

            # add aggregate landmark sensor reading to histogram
            # Note: should not need to compute average reading over agents in bin
            #   because neural net should be able to learn to do this using agent count
            #   histogram
            for lm in world.landmarks:
                reward_histogram_2d[rad_bin][ang_bin] += lm.reward_fn.get_value(*a.state.p_pos)

        # flatten histogram to 1d list
        agent_histogram = [val for sublist in agent_histogram_2d for val in sublist]

        # flatten reward histogram to 1d list and compute average
        reward_histogram = [val for sublist in reward_histogram_2d for val in sublist]

        # flatten, truncate/pad observed terminations to fixed length
        observed_terminations = [val for sublist in observed_terminations_2d for val in sublist]
        observed_terminations = truncate_or_pad(observed_terminations, 2*_N_OBSERVED_TERMINATIONS)

        # package new observation
        new_obs = np.asarray([agent.terminated] + agent.state.p_vel.tolist() + agent.state.p_pos.tolist() + [landmark_sensor_reading] + agent_histogram + reward_histogram + observed_terminations)

        # append previous observation for velocity estimation
        if agent.previous_observation is None:
            agent.previous_observation = 0.0*new_obs
        obs = np.append(new_obs, agent.previous_observation)
        agent.previous_observation = new_obs

        return obs

    def _direct_observation(self, agent, world):
        # get positions of all entities in this agent's reference frame

        def observe_agents(other_agent):
            ''' fill in information communicated/observed between agents
            '''

            # check if node is terminated
            is_terminated = 0
            if isinstance(other_agent, MortalAgent) and other_agent.terminated:
                is_terminated = 1

            # relative speed and position of other agent
            # dx = dy = dvx = dvy = 0.
            # if not is_terminated:
            dx, dy = delta_pos(other_agent, agent)
            dvx, dvy = delta_vel(other_agent, agent)

            # get local reward function at position of other agent
            other_landmark_sensor_reading = 0.0
            for lm in world.landmarks:
                other_landmark_sensor_reading += lm.reward_fn.get_value(*other_agent.state.p_pos)

            ag_obs = [is_terminated, dx, dy, dvx, dvy, other_landmark_sensor_reading]
            assert(len(ag_obs) == _AGENT_OBSERVATION_LEN)
            return ag_obs

        def observe_landmarks(landmark):
            ''' fill in information observed about landmarks
            '''
            # ld_obs = delta_pos(landmark, agent).tolist()

            # # check if within observation range and is observable
            # d = distance(landmark, agent)
            # if d > world.max_communication_distance:
            #     ld_obs = [0.0]*len(ld_obs)

            ld_obs = []

            # check reward signal received from landmark
            landmark_sensor_reading = 0.0
            for lm in world.landmarks:
                landmark_sensor_reading += lm.reward_fn.get_value(*agent.state.p_pos)

            ld_obs += [landmark_sensor_reading]

            assert(len(ld_obs) == _LANDMARK_OBSERVATION_LEN)
            return ld_obs

        landmark_observations = format_observation(observe = observe_landmarks,
                                                objects = world.landmarks, 
                                                num_observations = len(world.landmarks), 
                                                observation_size = _LANDMARK_OBSERVATION_LEN)

        agent_observations = format_observation(observe = observe_agents, 
                                            objects = [a for a in world.agents if a is not agent],
                                            num_observations = self.num_agents-1, 
                                            observation_size = _AGENT_OBSERVATION_LEN)

        new_obs = np.asarray([agent.terminated] + agent.state.p_vel.tolist() + agent.state.p_pos.tolist() + landmark_observations + agent_observations)

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
