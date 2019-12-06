""" Module for risk-exploiting landmark identification scenario
- Scenario is derived from ergo_spread_small.py which is, in turn, derived from simple_spread.py
- N-agents must disribute themselves to cover N-landmarks
- Reward is based on the distance from each landmark to the closest agent (identical_reward case)
- One of the landmarks is actually a hazard that can cause agents in its vicinity to be terminated. The hazardous
lanmark is unknown until one agent moves within its vicinity
- In contrast to ergo_spread_small, this scenario makes several parameters come from user inputs instead of 
    hardcoded. These user-defined inputs include
    - number of agents
    - number of hazards (0,1)
    - whether rewards are shared or "local"
    - whether observations are direct per entity or histogram based
    - Hazard failure risk
    - Collision failure risk
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
from particle_environments.mager.observation import format_observation, agent_histogram_observation, landmark_histogram_observation
from particle_environments.common import truncate_or_pad


# Scenario Parameters
_MAX_COMMUNICATION_DISTANCE = np.inf
# _AGENT_SIZE = 0.15
_LANDMARK_SIZE = 0.05
_AGENT_OBSERVATION_LEN = 5
_LANDMARK_OBSERVATION_LEN = 3
_N_RADIAL_BINS = 4
_N_ANGULAR_BINS = 8
_MAX_HISTOGRAM_OBSERVATION_DISTANCE = 1.0
_N_OBSERVED_HAZARDS = 1
_N_OBSERVED_TERMINATIONS = 5


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
        assert isinstance(num_agents, int)
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

        # create list of landmarks
        # Note: RadialReward function is not directly used for calculating reward in this scenario, thus peak value of 0.0. 
        #   non-zero radius used for numerical reasons
        landmarks = []
        for i in range(self.num_agents):
            landmarks.append(RiskRewardLandmark(risk_fn=None, reward_fn=RadialReward(1.0, 0.0)))
        for i in range(self.num_hazards):
            landmarks.append(RiskRewardLandmark(risk_fn=RadialRisk(_LANDMARK_SIZE, 0.5), reward_fn=RadialReward(1.0, 0.0)))
        self.scenario_landmarks = landmarks
        self.n_landmarks = len(self.scenario_landmarks)


    def make_world(self):
        world = HazardousWorld(collision_termination_probability=0.0)
        
        # observation-based communication
        world.dim_c = 0
        world.max_communication_distance = _MAX_COMMUNICATION_DISTANCE

        # collaborative rewards
        world.collaborative = True
        world.systemic_rewards = False
        world.identical_rewards = self.identical_rewards

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
        # find agent size as function of number of agents baselined off simple_spread 3-agent case
        agent_size = 0.15*np.sqrt(3.0/float(self.num_agents))
        # random properties for agents
        # add agents
        world.agents = [MortalAgent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_%d' % i
            agent.collide = True
            agent.silent = True
            agent.terminated = False
            agent.size = agent_size
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.previous_observation = None

        # shuffle landmarks to make sure hazard is not in same index
        shuffle(world.landmarks)
        for landmark in world.landmarks:
            # rename landmarks to preserve label ordering in joint state (see mager/environment.py:get_joint_state)
            landmark.name = 'landmark_%d' % i
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.hazard_tag = 0.0
            landmark.color = np.array([0.25, 0.25, 0.25])

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


    def reward(self, agent, world):
        if self.identical_rewards and world.identical_rewards:
            return self._identical_reward(agent, world)
        elif not self.identical_rewards and not world.identical_rewards:
            return self._local_reward(agent, world)
        else:
            raise Exception(
                "Inconsistent reward options: self.identical_rewards={} world.identical_rewards={}".format(
                self.identical_rewards, world.identical_rewards))

    def _identical_reward(self, agent, world):
        ''' use this function if all agents recieve identical rewards '''
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        assert self.identical_rewards == True
        assert world.identical_rewards == True
        rew = 0
        for lm in [l for l in world.landmarks if not l.is_hazard]:
            # dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            dists = [distance(ag,lm) for ag in world.agents]
            rew -= min(dists)

        if agent.collide:
            for a in world.agents:
                if is_collision(a, agent):
                    rew -= 1
        return rew

    def _local_reward(self, agent, world):
        ''' use this function if agents recieve separate "local" rewards, not identical '''
        raise NotImplementedError()


    def observation(self, agent, world):
        ''' call observation function based on type of observation function '''
        if self.observation_type == "direct":
            return self._direct_observation(agent, world)
        elif self.observation_type == "histogram":
            return self._histogram_observation(agent, world)
        else:
            raise Exception("Unrecognized observation type: {}".format(self.observation_type))

    def _histogram_observation(self, agent, world):
        ''' observation in histogram format of number of entities in spacial bins '''

        # get histogram of landmark observations (marking hazardous landmarks as needed)
        landmark_histogram_2d, observed_hazards_2d = landmark_histogram_observation(
            agent, world.landmarks, _MAX_HISTOGRAM_OBSERVATION_DISTANCE, _N_RADIAL_BINS, _N_ANGULAR_BINS)

        # get histogram of agent observations
        agent_histogram_2d, observed_terminations_2d = agent_histogram_observation(
            agent, world.agents, _MAX_HISTOGRAM_OBSERVATION_DISTANCE, _N_RADIAL_BINS, _N_ANGULAR_BINS)

        # flatten landmark and agent histograms to 1d list
        landmark_histogram = [val for sublist in landmark_histogram_2d for val in sublist]
        agent_histogram = [val for sublist in agent_histogram_2d for val in sublist]

        # flatten, truncate/pad observed hazards and terminations to fixed length
        observed_hazards = [val for sublist in observed_hazards_2d for val in sublist]
        observed_hazards = truncate_or_pad(observed_hazards, 2*_N_OBSERVED_HAZARDS)
        observed_terminations = [val for sublist in observed_terminations_2d for val in sublist]
        observed_terminations = truncate_or_pad(observed_terminations, 2*_N_OBSERVED_TERMINATIONS)

        # package new observation
        new_obs = np.asarray([agent.terminated] + agent.state.p_vel.tolist() + agent.state.p_pos.tolist() + landmark_histogram + observed_hazards + agent_histogram + observed_terminations)

        # append previous observation for velocity estimation
        if agent.previous_observation is None:
            agent.previous_observation = 0.0*new_obs
        obs = np.append(new_obs, agent.previous_observation)
        agent.previous_observation = new_obs

        return obs


    def _direct_observation(self, agent, world):
        ''' observation where each entity's state has it's own component of observe vector '''

        # get positions of all entities in this agent's reference frame
        def observe_agents(other_agent):
            ''' fill in information communicated/observed between agents
            '''
            # check if node is terminated
            is_terminated = 0
            if isinstance(other_agent, MortalAgent) and other_agent.terminated:
                is_terminated = 1

            dx = dy = dvx = dvy = 0.
            if not is_terminated:
                dx, dy = delta_pos(other_agent, agent)
                dvx, dvy = delta_vel(other_agent, agent)

            ag_obs = [is_terminated, dx, dy, dvx, dvy]
            assert(len(ag_obs) == _AGENT_OBSERVATION_LEN)
            return ag_obs

        def observe_landmarks(landmark):
            ''' fill in information observed about landmarks
            '''
            ld_obs = delta_pos(landmark, agent).tolist()

            # check if within observation range and is observable
            d = distance(landmark, agent)
            if d > world.max_communication_distance:
                ld_obs = [0.0]*len(ld_obs)

            # check if landmark is giving reward or hazard warning
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

        agent_observations = format_observation(observe = observe_agents, 
                                            objects = [a for a in world.agents if (a is not agent)],
                                            num_observations = self.num_agents-1, 
                                            observation_size = _AGENT_OBSERVATION_LEN)

        new_obs = np.asarray([agent.terminated] + agent.state.p_vel.tolist() + agent.state.p_pos.tolist() + landmark_positions + agent_observations)

        # append previous observation for velocity estimation
        # if agent.previous_observation is None:
        #     agent.previous_observation = 0.0*new_obs
        # obs = np.append(new_obs, agent.previous_observation)
        # agent.previous_observation = new_obs

        return new_obs


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
        '''
        raise NotImplementedError()

    def experience(self, obs, act, rew, new_obs, done, terminal):
        ''' Monte Carlo learning only record cumulative reward
        '''
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