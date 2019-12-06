""" Module for risk-exploiting landmark identification scenario
Environment will contain:
- Landmarks which can give reward and/or can cause an agent to fail with a probability based on distance (each has own function)
- A variable number of agents that can drop as agents are destroyed (at least 25 at start)
Agents will:
- know location location of landmarks but cannont directly observe the rewards/risk of landmarks except through indirect interaction
- Observe the location of failed/terminated agents
- be able to communicate with others within a certain distance (can communicate state, action, reward, and policy)
- have a high chance of failure on collision
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
        world.identical_rewards = True

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
            agent.previous_observation = None

        # shuffle landmarks to make sure hazard is not in same index
        shuffle(world.landmarks)
        for landmark in world.landmarks:
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
        ''' indicate a terminated agent as done '''
        # if agent.terminated: 
        #     return True
        # else:
        #     return False
        return False

    def centralized_state_callback(self, world):
        ''' return the world truth to return state of all agents for training purposes
        Notes:
         - This CANNOT be used during execution, only training since it assumes the ability
           to centralize observations and map them to a state of the system
        '''
        state = []
        for ag in world.agents:
            state.extend([ag.terminated])
            state.extend(ag.state.p_pos)
            state.extend(ag.state.p_vel)

        for lm in world.landmarks:
            state.extend([lm.is_hazard])
            state.extend(lm.state.p_pos)
            state.extend(lm.state.p_vel)

        return state


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
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

    # def counterfactual_reward(self, central_state):
    #     ''' reward calculated based on a given centralized state vector, not the present world
    #     '''
    #     rew = 0.0
    #     for 

    # def centralize_observations_to_partial_state_map(self, agent_observations):
    #     ''' use n-observations at a given time to generate a centralized, partial state
    #     Notes:
    #      - This CANNOT be used during execution, only training since it assumes a the ability
    #      to centralize observations and map them to a state of the system
    #     '''
    #     assert len(agent_observations) == _NUM_AGENTS
    #     state = []
    #     for ao in agent_observations:
    #         ao.extend(ao[0:5])


    def observation(self, agent, world):
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

        # if agent.terminated:
        #     # if agent is terminated, return all zeros for observation
        #     # TODO: make this more efficient. Right now it does a lot of unnecessary calcs which are all
        #     #   then set to zero. Done this way to ensure consistant array size
        #     new_obs = 0.0*new_obs

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

        # record cumulative reward
        self.cumulative_reward = 0

        # behavior params
        bp_dist = self.get_initial_policy_distribution()
        bp = dict()
        for k in bp_dist:
            bp[k] = bp_dist[k][0]
        self.behavior_params = bp

        # observation range indices
        assert(_N_LANDMARKS==4)
        oir = dict()
        oir['terminated'] = [0,0]
        oir['velocity'] = [1,2]
        oir['position'] = [3,4]
        oir['landmarks'] = [5,4+_LANDMARK_OBSERVATION_LEN*_N_LANDMARKS]
        oir['agents'] = [oir['landmarks'][1]+1, oir['landmarks'][1] + _AGENT_OBSERVATION_LEN*(_NUM_AGENTS-1)]
        assert(oir['agents'][1] == obs_shape_n[0][0]-1)
        # oir['previous_observation'] = [oir['agents'][1]+1, 2*oir['agents'][1]+1]
        # assert(oir['previous_observation'][1] == obs_shape_n[0][0]-1)
        self.observation_index_ranges = oir


    def get_initial_policy_distribution(self):
        ''' method for "jumpstarting" monte carlo group distribution
        '''

        # (mean, std)
        bp_dist = dict()
        bp_dist['landmark_proximity_gain'] = (0.05, 0.1)
        bp_dist['agent_proximity_gain'] = (0.05, 0.1)

        return bp_dist


    def action(self, obs):
        ''' maps observation array to action forces in x,y directions
        Notes:
        '''

        # rename for ease of use
        oir = self.observation_index_ranges
        fx = 0.0
        fy = 0.0

        # find closest landmark
        targets = obs[oir['landmarks'][0]:oir['landmarks'][1]]
        target_index = oir['landmarks'][0]
        min_dist = np.linalg.norm(obs[target_index:target_index+1])
        for i in range(oir['landmarks'][0], oir['landmarks'][1]+1, _LANDMARK_OBSERVATION_LEN):
            cur_dist =np.linalg.norm(obs[i:i+1])
            if cur_dist < min_dist:
                min_dist = cur_dist
                target_index = i

        # force is proportional to distance to target
        dx_tar = obs[target_index]
        fx += self.behavior_params['landmark_proximity_gain']*dx_tar
        dy_tar = obs[target_index+1]
        fy += self.behavior_params['landmark_proximity_gain']*dy_tar

        # # force proportional to distance to agents
        # for i in range(oir['terminals'][1]+1, oir['terminals'][1]+1+21, 5):
        #     dx_agt = obs[i+1]
        #     dy_agt = obs[i+2]
        #     dsqr_agt_safe = max(_ZERO_THRESHOLD**2, dx_agt**2 + dy_agt**2)
        #     fx += -(dx_agt/dsqr_agt_safe)*self.behavior_params['agent_proximity_gain']
        #     fy += -(dy_agt/dsqr_agt_safe)*self.behavior_params['agent_proximity_gain']


        # check action force is valid
        assert(not np.isnan(fx))
        assert(not np.isnan(fy))

        # Due to (bizarre) formulation of action input in MultiAgentEnv __init__ and _set_action
        # we need to create a 5-element vector where element [1] and [3] are the x,y componets of 
        # action force, respectivel
        act_force = np.zeros(5)
        act_force[1] = fx
        act_force[3] = fy

        return act_force

    def experience(self, obs, act, rew, new_obs, done, terminal):
        ''' Monte Carlo learning only record cumulative reward
        '''
        # record cumulative reward
        self.cumulative_reward += rew


    def preupdate(self):
        '''unused function handle compatibility with train.py
        '''
        pass

    def update(self, agents, t):
        '''unused function handle compatibility with train.py
        '''
        pass

    def group_policy_update(self, group_policy):
        '''update behavior parameters based on group policy
        '''
        for k in group_policy:
            self.behavior_params[k] = group_policy[k]['clambda']

        # reset cumulative reward
        self.cumulative_reward = 0
