import numpy as np
# import logging
from multiagent.core import World, Agent, Landmark
from copy import deepcopy
from particle_environments.common import (
    is_collision, delta_pos, distance,
    RewardFunction, RadialPolynomialRewardFunction2D, 
    RiskFunction, NoneRiskFunction)

class RiskRewardLandmark(Landmark):
    def __init__(self, risk_fn=None, reward_fn=None):
        ''' Landmark that can reward and/or risk failure of agent in proximity
        '''
        super().__init__()

        self.risk_fn = deepcopy(risk_fn)
        self.reward_fn = deepcopy(reward_fn)

        if self.risk_fn is None:
            self.risk_fn = NoneRiskFunction()

        if not isinstance(self.reward_fn, RewardFunction):
            raise Exception('Invalid reward function')
        if not isinstance(self.risk_fn, RiskFunction):
            raise Exception('Invalid risk function')

        self.is_hazard = not isinstance(self.risk_fn, NoneRiskFunction)


class TemporarilyObservableRiskRewardLandmark(RiskRewardLandmark):
    def __init__(self, risk_fn=None, reward_fn=None, init_observe_clock=0.0, observe_duration=0.5):
        ''' Landmark that rewards and/or risks nearby agents and may/may not be observed
        Args:
         - observe_clock: clock determining how lock the landmark will be observable 
         - cumulative_distributed_reward: total reward distributed by landmark over all agents and timesteps
        '''
        super().__init__(risk_fn, reward_fn)

        # check reward function
        if not isinstance(self.reward_fn, RadialPolynomialRewardFunction2D):
            raise Exception('Invalid Reward Function. TemporarilyObservableRiskRewardLandmark currently only supports RadialPolynomialRewardFunction2D reward functions')

        self.init_observe_clock = init_observe_clock
        self.cumulative_distributed_reward = 0.0
        self.observe_duration = observe_duration
        self.observe_clock = None
        self.reward_count = None
        self.reset_landmark()

    def reset_landmark(self):
        ''' reset landmark to its init state '''
        self.reset_observe_clock()
        self.reset_cumulative_distributed_reward()

    def reset_observe_clock(self):
        ''' reset observe_clock to make observable
        '''

        # check init observe_clock and reward_count
        if self.init_observe_clock < 0:
            raise Exception('Invalid init_observe_clock {}, cannot be negative.'.format(self.init_observe_clock))

        self.observe_clock = self.init_observe_clock

    def set_observe_clock(self):
        ''' set the observe_clock to observe_duration to make landmark observable
        '''

        if self.observe_duration < 0:
            raise Exception('Invalid observe_duration {}, cannot be negative.'.format(self.observe_duration))

        self.observe_clock = self.observe_duration


    def decrement_observe_clock(self, delta_t=None):
        ''' update observe_clock based on time step 
        '''
        if not isinstance(delta_t, float) or delta_t < 0 :
            raise Exception(
                'Invalid observation_clock decrement time type. expected positive float, received {}'.format(
                    type(delta_t)))

        self.observe_clock = max(self.observe_clock - delta_t, 0.0)

    def is_observable(self):
        ''' return True if landmark can currently be observed
        '''
        return bool(self.observe_clock)

    def update_cumulative_distributed_reward(self, reward_distributed):
        ''' update the amount of reward landmark has distributed to all agents
        '''
        self.cumulative_distributed_reward += reward_distributed

    def reset_cumulative_distributed_reward(self):
        self.cumulative_distributed_reward = 0.0

class HazardousWorld(World):
    def __init__(self, 
                    collision_termination_probability=0.7, 
                    flyoff_termination_radius = 1e10,
                    flyoff_termination_speed = 1e10,
                    spontaneous_termination_probability = 0.0):
        '''
        Args:
         - collision_termination_probability: probability collision of agents causes termination, if collide=true
         - flyoff_termination_radius: radius at which agents automatically terminate if they move this far from origin. 
         - flyoff_termination_speed: speed at which agents automatically terminate if they move this fast
         - spontaneous_termination_probability: probability that an agent will spontaneously terminated at a give timestep

        Notes:
         - flyoff termination ued to prevent numerical errors of very large number, guide training toward relevant domain, 
            and is eaiser to implement than hard constraints on movement of agents
         - spontaneous termination used to generate sufficient training examples for terminated baseline counterfactual values
        '''
        super().__init__()
        self.collision_termination_probability = collision_termination_probability
        self.flyoff_termination_radius = flyoff_termination_radius
        self.flyoff_termination_speed = flyoff_termination_speed
        self.spontaneous_termination_probability = spontaneous_termination_probability

    def step(self):
        super().step()

        # decrement observability and reward function of certain landmarks
        for landmark in self.landmarks:
            if isinstance(landmark, TemporarilyObservableRiskRewardLandmark):
                landmark.decrement_observe_clock(self.dt)
                # landmark.update_reward_function()

        # check for casualties
        for i, agent in enumerate(self.agents):

            # skip terminated agents since already a casualty
            if agent.terminated:
                # self.render_geoms = None # force resetting of object rendering list
                continue

            # check for destruction by hazard
            for landmark in self.landmarks:
                if isinstance(landmark, RiskRewardLandmark):
                    rel_pos = delta_pos(agent, landmark)
                    agent_failure = landmark.risk_fn.sample_failure(rel_pos[0], rel_pos[1])
                    if agent_failure:
                        agent.terminate_agent()
                        
                        # If landmark caused a failure, make landmark observable
                        if isinstance(landmark, TemporarilyObservableRiskRewardLandmark):
                            landmark.set_observe_clock()
                        break

            # check for destruction by flyoff or excess velocity
            # NOTE: this is a hack, certain scenarios were causing early training to send
            #       agents to positions and velocities that overflowed floating point operations. this caused
            #       the policy to return NaN actions. By using the termination state, we shouldn't 
            #       need to impose "walls" on the motion of agents, instead just cause them to 
            #       be terminated if they fly too far off
            if (np.linalg.norm(agent.state.p_pos) > self.flyoff_termination_radius or
                np.linalg.norm(agent.state.p_vel) > self.flyoff_termination_speed):
                agent.terminate_agent()

            # check if spontaneous destruction of agent
            if np.random.rand() < self.spontaneous_termination_probability:
                agent.terminate_agent()

            # recheck for terminated agents and skip
            if agent.terminated:
                # self.render_geoms = None # force resetting of object rendering list
                continue

            # check for collisions
            for other_agent in self.agents[i+1:]:

                if other_agent.terminated:
                    # skip if other agent alread terminated
                    continue

                if is_collision(agent, other_agent):
                    if np.random.random() < self.collision_termination_probability:
                        # terminated other agent
                        other_agent.terminate_agent()
                        # logging.info('fatal crash')
                    if np.random.random() < self.collision_termination_probability:
                        # terminated current agent
                        agent.terminate_agent()
                        # logging.info('fatal crash')
                        break


class MortalAgent(Agent):
    '''
    Agent that can be destroyed/eliminate/terminated
    '''
    def __init__(self):
        super().__init__()
        self.terminated = False

    def terminate_agent(self):
        ''' set agent object vars to appropriate terminated state'''
        self.terminated = True
        self.movable = False
        self.silent = True
        self.blind = True
        self.collide = False
        # self.state.p_pos = None
        # self.state.p_vel = None
        self.state.c = None
        self.action.u = None
        self.action.c = None

class SensingLimitedMortalAgent(MortalAgent):
    ''' Mortal agent with limited communication and observation capabilities
    '''
    def __init__(self, observation_range, transmit_range):
        super().__init__()

        if observation_range < 0 or transmit_range < 0:
            raise Exception('observation_range and transmit_range must be non-negative')

        self.observation_range = observation_range
        self.transmit_range = transmit_range
        self.deaf = False

        # enable the concept of communication without adding a separate communication
        # channel to the action space of the environment
        self.passive_communication = True

    def is_observable(self):
        ''' Check that agent is observable by others
        Note:
         - for now, this is always true, even if terminated
        '''
        return True

    def is_entity_observable(self, entity):
        ''' Check if other entity can be observed
        '''

        # check valid observation range
        if self.observation_range < 0:
            raise Exception('observation_range must be non-negative')

        # check if agent is blind
        if self.blind:
            return False

        # check that other entity is not hidden
        if hasattr(entity, 'is_observable') and callable(entity.is_observable):
            if not entity.is_observable:
                return False

        # check max observation distance    
        return distance(self, entity) <= self.observation_range

    def is_entity_transmittable(self, entity):
        ''' Check if entity is within transmission range
        '''

        # check valid observation range
        if self.transmit_range < 0:
            raise Exception('transmit_range must be non-negative')

        # check if agent is silent
        if self.silent:
            return False

        # check that entity is not deaf
        if hasattr(entity, 'deaf') and entity.deaf:
            return False

        # check transmission range
        return distance(self, entity) <= self.transmit_range

    # def is_entity_communicable(self, entity):
    #     ''' Check if two-way communication is possible
    #     '''

    #     # check that both are in each other's transmission range
    #     if hasattr(entity, 'is_entity_transmittable') and callable(entity.is_entity_transmittable):
    #         return (self.is_entity_transmittable(entity) and entity.is_entity_transmittable(self))
    #     else:
    #         # entity assumed to have unlimited transmission range
    #         return self.is_entity_transmittable(entity)

