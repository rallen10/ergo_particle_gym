import numpy as np
from multiagent.environment import MultiAgentEnv
from particle_environments.common import check_2way_communicability
from multiagent.core import Agent, Landmark
from particle_environments.mager.world import RiskRewardLandmark, MortalAgent
from gym import spaces

class MultiAgentRiskEnv(MultiAgentEnv):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True,
                 discrete_action_space=True,
                 legacy_multidiscrete=True):

        super().__init__(world=world, 
                        reset_callback=reset_callback, 
                        reward_callback=reward_callback,
                        observation_callback=observation_callback, 
                        info_callback=info_callback,
                        done_callback=done_callback, 
                        shared_viewer=shared_viewer,
                        discrete_action_space=discrete_action_space, 
                        legacy_multidiscrete=legacy_multidiscrete)

        # overwrite action_space to use OpenAI Gym's MultiDiscrete and clarify action formulation
        # configure spaces
        # self.action_space = []
        # for agent in self.agents:
        #     total_action_space = []
        #     # physical action space
        #     if self.discrete_action_space:
        #         u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
        #     else:
        #         u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
        #     if agent.movable:
        #         total_action_space.append(u_action_space)
        #     # communication action space
        #     if self.discrete_action_space:
        #         c_action_space = spaces.Discrete(world.dim_c)
        #     else:
        #         c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
        #     if not agent.silent:
        #         total_action_space.append(c_action_space)
        #     # total action space
        #     if len(total_action_space) > 1:
        #         # all action spaces are discrete, so simplify to MultiDiscrete action space
        #         if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
        #             act_space = spaces.MultiDiscrete([max(0,act_space.n-1) for act_space in total_action_space])
        #         else:
        #             act_space = spaces.Tuple(total_action_space)
        #         self.action_space.append(act_space)
        #     else:
        #         self.action_space.append(total_action_space[0])
        #     agent.action.c = np.zeros(self.world.dim_c)


    def step(self, action_n):
        ''' time step of environment with OpenAI formatted ouput (observation, reward, done, info)
        Notes:
         - Needed to completely overwrite MultiAgentEnv.step function due to multiple discrepancies
        '''

        # ensure consistency in size of action list and agents
        assert len(action_n) == self.n
        assert not any([any(np.isnan(a)) for a in action_n])

        # initialize return data
        obs_n = [None]*self.n
        reward_n = [0.0]*self.n
        done_n = [None]*self.n
        info_n = {'n': [None]*self.n}


        # set action for each agent
        self.agents = self.world.policy_agents
        for i, agent in enumerate(self.agents):
            if agent.terminated:
                # skip agents that have been terminated
                continue
            self._set_action(action_n[i], agent, self.action_space[i])

        # advance world state
        self.world.step()

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            # if agent.terminated:
            #     # if agent is terminate, mark done and skip to next
            #     done_n[i] = True
            #     continue
            obs_n[i] = self._get_obs(agent)
            reward_n[i] = self._get_reward(agent)
            done_n[i] = self._get_done(agent)
            info_n['n'][i] = self._get_info(agent)
            # centralized_state = self._get_centralized_state()


        # handle systemic or shared rewards
        if self.systemic_rewards:
            # give all agents the same system-wide reward (distinct from summing individual rewards)
            reward_n = self.reward_callback(None, self.world, systemic_call=True)
            if len(reward_n) != self.n:
                raise Exception('Improperly formatted systemic reward. Expected length {}, received length {}'.format(self.n, len(reward_n)))
            
        if self.shared_reward:
            # all agents get total reward in cooperative case
            sum_reward = np.sum(reward_n)
            reward_n = [sum_reward] * self.n


        return obs_n, reward_n, done_n, info_n

    def get_joint_state(self):
        ''' returns global/centralized/joint/(whatever you want to call it) state of all entities in the world '''
        global_state = []
        labels = []
        for ent in self.world.entities:
            ent_state = np.concatenate((ent.state.p_pos, ent.state.p_vel))
            # Note: communications are not captured in the global state because they are consider a form
            # of an action, which is also not considered part of the state

            # capture hazard or terminated as part of state
            if isinstance(ent, Agent):
                if isinstance(ent, MortalAgent):
                    ent_state = np.concatenate((ent_state, [ent.terminated]))
                else:
                    ent_state = np.concatenate((ent_state, [0]))
            elif isinstance(ent, Landmark):
                if isinstance(ent, RiskRewardLandmark):
                    ent_state = np.concatenate((ent_state, [ent.is_hazard]))
                else:
                    ent_state = np.concatenate((ent_state, [0]))
            else:
                raise Exception("Unrecognized entity type: {}".format(type(ent)))

            global_state.append(ent_state)

            # capture labels
            labels.append(ent.name)

        return {"state": global_state, "labels": labels}


    def render(self, mode='human'):
        ''' render environment
        Notes:
         - Needed to completely overwrite MultiAgentEnv.step function due to multiple discrepancies
        '''

        # print messages
        if mode == 'human' and self.world.dim_c > 0:
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        self.render_geoms = None
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                if hasattr(entity, 'terminated') and entity.terminated:
                    # render as x
                    xvertices = entity.size*np.array([[-1,1],[1,-1],[0,0],[1,1],[-1,-1]])
                    geom = rendering.make_polyline(xvertices)
                else:
                    # render as o
                    geom = rendering.make_circle(entity.size)

                # render connections between entities
                connection_geoms = []
                if hasattr(self.world, 'render_connections') and self.world.render_connections:
                    for ag in self.world.agents:
                        if entity == ag: continue
                        if hasattr(entity, 'ignore_connection_rendering') and entity.ignore_connection_rendering: continue
                        if check_2way_communicability(entity, ag):
                            connection_geoms.append(
                                rendering.make_polyline(
                                    [entity.state.p_pos.tolist(), ag.state.p_pos.tolist()]))

                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                    if hasattr(entity, 'terminated') and entity.terminated:
                        geom.set_color(1,0,0, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms.extend(connection_geoms)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

