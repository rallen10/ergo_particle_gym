
import numpy as np
from copy import deepcopy
from bisect import bisect, insort
from particle_environments.common import distance, delta_pos
from particle_environments.mager.world import SensingLimitedMortalAgent

def format_observation(observe, objects, num_observations, observation_size, sort_key=None):
    """
    Formats a list of observations
    :param observe: function that computes observation given an object from objects
    :param objects: list of objects
    :param num_observations: # of observations, truncation or padding will be applied to meet this
    :param observation_size: length of observation arrarys, list, tuples
    :param sort_key: function of (object, observation) that gives a value to sort by (in ascending order)
    :return: a 1-d array of formatted observation values
    """
    assert callable(observe)
    assert num_observations >= 1
    # TODO: handle case where no objects are present (e.g. all other agents terminated)
    #assert len(objects) > 0

    # sorting
    observations = []
    if sort_key is None:
        for obj in objects:
            observation = observe(obj)
            if isinstance(observation, (list, tuple)):
                assert len(observation) == observation_size
                observations += observation
            else:
                observations.append(observation)
    else:
        unsorted_observations = []
        for obj in objects:
            unsorted_observations.append( (obj, observe(obj)) ) # appending tuple
        sorted_observations = sorted(unsorted_observations, key=lambda o: sort_key(o[0]))
        for obj, observation in sorted_observations:
            if isinstance(observation, (list, tuple)):
                assert len(observation) == observation_size
                observations += observation
            else:
                observations.append(observation)

    # padding/truncation
    missing_objects = num_observations - len(objects)
    if missing_objects > 0:
        # TODO: I think non-communication should send None instead of zero, because zero has real meaning
        #   however this causes a problem with action function
        padding = [0]*observation_size*missing_objects
        observations += padding
        return observations
    else:
        return observations[:num_observations*observation_size]

def agent_histogram_observation(cur_agent, agents, obs_distance, n_radial_bins, n_angular_bins):
    ''' generate observation histogram of agents and list of terminated agents
    '''

    # generate radial histogram bins based on sensing limitations
    bin_depth = obs_distance/10.0
    radial_bins = np.logspace(np.log10(bin_depth), np.log10(obs_distance), num=n_radial_bins)

    # generate angular histogram bins
    bin_angle = 2.0*np.pi/float(n_angular_bins)
    angular_bins = np.linspace(bin_angle/2.0, 2*np.pi - bin_angle/2.0, num=n_angular_bins)
    agent_histogram_2d = np.array([[0]*n_angular_bins]*n_radial_bins)


    # establish observation of failures
    observed_terminations_2d = []
    observed_terminations_dists = []

    # count agents in each bin
    for a in agents:
        dist = distance(a, cur_agent)

        # skip if agent is agent
        if a == cur_agent:
            continue

        # record observed termination
        if a.terminated:
            insert_index = bisect(observed_terminations_dists, dist)
            observed_terminations_dists.insert(insert_index, dist)
            observed_terminations_2d.insert(insert_index, delta_pos(a, cur_agent))
            continue

        # skip if outside of observation range
        if dist > obs_distance or (
            isinstance(cur_agent, SensingLimitedMortalAgent) and not cur_agent.is_entity_observable(a)):
            continue

        # find radial bin
        rad_bin = np.searchsorted(radial_bins, dist)

        # calculate angle
        dx, dy = delta_pos(a, cur_agent)
        ang = np.arctan2(dy, dx)
        if ang < 0:
            ang += 2*np.pi

        # find angular bin
        ang_bin = np.searchsorted(angular_bins, ang)
        if ang_bin == n_angular_bins:
            ang_bin = 0

        # add count to histogram
        agent_histogram_2d[rad_bin][ang_bin] = agent_histogram_2d[rad_bin][ang_bin] + 1

    return agent_histogram_2d, observed_terminations_2d

def landmark_histogram_observation(cur_agent, landmarks, obs_distance, n_radial_bins, n_angular_bins):
    ''' generate observation histogram of landmarks and list of hazards 
    '''

    # generate radial histogram bins based on sensing limitations
    bin_depth = obs_distance/10.0
    radial_bins = np.logspace(np.log10(bin_depth), np.log10(obs_distance), num=n_radial_bins)

    # generate angular histogram bins
    bin_angle = 2.0*np.pi/float(n_angular_bins)
    angular_bins = np.linspace(bin_angle/2.0, 2*np.pi - bin_angle/2.0, num=n_angular_bins)
    landmark_histogram_2d = np.array([[0]*n_angular_bins]*n_radial_bins)

    # establish observation of failures
    observed_hazards_2d = []
    observed_hazards_dists = []

    # count agents in each bin
    for lm in landmarks:
        dist = distance(lm, cur_agent)

        # check if landmark is giving reward or hazard warning
        # NOTE: This modifies the landmarks list
        if dist < lm.size:
            if lm.is_hazard:
                lm.hazard_tag = 1.0
                lm.color = np.array([1.1, 0, 0])

        # record observed hazard
        if lm.hazard_tag > 0.0:
            insert_index = bisect(observed_hazards_dists, dist)
            observed_hazards_dists.insert(insert_index, dist)
            observed_hazards_2d.insert(insert_index, delta_pos(lm, cur_agent))
            continue

        # skip if outside of observation range
        if dist > obs_distance or (
            isinstance(cur_agent, SensingLimitedMortalAgent) and not cur_agent.is_entity_observable(lm)):
            continue

        # find radial bin
        rad_bin = np.searchsorted(radial_bins, dist)

        # calculate angle
        dx, dy = delta_pos(lm, cur_agent)
        ang = np.arctan2(dy, dx)
        if ang < 0:
            ang += 2*np.pi

        # find angular bin
        ang_bin = np.searchsorted(angular_bins, ang)
        if ang_bin == n_angular_bins:
            ang_bin = 0

        # add count to histogram
        landmark_histogram_2d[rad_bin][ang_bin] = landmark_histogram_2d[rad_bin][ang_bin] + 1

    return landmark_histogram_2d, observed_hazards_2d