# Attempting to make probabilty of connecting terminals in ergo_graph_variable
# given random placement of nodes constant over a range of possible nodes
# This requires finding a connection distance as a function of number of nodes (agents)
# while this should be possible to derive analtically, it seems faster to just
# run a bunch of trials to measure it empirically

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('..')
from particle_environments.common import SimpleNetwork, check_2way_communicability, linear_index_to_lower_triangular
from scipy.optimize import curve_fit

_D = 2.0
_BOUNDS = 1.0

def regress_func_exp_abc(x, a, b, c):
    return a * np.exp(-b * x) + c

def regress_func_inv_abc(x, a, b, c):
    return a * x**(-b) + c

def generate_probabilities(n_agents, n_radii, n_trials):
    d = _D # distance between terminals
    bounds = _BOUNDS
    n_agents = np.asarray(range(1,n_agents+1))
    # n_radii = 20
    # n_trials = 1000

    radii = {n:[] for n in n_agents}
    counts = dict()
    # counts['d'] = d
    # counts['n_agents'] = n_agents
    # counts['n_radii'] = n_radii
    # counts['n_trials'] = n_trials
    max_radius_to_test = d

    for n in n_agents:

        # radii[n] = np.linspace(d/float(n+1), d/float(n), n_radii)
        radii[n] = np.linspace(d/float(n+1), max_radius_to_test, n_radii)
        counts[n] = [[r, 0, 0.0] for r in radii[n]]

        for ri, r in enumerate(radii[n]):


            for t in range(n_trials):

                # set terminal locations
                th = np.random.uniform(0, 2.0*np.pi)
                dx = d/2.0*np.cos(th)
                dy = d/2.0*np.sin(th)
                A = np.array([dx, dy])
                B = np.array([-dx, -dy])

                # set node locations
                nodes = [A, B]
                for ni in range(n):
                    nodes.append(np.random.uniform(-bounds, bounds, 2))
                # n_nodes = len(nodes)

                # establish graph and graph edges
                graph = SimpleNetwork(nodes)
                n_pairs = int(graph.n_nodes*(graph.n_nodes+1)/2)
            
                # calculate direct communication resistance between agents
                for k in range(n_pairs):
                    i,j = linear_index_to_lower_triangular(k)
                    if i != j and np.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2) <= r:
                        graph.add_edge(i, j)

                # check terminal connection
                counts[n][ri][1] += graph.breadth_first_connectivity_search(0,1)

            # record probabilities
            counts[n][ri][2] = 100.0*counts[n][ri][1]/float(n_trials)

            # adapt maximum radius to test
            if counts[n][ri][1] == n_trials and r < max_radius_to_test:
                max_radius_to_test = r 

    return counts

def interpolate_radius_from_probability(counts, prob, n):
    return np.interp(prob, [rc[2] for rc in counts[n]], [rc[0] for rc in counts[n]])

def curve_fit_radius_vs_agents(counts, prob, regress_func):
    interp_radii = []
    n_agents = []
    for n in counts.keys():
        n_agents.append(n)
        interp_radii.append(interpolate_radius_from_probability(counts, prob, n))

    # fit regression function
    popt, pcov = curve_fit(regress_func, n_agents, interp_radii)

    return n_agents, interp_radii, popt, pcov

def plot_curves(counts, prob=None, regress_func=None):

    color_list =  plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_colors = len(color_list)

    fig1 = plt.figure("connection_probability_curves")
    for n in counts.keys():
        # plot counts (probabilities)
        color_ind =  n % n_colors
        color = color_list[color_ind]
        xdata = [rc[0] for rc in counts[n]]
        ydata = [rc[2] for rc in counts[n]]
        plt.plot(xdata, ydata, color)
    plt.xlabel('connection radius')
    plt.ylabel('connection percentage')
    plt.title('Connection Rate Between Terminals at Distance d={}\nfor Randomly Placed Nodes in {}x{} Area'.format(_D, 2*_BOUNDS, 2*_BOUNDS))
    plt.legend(counts.keys())


    if prob is not None and regress_func is not None:
        fig2 = plt.figure("probability_radius_curve")
        n_agents, interp_radii, popt, pcov = curve_fit_radius_vs_agents(counts, prob, regress_func)
        label = 'fit: '
        coef = 'a'
        for elem in popt:
            label += coef + '=%5.3f, '%elem
            coef = chr(ord(coef) + 1)

        plt.plot(n_agents, interp_radii, "*", label='data')
        # plt.plot(n_agents, regress_func(np.asarray(n_agents), *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.plot(n_agents, regress_func(np.asarray(n_agents), *popt), 'r-',label=label)
        plt.xlabel('# of connection nodes (agents)')
        plt.ylabel('apprx radii for {}% probability of connection'.format(prob))
        plt.legend()
        plt.title("Interpolated Radius for Probability of Connection = {}%".format(prob))

    plt.show()

if __name__ == '__main__':

    # generate probability counts
    n_agents=50
    n_radii=20
    n_trials = 1000
    target_probability = 5.0
    counts = generate_probabilities(n_agents=n_agents, n_radii=n_radii, n_trials=n_trials)

    # save data
    with open('counts_{}agents_{}radii_{}trials.pkl'.format(n_agents, n_radii, n_trials), 'wb') as fp:
        pickle.dump(counts, fp)

    # plot data
    plot_curves(counts, target_probability, regress_func_inv_abc)

