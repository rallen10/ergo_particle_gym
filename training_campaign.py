# Used to run a campaign of training scenarios across a range of parameters
from collections import OrderedDict

scenario_params = OrderedDict()
scenario_params['scenarios'] = ['ergo_spread_variable', 'ergo_graph_variable']
scenario_params['reward_function'] = ['identical', 'local']
scenario_params['n_agents'] = [1, 3, 5, 10]

learning_params = OrderedDict()
learning_params['value_function'] = ['centralized', 'private']
learning_params['learning_rate'] = [(3e-4, 3e-4), (1e-5, 1e-3)]
learning_params['batch_size'] = [16, 64, 256, 1024]
learning_params['cliprange'] = [0.1, 0.2, 0.3]
learning_params['entropy_coef'] = [0.001, 0.01, 0.1]
learning_params['activation'] = ['relu', 'tanh', 'elu']
learning_params['n_minibatches'] = [4, 16]
learning_params['n_opt_epochs'] = [4, 8]
learning_params['n_layers'] = [2]
learning_params['n_units'] = [64]
learning_params['gamma'] = [0.95]
# learning_params['lambda_coef'] = [1.0]
learning_params['value_coef'] = [0.5]
learning_params['max_episode_len'] = [50]
learning_params['num_episodes'] = [100000]
learning_params['save_rate'] = [1000]

n_experiments = 1
for k in scenario_params.keys():
    n_experiments *= len(scenario_params[k])
for k in learning_params.keys():
    n_experiments *= len(learning_params[k])

worst_case_seconds_per_episode = 1.0
best_case_seconds_per_episode = 0.15
worst_case_sequential_compute_time = float(n_experiments*max(learning_params['num_episodes'])*worst_case_seconds_per_episode)
best_case_sequential_compute_time = float(n_experiments*min(learning_params['num_episodes'])*best_case_seconds_per_episode)
approximate_space_per_experiment = 100



if __name__ == '__main__':
    print('Total Experiments: {}'.format(n_experiments))
    print('worst case expected compute time: {} days'.format(
        worst_case_sequential_compute_time/(24.0*3600.0)))
    print('best case expected compute time: {} days'.format(
        best_case_sequential_compute_time/(24.0*3600.0)))
    print('approximate total disk space used = {} Gb'.format(n_experiments*approximate_space_per_experiment/1000.0))