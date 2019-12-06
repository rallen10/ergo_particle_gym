import matplotlib
# prevent rasterized fonts for submitting figures to IEEE
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import glob
import os
import copy

import itertools

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser("Results Plotter for Reinforcement Learning Experiments for Multiagent Environments")
    # list of experiment directories
    parser.add_argument("--experiments", nargs="*", help="list of experiment directories")
    parser.add_argument("--short-legend", type=str2bool, nargs='?', const=True, default=False, help="truncate legend to algorithm only")
    parser.add_argument("--custom-legend", nargs="+", default=[], help="use custom legend entries")
    parser.add_argument("--plot-losses", type=str2bool, nargs='?', const=True, default=False, help="create seprate plots for loss values")
    parser.add_argument("--max-timestep", type=float, default=-1, help="limits timesteps to plot")
    parser.add_argument("--trend-plot", type=str, default="mean", help="how to represent the moving center of rewards")
    parser.add_argument("--deviation-plot", type=str, default="std", help="how to represent devation from center")
    parser.add_argument("--smoothing-window", type=int, default=-1, help=" Number of episodes used to compute moving average window to us. -1=save_rate")
    parser.add_argument("--custom-title", type=str, default='', help="Plot title")
    parser.add_argument("--group-breaks", nargs="*", type=int, help="Indices of groupings in experiment list")
    parser.add_argument("--plot-markers", type=str2bool, nargs='?', const=True, default=False, help="plot using markers and colors, good for black & white printing")
    return parser.parse_args()


def plot_results(plotter_inputs):

    color_list =  plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_colors = len(color_list)
    color_ind = -1
    scenarios = []
    training_algorithms = []
    n_experiments = len(plotter_inputs.experiments)
    n_groups = 1
    use_grouping = False
    label_fontsize = 14
    title_fontsize = 16

    # check for grouping of experiments for coloring and aggregate learning curves
    if plotter_inputs.group_breaks is not None:
        use_grouping = True
        n_groups = len(plotter_inputs.group_breaks) + 1
        group_data = [None]*n_groups
        assert all(plotter_inputs.group_breaks[i] <= plotter_inputs.group_breaks[i+1] for i in range(n_groups-2)), "Group Breaks must be sorted"

    # check that custom legend is properly formatted
    use_custom_legend = False
    if len(plotter_inputs.custom_legend) > 0:
        assert len(plotter_inputs.custom_legend) == n_experiments or len(plotter_inputs.custom_legend) == n_groups
        use_custom_legend = True


    # iterate through each experiment (or sequence of experiment extension)s, load data and plot
    group_ind = -1
    marker = itertools.cycle(('o', 'X', '^', 'P', 'v', '*', '+', '.', '1', '2'))
    for ext_exp_iter, extended_experiment in enumerate(plotter_inputs.experiments):
        fig = plt.figure("learning_curves")

        # separate experiment into sequence of restored extensions, if applicable
        experiment_extension_list = str.split(extended_experiment, '+')

        # break experiment name into algorithm and scenario:
        exp_seg_0_name_split = str.split(experiment_extension_list[0], '.')
        alg_scen = None
        training_algorithms.append(copy.copy(experiment_extension_list[0]))
        scenarios.append('unknown')
        if len(exp_seg_0_name_split) == 4:
            # old format
            alg_scen = str.split(exp_seg_0_name_split[2], '_', 1)
            training_algorithms[-1] = alg_scen[0]
            scenarios[-1] = alg_scen[1]
        if len(exp_seg_0_name_split) == 5:
            # new format
            training_algorithms[-1] = exp_seg_0_name_split[2]
            scenarios[-1] = exp_seg_0_name_split[3]

        exp_seg_0_training_algorithm = training_algorithms[-1]
        exp_seg_0_scenario = scenarios[-1]

        # check scenario is consistent for all experiments
        if not all([scen == scenarios[0] for scen in scenarios]):
            raise Exception('Scenario not common between all experiments: {}'.format(scenarios))

        # get group number and color for plotting
        if use_grouping:
            if ext_exp_iter == 0 or any(ext_exp_iter == gb for gb in plotter_inputs.group_breaks):
                # increment group index and prep group data 
                group_ind += 1
                group_data[group_ind] = []
            # use same color per group
            color_ind =  group_ind % n_colors
            color = color_list[color_ind]
        else:
            # use different color for each experiment
            color_ind += 1
            color_ind =  color_ind % n_colors
            color = color_list[color_ind]

        # iterate through sequence of experiment segments, plotting each at the appropriate start episode
        final_segment_timestep = 0
        extended_experiment_timesteps = []
        extended_experiment_trend = []
        for exp_seg_iter, experiment_segment in enumerate(experiment_extension_list):

            # use arglist to parse information, if available
            arglist_path = (glob.glob(os.path.join(experiment_segment, '*_arglist.pkl')) + glob.glob(os.path.join(experiment_segment, '*.arglist.pkl')))
            use_arglist = False
            if len(arglist_path) > 1:
                raise Exception('More than one arglist present, unclear which to use: {}'.format(arglist_path))
            if len(arglist_path) == 1:
                arglist_path = arglist_path[0]
                use_arglist = True
                arglist = pickle.load(open(arglist_path, "rb"))
        

            # extract parameters for arglist, notes, or assumed defaults
            save_rate = 1000
            max_episode_len = 50
            n_minibatches = 4
            n_opt_epochs = 4
            batch_size = 1024
            if use_arglist:
                print("Using arglist to interpret inputs for {}".format(experiment_segment))
                save_rate = arglist.save_rate
                max_episode_len = arglist.max_episode_len
                batch_size = arglist.batch_size
                if hasattr(arglist, "num_minibatches"):
                    n_minibatches = arglist.num_minibatches
                else:
                    print("------> assuming default n_minibatches of {}".format(n_minibatches))
                if hasattr(arglist, "num_opt_epochs"):
                    n_opt_epochs = arglist.num_opt_epochs
                else:
                    print("------> assuming default n_opt_epochs of {}".format(n_opt_epochs))

            else:
                print("Using notes.md and defaults to interpret inputs for {}".format(experiment_segment))
                notes_path = os.path.join(experiment_segment, 'notes.md')
                notes_file = open(notes_path, "r")
                notes_call = str.split(notes_file.readline(), " ")
                notes_file.close()
                assert notes_call[0] == "Call:" # check formatting is as expected
                if "--save-rate" in notes_call:
                    save_rate = int(notes_call[notes_call.index("--save-rate")+1])
                else:
                    print("------> assuming default save_rate of {}".format(save_rate))
                if "--max-episode-len" in notes_call:
                    max_episode_len = int(notes_call[notes_call.index("--max-episode-len")+1])
                else:
                    print("------> assuming default max_episode_len of {}".format(max_episode_len))
                if "--num-minibatches" in notes_call:
                    n_minibatches = int(notes_call[notes_call.index("--num-minibatches")+1])
                else:
                    print("------> assuming default n_minibatches of {}".format(n_minibatches))
                if "--num-opt-epochs" in notes_call:
                    n_opt_epochs = int(notes_call[notes_call.index("--num-opt-epochs")+1])
                else:
                    print("------> assuming default n_opt_epochs of {}".format(n_opt_epochs))
                if "--batch-size" in notes_call:
                    batch_size = int(notes_call[notes_call.index("--batch-size")+1])
                else:
                    print("------> assuming default batch_size of {}".format(batch_size))
                

            # Check that each segment of experiment aligns with original
            exp_seg_name_split = str.split(experiment_segment, '.')
            exp_seg_alg_scen = None
            exp_seg_training_algorithm = copy.copy(experiment_segment)
            exp_seg_scenario = 'unknown'
            if len(exp_seg_name_split) == 4:
                # old format
                exp_seg_alg_scen = str.split(exp_seg_0_name_split[2], '_', 1)
                exp_seg_training_algorithm = exp_seg_alg_scen[0]
                exp_seg_scenario = exp_seg_alg_scen[1]
            if len(exp_seg_name_split) == 5:
                # new format
                exp_seg_training_algorithm = exp_seg_name_split[2]
                exp_seg_scenario = exp_seg_name_split[3]
            if (not exp_seg_training_algorithm == exp_seg_0_training_algorithm or 
                not exp_seg_scenario == exp_seg_0_scenario):
                raise Exception("Experiment extension {} does not align with original experiment {}".format(exp_seg_name_split, exp_seg_0_name_split))

            # extract reward statistics
            reward_stats_path = (glob.glob(os.path.join(experiment_segment,'*_reward_stats.pkl')) + 
                                glob.glob(os.path.join(experiment_segment,'*_rewards_stats.pkl')) + 
                                glob.glob(os.path.join(experiment_segment,'*.rewards_stats.pkl')))
            for rp_ind, rp in enumerate(reward_stats_path):
                # remove per-agent reward files
                if 'agent_reward' in rp:
                    reward_stats_path.pop(rp_ind)
            if len(reward_stats_path) < 1:
                raise Exception('No such experiment found: {}'.format(experiment_segment))
            elif len(reward_stats_path) > 1:
                raise Exception('Multiple reward statistics files found: {}'.format(reward_stats_path))
            reward_stats_path = reward_stats_path[0]
            r_stats = pickle.load(open(reward_stats_path, "rb"))
            r_mean = np.array([r[0] for r in r_stats])
            r_std = np.array([r[1] for r in r_stats])

            # extract raw episode rewards
            episode_rewards_path = glob.glob(os.path.join(experiment_segment,'*.rewards.pkl'))
            if len(episode_rewards_path) < 1:
                raise Exception('No such experiment found: {}'.format(experiment_segment))
            elif len(episode_rewards_path) > 1:
                raise Exception('Multiple reward files found: {}'.format(reward_path))
            episode_rewards_path = episode_rewards_path[0]
            episode_rewards = pickle.load(open(episode_rewards_path, "rb"))
            n_episodes = len(r_stats)*save_rate
            assert n_episodes <= len(episode_rewards)

            # Get or compute timesteps and moving average
            if plotter_inputs.smoothing_window == -1:
                averaging_window = save_rate
            elif plotter_inputs.smoothing_window > 0:
                averaging_window = plotter_inputs.smoothing_window
            else:
                raise Exception("Invalid moving average window")
            r_mean_recalc = [np.mean(episode_rewards[r_ind:r_ind+averaging_window]) for r_ind in range(0, n_episodes, averaging_window)]
            r_mean = np.asarray(r_mean_recalc)
            learning_timesteps = np.arange(0, max_episode_len*n_episodes, max_episode_len*averaging_window) + final_segment_timestep + max_episode_len*averaging_window

            # select/calculate trend line
            if plotter_inputs.trend_plot.lower() in ["mean", "average", "avg"]:
                # use mean as estimate of moving average of episode rewards
                r_trend = r_mean
            elif plotter_inputs.trend_plot.lower() in ["median", "med"]:
                # use median as estimate of moving average of episode rewards
                r_trend = np.asarray([np.percentile(episode_rewards[r_ind:r_ind+save_rate], 50) for r_ind in range(0, n_episodes, averaging_window)])
            else:
                raise Exception("Unrecognized option trend_plot:{}".format(plotter_inputs.trend_plot))

            # plot deviation from trend line (standard deviation, percentile)
            if plotter_inputs.deviation_plot.lower() in ["std", "standard_deviation", "standard-deviation"]:
                # plot standard deviation
                r_std_recalc = [np.std(episode_rewards[r_ind:r_ind+averaging_window]) for r_ind in range(0, n_episodes, averaging_window)]
                r_std = r_std_recalc
                r_dev_lower = r_mean-r_std
                r_dev_upper = r_mean+r_std
                # plt.fill_between(learning_timesteps, r_mean-r_std, r_mean+r_std, facecolor=color, alpha=0.2)

            elif plotter_inputs.deviation_plot.lower() in ["percentile"]:
                # compute percentile
                r_dev_lower = [np.percentile(episode_rewards[r_ind:r_ind+averaging_window], 25) for r_ind in range(0, n_episodes, averaging_window)]
                r_dev_upper = [np.percentile(episode_rewards[r_ind:r_ind+averaging_window], 75) for r_ind in range(0, n_episodes, averaging_window)]

                # plot percentiles
                # plt.fill_between(learning_timesteps, r_per25, r_per75, facecolor=color, alpha=0.2)

            elif plotter_inputs.deviation_plot.lower() in ["none", "off", "no", "n"]:
                # do not plot deviation from mean
                pass

            else:
                raise Exception("Unrecognized option deviation_plot:{}".format(plotter_inputs.deviation_plot))

            # Tie extension to prior segment
            if exp_seg_iter > 0:
                learning_timesteps = np.insert(learning_timesteps, 0, final_segment_timestep)
                r_trend = np.insert(r_trend, 0, final_segment_trend)
                r_dev_lower = np.insert(r_dev_lower, 0, final_segment_dev_lower)
                r_dev_upper = np.insert(r_dev_upper, 0, final_segment_dev_upper)

            # plot trend line
            if plotter_inputs.plot_markers:
                plt.plot(learning_timesteps, r_trend, marker=next(marker), color=color,linewidth=0.5)
            else:
                plt.plot(learning_timesteps, r_trend, color=color)
            # plt.plot(learning_timesteps, r_trend, color)

            # fill between deviations
            if not plotter_inputs.deviation_plot.lower() in ["none", "off", "no", "n"]:
                plt.fill_between(learning_timesteps, r_dev_upper, r_dev_lower, facecolor=color, alpha=0.2)

            # store extended experiment information
            final_segment_timestep = learning_timesteps[-1]
            final_segment_trend = r_trend[-1]
            final_segment_dev_lower = r_dev_lower[-1]
            final_segment_dev_upper = r_dev_upper[-1]
            extended_experiment_timesteps.extend(learning_timesteps)
            extended_experiment_trend.extend(r_trend)

            if plotter_inputs.max_timestep > 0:
                plt.xlim(0, int(plotter_inputs.max_timestep))

            ######################
            ## Plot Loss Curves ##
            ######################
            # plot_losses = False
            losses_path = (glob.glob(os.path.join(experiment_segment, '*_losses.pkl')) + glob.glob(os.path.join(experiment_segment, '*.losses.pkl')))
            if len(losses_path) > 1:
                raise Exception('More than one losses file present, unclear which to use: {}'.format(arglist_path))
            if len(losses_path) == 1:
                losses_path = losses_path[0]
                # plot_losses = True
            else:
                losses_path = None

            if losses_path is not None and plotter_inputs.plot_losses:
                fig = plt.figure("loss_curves_{}".format(experiment_segment))
                loss_data = pickle.load(open(losses_path, "rb"))

                # extract losses
                loss_names = loss_data[0]
                loss_values = loss_data[1]
                policy_loss_index = loss_names.index("policy_loss")
                policy_loss = [v[policy_loss_index] for v in loss_values]
                value_loss_index = loss_names.index("value_loss")
                value_loss = [v[value_loss_index] for v in loss_values]
                policy_entropy_index = loss_names.index("policy_entropy")
                policy_entropy = [v[policy_entropy_index] for v in loss_values]
                policy_clipfrac_index = loss_names.index("clipfrac")
                policy_clipfrac = [v[policy_clipfrac_index] for v in loss_values]
                central_value_loss = None
                central_value_explained_variance = None
                if "central_value_loss" in loss_names:
                    central_value_loss_index = loss_names.index("central_value_loss")
                    cvl = [v[central_value_loss_index] for v in loss_values]
                    if len(cvl) == len(value_loss):
                        central_value_loss = cvl
                if "central_value_explained_variance" in loss_names:
                    central_value_explained_variance_index = loss_names.index("central_value_explained_variance")
                    central_value_explained_variance = [v[central_value_explained_variance_index] for v in loss_values]

                # generate batch and timestep numbers
                n_updates = len(loss_values)
                update_steps = np.arange(n_updates)
                n_batches = n_updates/(n_minibatches * n_opt_epochs)
                n_timesteps = n_batches * batch_size
                ax1 = fig.add_subplot(111)
                ax1.plot(update_steps, policy_loss)
                ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, t: '%.2g' % (x//(n_minibatches*n_opt_epochs) * batch_size * max_episode_len)))
                ax1.set_ylabel('metrics',color='b')
                ax1.set_xlabel('timesteps')
                ax1.plot(update_steps, policy_entropy)
                ax1.plot(update_steps, policy_clipfrac)
                if central_value_explained_variance is not None:
                    ax1.plot(update_steps, central_value_explained_variance)
                    ax1.legend(['policy_loss', 'policy_entropy', 'policy_clipfrac', 'central_value_explained_variance'])
                else:
                    ax1.legend(['policy_loss', 'policy_entropy', 'policy_clipfrac'])
                
                # restrict y-axis if scale is too large
                ax1_ylim = ax1.get_ylim()
                ax1_ylim = np.clip(ax1_ylim, -1e5, 1e5)
                ax1.set_ylim(ax1_ylim)

                ax2 = ax1.twinx()
                ax2.plot(update_steps, value_loss, 'y')
                ax2.set_ylabel('value_loss')
                if central_value_loss is not None:
                    ax2.plot(update_steps, central_value_loss, 'k')
                    ax2.legend(['value_loss', 'central_value_loss'])
                else:
                    ax2.legend(['value_loss'])

                ax3 = ax1.twiny()
                ax3.plot(update_steps, policy_loss) # just throwing this on so we can get a x-axis that marks training batch
                ax3.set_xlabel("training batch")
                ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, t: '%d' %(x // (n_minibatches*n_opt_epochs))))

        # record group data
        if use_grouping:
            group_data[group_ind].append((extended_experiment_timesteps, extended_experiment_trend))


    # label learning curves
    fig = plt.figure("learning_curves")
    if use_custom_legend and len(plotter_inputs.custom_legend) == n_experiments:
        plt.legend(plotter_inputs.custom_legend)
    elif plotter_inputs.short_legend:
        plt.legend(training_algorithms)
    else:
        date_alg_scen_num = []
        for ext_exp_iter, extended_experiment in enumerate(plotter_inputs.experiments):
            experiment_extension_list = str.split(extended_experiment, '+')
            for exp_seg_iter, experiment_segment in enumerate(experiment_extension_list):
                exp_seg_name_split_1 = str.split(experiment_segment, '.', 1)
                if len(exp_seg_name_split_1) == 1:
                    date_alg_scen_num.append(exp_seg_name_split_1[0])
                elif len(exp_seg_name_split_1) == 2:
                    date_alg_scen_num.append(exp_seg_name_split_1[1])
                else:
                    raise Exception("Experiment name mis-formatted")
        plt.legend(date_alg_scen_num)
    plt.xlabel('timesteps', fontsize=label_fontsize); plt.ylabel('total rewards per episode', fontsize=label_fontsize)
    if plotter_inputs.custom_title == '':
        plt.title('Learning Curves for Scenario: {}'.format(scenarios[0]), fontsize=title_fontsize)
    else:
        plt.title(plotter_inputs.custom_title, fontsize=title_fontsize)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()

    ####################################################
    # PLOT GROUP LEARNING CURVES #
    ####################################################
    if use_grouping:
        marker = itertools.cycle(('o', 'X', '^', 'P', 'v', '*', '+', '.', '1', '2'))
        fig = plt.figure("group_learning_curves")
        for group_ind, group in enumerate(group_data):

            # handle un-aligned timesteps
            group_timesteps = list(zip(*[exp_data[0] for exp_data in group]))
            group_timesteps_mean = [np.mean(times_t) for times_t in group_timesteps]

            # calculate average performance across experiments
            group_rewards = list(zip(*[exp_data[1] for exp_data in group]))
            group_rewards_mean = [np.mean(rewards_t) for rewards_t in group_rewards]
            group_rewards_min = [np.min(rewards_t) for rewards_t in group_rewards]
            group_rewards_max = [np.max(rewards_t) for rewards_t in group_rewards]

            # plot averaged rewards and min-max from group
            if plotter_inputs.plot_markers:
                plt.plot(group_timesteps_mean, group_rewards_mean, marker=next(marker), linewidth=0.5)
            else:
                plt.plot(group_timesteps_mean, group_rewards_mean)
            plt.fill_between(group_timesteps_mean, group_rewards_min, group_rewards_max, alpha=0.2)

        # create legend, title, and axis markers
        if use_custom_legend and len(plotter_inputs.custom_legend) == n_groups:
            plt.legend(plotter_inputs.custom_legend)
        else:
            plt.legend(["group {}".format(i) for i in range(n_groups)])

        plt.xlabel('timesteps',fontsize=label_fontsize); plt.ylabel('total rewards per episode',fontsize=label_fontsize)
        if plotter_inputs.custom_title == '':
            plt.title('Aggregated Learning Curves for Scenario: {}'.format(scenarios[0]), fontsize=title_fontsize)
        else:
            plt.title(plotter_inputs.custom_title, fontsize=title_fontsize)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.tight_layout()

    

if __name__ == '__main__':

    # parse plot inputs
    plotter_inputs = parse_args()

    # create learning and loss plots
    plot_results(plotter_inputs)
    # plot_loss_curves(plotter_inputs, use_custom_legend)

    # show plots
    plt.show()