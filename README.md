# ERGO: Exploiting Risk-taking in Group Operations

This is a library for developing and testing new multi-agent reinforcement learning algorithms; particulary for multi-agent systems operating in hazardous environments.

## Distribution Statement

___NOTE: strikethrough to be removed upon final release approval, until then it is not publicly released___

~~DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.~~

~~This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.~~

~~© 2019 Massachusetts Institute of Technology.~~

~~The software/firmware is provided to you on an As-Is basis~~

~~Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.~~

## Papers & Publications

[Health-Informed Policy Gradients for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1908.01022)

<pre>
@article{allen2019health,
  title={Health-Informed Policy Gradients for Multi-Agent Reinforcement Learning},
  author={Allen, Ross E and Bear, Javona White and Gupta, Jayesh K and Kochenderfer, Mykel J},
  journal={arXiv preprint arXiv:1908.01022},
  year={2019}
}
</pre>

## Related Projects

This project leverages: 
+ [OpenAI Gym](https://gym.openai.com/docs/) as a unified RL training infrastructure
+ [TensorFlow](https://www.tensorflow.org/) to define neural networks used for policy and value estimation
+ [Multi-Agent Particle Environment](https://github.com/openai/multiagent-particle-envs) to define scenarios/problems for RL training of simple, particle-like robots in low-dimension environments
+ [OpenAI Baselines](https://github.com/openai/baselines) for comparison to other RL algorithms.


## Setup

This project uses a conda environment to manage dependencies. [Use these instructions](https://conda.io/docs/user-guide/install/linux.html) to install miniconda.

**Note:** Do not confuse the concept of a conda environment with that of a OpenAI Gym environment; they are complete distinct. Conda environments are a way of managing dependencies for different software, gym environments are a "simulator" used to test reinforcement learning algorithms

The instructions for conda to create the environment are found in `environment.yaml`. To create the conda environment:

```
conda env create -f environment.yaml
```

Then switch to the environment (this must be done in each individual terminal that you wish to run the environment)
```
conda activate ergo_particle_gym
```

## Testing

To ensure that the repository, submodules, and dependencies have been properly installed, you can run the following smoke tests and unit/integration tests (_these are also very useful during normal development cycles to ensure new changes haven't broken old functionality_):

__Smoke Tests:__ used to check that various combinations of algorithms, environments, scenarios, and input options all startup without throwing errors (i.e. _plug it in and see if it smokes_). Note: this uses 4 processors to run tests in parallel, this can be scaled up or down based on your computer

    ```
    cd testing
    python smoke_tests.py --num-cores 4
    ```

__Unit/Integration Tests:__ used to test individual functions and groups of functions. Note: this uses 4 processors to run tests in parallel, this can be scaled up or down based on your computer

    ```
    cd testing
    nosetests --nologcapture --nocapture --verbose --exe --processes=4 --process-timeout=180
    ```

___TODO:___ Move to an [automated testing](https://travis-ci.org/) infrastructure

## Environments / Scenarios

The learning environments and scenarios are found in [particle_environments](./particle_environments) subdirectory. This repository contains a number of scenarios forked directly from [Multi-Agent Particle Environments](https://github.com/openai/multiagent-particle-envs) as well as a number of custom-made environments that incorporate the ideas of hazards, risk-taking, agent health, and partial observability which can be found in the [mager](./particle_environments/mager) subdirectory. 

Note that some scenarios are obsolete while others are yet-to-be implemented, the most actively used and developed environments are:

+ [ergo_spread_variable](./particle_environments/mager/scenarios/ergo_spread_variable.py)
+ [ergo_graph_variable](./particle_environments/mager/scenarios/ergo_graph_variable.py)
+ [ergo_perimeter2_variable](./particle_environments/mager/scenarios/ergo_perimeter2_variable.py)



## Algorithms

The multi-agent reinforcement learning algorithms used for training within the various environments are found in the [rl_algorithms](./rl_algorithms) subdirectory. The primary focus of this work is the development of a novel, multi-agent variant of the proximal policy optimization algorithm ([MAPPO](./rl_algorithms/mappo.py)) that can leverage health information to improve multi-agent learning. Other algorithms for comparison and testing come from a fork of [MADDPG](https://github.com/openai/maddpg), a fork of OpenAI's [baselines](https://github.com/openai/baselines), as well as a novel [scenario-specific cross-entropy method](./rl_algorithms/mclearning.py).


## Training Examples

The script to kick off training [`train.py`](./train.py). The various input arguments and options are listed `train.py:parse_args()`. Instead of giving details on all the input paramenters, here are a few examples of training experiments 

+ ___Example:___ Run a bare-bones, very small example case with OpenAI's version of MADDPG algorithm

    ```
    python train.py --environment MultiAgentEnv --scenario simple --training-algorithm MADDPGAgentTrainer --num-episodes 10 --save-rate 5
    ```

+ ___Example:___ Run a large simulation of the `simple_graph_small` scenario and homemade version of multi agent PPO (may take a few hours to complete). Note that this assumes that you have cloned ergo_particle_gym to your home directory. The `--record-experiment` flag enforces that all code changes have been committed and then records the exact input command and commit hash so that the experiment can be reproduced exactly in the future. 
    ```
    python train.py --record-experiment --environment MultiAgentRiskEnv --scenario simple_graph_small --training-algorithm PPOGroupTrainer --save-dir ~/ergo_particle_gym/experiments/ --max-episode-len 50 --num-episodes 500000 --save-rate 100 --batch-size 256 --entropy-coef 0.001 --force-private-value --learning-rate 5e-3 --variable-learning-rate --activation tanh
    ```
    + To view the resulting behavior after training, re-run the same command replacing `--record-experiment` with `--display`, and `--save-dir ...` with `--load-dir ...` pointing to your saved experiment


+ ___Example:___ Run one of simulation of the `ergo_graph_variable` (i.e. hazardous communication network) environment with multi-agent proximal policy optimization (MAPPO) and minimum-health crediting
    ```
     python train.py --record-experiment --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer --save-dir ~/ergo_particle_gym/experiments/ --max-episode-len 50 --num-episodes 50000 --save-rate 100 --batch-size 256 --num-minibatches 8 --num-opt-epochs 8 --entropy-coef 0.01 --learning-rate 1e-3 --activation tanh --variable-num-agents 10 --variable-num-hazards 1 --variable-observation-type histogram --critic-type central_joint_state --central-critic-learning-rate 5e-3 --central-critic-num-units 64 --central-critic-num-layers 8 --central-critic-activation elu --crediting-algorithm terminated_baseline
    ```

## Output and Post Processing

The results of a training run are stored in a new directory with name format `expdata.YYYY-MM-DD.<algorithm_name>.<scenario_name>.<count>/`. The directory contains information episode rewards over time (`rewards.pkl` and `reward_stats.pkl`), loss values to debug training process (`.losses.pkl`), and the trained neural network policy.

The results can be processed and visualized with a [custom plotting script](./experiments/plotter.py), `plotter.py`. Here are a few example use cases of the plotter (_NOTE:_ This assumes you have experiment data stored in your `experiments` directory) (_NOTE:_ These can be run while training is occurring to monitor progress)

+ ___Example:___ Plot learning 3 learning curves (assuming you have the necessary experiments completed

    ```
    cd experiments
    python plotter.py --custom-legend MADDPG MAPPO X-Entropy --experiments <maddpg_experiment_directory_name> <mappo_experiment_directory_name> <xentropu_experiment_directory_name>
    ```

+ ___Example:___ For a more expansive campaign of 16 different experiments, collected and averaged into groups of 4 based on the algorithm used for each experiment, use this plotting call
    ```
    cd experiments
    python plotter.py --experiments <min_health_exp_dir_0> <min_health_exp_dir_1> <min_health_exp_dir_2> <min_health_exp_dir_3> <central_critic_exp_dir_0> <central_critic_exp_dir_1> <central_critic_exp_dir_2> <central_critic_exp_dir_3> <local_critic_exp_dir_0> <local_critic_exp_dir_1> <local_critic_exp_dir_2> <local_critic_exp_dir_3> <maddpg_exp_dir_0> <maddpg_exp_dir_1> <maddpg_exp_dir_2> <maddpg_exp_dir_3> <maddpg_exp_dir_4> --group-breaks 4 8 12 --custom-legend "MAPPO: min-health crediting" "MAPPO: central critic" "MAPPO: local critic" "MADDPG" --custom-title "Learning Curves for 4-Agent Navigation Scenario" --smoothing-window 2048 --plot-markers True
    ```

+ ___Example:___ You can also monitor the trends in loss values (policy loss, value loss) to help debug how you learning algorithm is performing or where it can be modified
    ```
    cd experiments
    python plotter.py --experiments <experiment_directory> --plot-losses
    ```


