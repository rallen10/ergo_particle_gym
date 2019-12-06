""" Smoke tests for ergo_particle_gym library

Notes:
- Using commandline calls with arglists to exactly replicate how training is run
- Because of using command line calls, we need to actually cd into the proper directory,
not just import the train.py module
"""
import os
import subprocess
import sys
import argparse

from joblib import Parallel, delayed
import multiprocessing

def execute_single_smoketest(test_index, input_string):

    FNULL = open(os.devnull, 'w')

    # check filter
    if test_argslist.filter in input_string:
        proc = subprocess.Popen([input_string], stdout=FNULL, stderr=subprocess.PIPE, shell=True)
        out, err = proc.communicate()
        if err:
            print("\n-------------------------------------------------------------------------------")
            print("\nFound a problem!\nERROR IN SMOKE TEST #{}\n".format(test_index+1, input_string))
            print("Command Input:\n{}\n".format(input_string))
            print("Error Ouptut:\n{}".format(err.decode()))
            print("-------------------------------------------------------------------------------\n")
    else:
        print("\nSKIPPING TEST #{} DUE TO FILTER: {}\n".format(test_index+1, test_argslist.filter))
        # continue

    print("\n\nPASSED SMOKE TEST #{}".format(test_index+1))

if __name__ == '__main__':

    # handle path to results directories that ensure they don't clash with real runs
    if not os.path.exists('/tmp/test_policy'): os.makedirs('/tmp/test_policy')
    if not os.path.exists('/tmp/test_learning_curves'): os.makedirs('/tmp/test_learning_curves')
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/../' 

    # define tests to run
    smoketests = []
    smoketests.append("python train.py --environment MultiAgentEnv --scenario simple --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentEnv --scenario simple --training-algorithm PPOGroupTrainer --num-episodes 3 --batch-size 2 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentEnv --scenario simple_spread --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves'")
    smoketests.append("python train.py --environment MultiAgentEnv --scenario simple_spread --training-algorithm PPOGroupTrainer --num-episodes 3 --batch-size 2 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentEnv --scenario simple_spread --training-algorithm PPOGroupTrainer --num-episodes 3 --batch-size 2 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_testing1 --training-algorithm NonLearningAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_basic --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_basic --training-algorithm PPOGroupTrainer --num-episodes 3 --batch-size 2 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_variable --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 3  --variable-observation-type direct --variable-num-hazards 0 --critic-type distributed_local_observations --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 3  --variable-observation-type direct --variable-num-hazards 1 --critic-type distributed_local_observations --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 3  --variable-observation-type direct --variable-num-hazards 0 --critic-type central_joint_observations    --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 3  --variable-observation-type direct --variable-num-hazards 1 --critic-type central_joint_observations    --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 3  --variable-observation-type histogram --variable-num-hazards 0 --critic-type distributed_local_observations --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 3  --variable-observation-type histogram --variable-num-hazards 1 --critic-type distributed_local_observations --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 3  --variable-observation-type histogram --variable-num-hazards 0 --critic-type central_joint_observations    --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 3  --variable-observation-type histogram --variable-num-hazards 1 --critic-type central_joint_observations    --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_small    --training-algorithm MADDPGAgentTrainer            --num-episodes 3                --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_small    --training-algorithm ScenarioHeuristicAgentTrainer --num-episodes 3                --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_small    --training-algorithm ScenarioHeuristicGroupTrainer --num-episodes 3                --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_spread_small    --training-algorithm PPOGroupTrainer               --num-episodes 3 --batch-size 2 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 20 --variable-observation-type histogram --variable-num-hazards 0 --critic-type central_joint_observations                                  --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 20 --variable-observation-type histogram --variable-num-hazards 0 --critic-type central_joint_state                                         --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 20 --variable-observation-type histogram --variable-num-hazards 0 --critic-type central_joint_state --crediting-algorithm 'terminated_baseline' --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 20 --variable-observation-type histogram --variable-num-hazards 0 --critic-type distributed_local_observations  --variable-local-rewards    --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 20 --variable-observation-type histogram --variable-num-hazards 1 --critic-type central_joint_observations                                  --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 20 --variable-observation-type histogram --variable-num-hazards 1 --critic-type central_joint_state                                         --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 20 --variable-observation-type histogram --variable-num-hazards 1 --critic-type central_joint_state --crediting-algorithm 'terminated_baseline' --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 20 --variable-observation-type histogram --variable-num-hazards 1 --critic-type distributed_local_observations  --variable-local-rewards    --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type central_joint_observations                                  --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type central_joint_state                                         --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type distributed_local_observations  --variable-local-rewards    --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type central_joint_observations                                  --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type central_joint_state                                         --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type distributed_local_observations  --variable-local-rewards    --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type central_joint_observations                                  --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type central_joint_state                                         --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type distributed_local_observations                              --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type central_joint_state --crediting-algorithm 'terminated_baseline' --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type central_joint_observations                                  --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type central_joint_state                                         --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type distributed_local_observations                              --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type central_joint_state --crediting-algorithm 'terminated_baseline' --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type histogram --variable-num-hazards 0 --critic-type central_joint_observations                                  --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type histogram --variable-num-hazards 0 --critic-type central_joint_state                                         --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type histogram --variable-num-hazards 0 --critic-type distributed_local_observations                              --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type histogram --variable-num-hazards 0 --critic-type central_joint_state --crediting-algorithm 'terminated_baseline' --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type histogram --variable-num-hazards 1 --critic-type central_joint_observations                                  --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type histogram --variable-num-hazards 1 --critic-type central_joint_state                                         --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type histogram --variable-num-hazards 1 --critic-type distributed_local_observations                              --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter2_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type histogram --variable-num-hazards 1 --critic-type central_joint_state --crediting-algorithm 'terminated_baseline' --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    
    # define archived tests
    archived_smoketests = []
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_large --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_large --training-algorithm ScenarioHeuristicAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_large --training-algorithm ScenarioHeuristicGroupTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_large --training-algorithm PPOGroupTrainer --num-episodes 3 --batch-size 2 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario simple_graph_small --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario simple_graph_small --training-algorithm ScenarioHeuristicAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario simple_graph_small --training-algorithm ScenarioHeuristicGroupTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario simple_graph_small --training-algorithm PPOGroupTrainer --num-episodes 3 --batch-size 2 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario simple_graph_large --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario simple_graph_large --training-algorithm ScenarioHeuristicAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario simple_graph_large --training-algorithm ScenarioHeuristicGroupTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario simple_graph_large --training-algorithm PPOGroupTrainer --num-episodes 3 --batch-size 2 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_circuit --training-algorithm NonLearningAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_circuit --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_circuit_simplified --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_small --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_small --training-algorithm ScenarioHeuristicAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_small --training-algorithm ScenarioHeuristicGroupTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_graph_small --training-algorithm PPOGroupTrainer --num-episodes 3 --batch-size 2 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_hazards --training-algorithm NonLearningAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_hazards --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_small --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_small --training-algorithm PPOGroupTrainer --num-episodes 3 --batch-size 2 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_small --training-algorithm ScenarioHeuristicAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_small --training-algorithm ScenarioHeuristicGroupTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_variable --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type central_joint_observations                                  --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type central_joint_state                                         --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type distributed_local_observations                              --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type central_joint_state --crediting-algorithm 'terminated_baseline' --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    # archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 0 --critic-type distributed_local_observations  --variable-local-rewards    --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type central_joint_observations                                  --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type central_joint_state                                         --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type distributed_local_observations                              --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type central_joint_state --crediting-algorithm 'terminated_baseline' --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    # archived_smoketests.append("python train.py --environment MultiAgentRiskEnv --scenario ergo_perimeter_variable --training-algorithm PPOGroupTrainer    --num-episodes 3 --batch-size 2 --variable-num-agents 4  --variable-observation-type direct    --variable-num-hazards 1 --critic-type distributed_local_observations  --variable-local-rewards    --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'")
    
    # setup input argument parsing
    parser = argparse.ArgumentParser("Smoke Tests for ERGO Particle Gym")
    parser.add_argument("--run-archived", action="store_true", default=False)
    parser.add_argument("--run-serial", action="store_true", default=False)
    parser.add_argument("--filter", type=str, default="", help="exclusive run tests that match filter string")
    parser.add_argument("--num-cores", type=int, default=8)
    test_argslist = parser.parse_args()

    # print python version
    os.system("python --version")
    os.chdir(path)

    # add archived smoke tests if desired
    if test_argslist.run_archived:
        smoketests += archived_smoketests

    # run tests in parallel or serial
    if not test_argslist.run_serial:
        Parallel(n_jobs=test_argslist.num_cores)(delayed(execute_single_smoketest)(i,st) for i,st in enumerate(smoketests))

    else:

        for i, st in enumerate(smoketests):
            print("\n-----------------------")
            print("\nRUNNING SMOKE TEST {}/{}:\n{}\n".format(i+1, len(smoketests), st))

            # check filter
            if test_argslist.filter in st:
                try:
                    subprocess.check_call([st], shell=True)
                except subprocess.CalledProcessError:
                    print("\nFound a problem!\nError in SMOKE TEST #{}: {}".format(i+1, st))
                    print("\nTerminating smoke tests...")
                    sys.exit()
            else:
                print("\nSKIPPING TEST #{} DUE TO FILTER: {}\n".format(i+1, test_argslist.filter))
                continue

            print("\n\nPASSED SMOKE TEST #{}".format(i+1))