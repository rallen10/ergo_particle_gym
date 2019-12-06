#!/bin/sh
module load python
python train.py --environment MultiAgentEnv --scenario simple --training-algorithm MADDPGAgentTrainer --num-episodes 3 --save-dir '/tmp/test_policy/' --plots-dir '/tmp/test_learning_curves/'