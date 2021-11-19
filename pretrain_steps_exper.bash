#! /bin/bash

xvfb-run -a python src/experiments/pretrain_steps_exper.py --env "MountainCarContinuous-v0"
xvfb-run -a python src/experiments/pretrain_steps_exper.py --env "InvertedPendulum-v2"
xvfb-run -a python src/experiments/pretrain_steps_exper.py --env "HalfCheetah-v2"
xvfb-run -a python src/experiments/pretrain_steps_exper.py --env "Hopper-v2"
xvfb-run -a python src/experiments/pretrain_steps_exper.py --env "Ant-v2"
