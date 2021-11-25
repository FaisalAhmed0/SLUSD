#! /bin/bash

#xvfb-run -a python src/finetune.py --env "MountainCarContinuous-v0" --alg ppo --presteps 20000000 --skills 4
#xvfb-run -a python src/finetune.py --env "Swimmer-v2" --alg ppo --presteps 20000000 --skills 4
#xvfb-run -a python src/finetune.py --env "HalfCheetah-v2" --alg ppo --presteps 50000000 --skills 4
#xvfb-run -a python src/finetune.py --env "Hopper-v2" --alg ppo --presteps 70000000 --skills 4
#xvfb-run -a python src/finetune.py --env "Ant-v2" --alg ppo --presteps 100000000 --skills 4


xvfb-run -a python src/finetune.py --env "MountainCarContinuous-v0" --alg sac --presteps 500000 --skills 4
xvfb-run -a python src/finetune.py --env "Swimmer-v2" --alg sac --presteps 500000 --skills 4
xvfb-run -a python src/finetune.py --env "HalfCheetah-v2" --alg sac --presteps 2000000 --skills 4
xvfb-run -a python src/finetune.py --env "Hopper-v2" --alg sac --presteps 4000000 --skills 4
xvfb-run -a python src/finetune.py --env "Ant-v2" --alg ppo --presteps 5000000 --skills 4
