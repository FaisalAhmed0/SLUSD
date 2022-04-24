# An Empirical Investigation of Mutual Information Skill Learning.
## Abstract
Unsupervised skill learning methods are a form of unsupervised pre-training for reinforcement learning (RL) that has the potential to improve the sample efficiency of solving downstream tasks. Prior work has proposed several methods for unsupervised skill discovery based on mutual information (MI) objectives, with different methods varying in how this mutual information is estimated and optimized. This paper studies how different design decisions in skill learning algorithms affect the sample efficiency of solving downstream tasks. Our key findings are that the sample efficiency of downstream adaptation under off-policy backbones is better than their on-policy counterparts. In contrast, on-policy backbones result in better state coverage, moreover, regularizing the discriminator gives better downstream results, and careful choice of the mutual information lower bound and the discriminator architecture yields significant improvements in downstream returns, also, we show empirically that the learned representations during the pre-training step correspond to the controllable aspects of the environment.

## Getting Started
### Clone the repository

```bash
git clone https://github.com/FaisalAhmed0/SLUSD
```

### Create a new conda environment and activate it

```bash
cd SLUSD
conda create -n slusd python=3.8
conda activate slusd
```

### Install a custom version of stablebaselines 3
```bash
git clone https://github.com/FaisalAhmed0/stable-baselines3.git
cd stable-baselines3
pip install -e .
```
### Install MBRL 
```bash
git clone https://github.com/facebookresearch/mbrl-lib.git
cd ../mbrl-lib
pip install -e .
```
### Make sure that mujoco is installed by following the instruction in https://github.com/openai/mujoco-py


### Install the paper specific code
```bash
cd ../SLUSD
pip install -r requirements.txt
pip3 install -e .
```

## To run the same adaptation expeirment of the paper.
```bash
python src/finetune.py --run_all True
```
When running with run_all is True, all random seeds will run on parallel.


## To run for a specific environment the cmd args are as follows
|Arg|Description|Supported values|Default Value
|--|-----|------|----|
|env|Environment name|All OpenAI gym environment with continuous actions and state vectros|"MountainCarContinuous-v0"|
alg|Deep RL algorithm|"sac" for soft-Actor critic, "ppo" for Proximal Policy Optimization|"ppo"|
skills| Number of skills to  learn|  Positive integers|6|
presteps| Number of pretraining steps | Positive integers|1000000
lb | Mutual Information lower bound|"ba" for $I_{BA}$, "nce" for $I_{NCE}$, "nwj" for $I_{NWJ} and "interpolate" for $I_{\alpha}$. | ba
pm | Discriminator parameterization |  "MLP" for a feed forward neural network, "Seprabale" for the seperable architecture, and "Concat" for the concatenation architecture and "linear" for the linear parametrization| MLP

## To run the same scalability expeirment of the paper.
```bash
python src/experiments/scalability_exper.py --run_all True
```
## For the regulrization experiment
```bash
python src/experiments/regularization_exper.py --run_all True
```

## To run the same scalability expeirment of the paper.
```bash
python src/experiments/scalability_exper.py --run_all True
```
### To observe the training dynamics run tensorboard inside the SLUSD folder
```bash
tensorboard --logdir ./logs_finetune
```

### To record a video for all skills
```bash
python record.py --env <env_name> --stamp <timestamp> --skills <no. skills> --cls <pm> --lb <mi lower bound>
```
Where stamp is the timestamp for the experiment you can copy it from the experiment's folder name.
If you are running this code on a server make sure xvfb is installed.
```bash
sudo apt-get install xvfb
```
And run your the recording script.
```bash
xvfb-run -a python record.py --env <env_name> --stamp <timestamp> --skills <no. skills> --cls <pm> --lb <mi lower bound>
```
