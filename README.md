# The Scaling Limits of Unsupervised Skill Discovery
An emprical study of mutual information based skill discovery.

# Getting Started
### Clone the repository

```bash
git clone https://github.com/FaisalAhmed0/SLUSD
```

### Create a new environment, install requirements.txt, run the setup, and activate the environment

```bash
cd SLUSD
conda create -n slusd python=3.8
conda activate slusd
pip3 install -r requirements.txt 
pip3 install -e .
```

### An Example for running a finetuning expeirment with a deep RL algorithm
```bash
python src/diyan_ppo.py --env "MountainCarContinuous-v0"  --alg "sac" --skills 6 --presteps 500000
```
The cmd arguemnts are
|Arg|Description|Supported values|Default Value
|--|-----|------|----|
|env|Environment name|All OpenAI gym environment with continuous actions and state vectros|"MountainCarContinuous-v0"|
alg|Deep RL algorithm|"sac" for soft-Actor critic, "ppo" for Proximal Policy Optimization|"ppo"|
skills| Number of skills to  learn|  Any positive integer|6|
presteps| Number of pretraining timesteps | Any positive integer|1000000

### To observe the training dynamics run tensorboard inside the SLUSD folder
```bash
tensorboard --logdir ./logs_finetune
```

### To record a video for all skills
```bash
python record.py --stamp "<timestamp>" --alg ppo --skills 6
```
where stamp is the timestamp for the experiment

### To calculate the state coverage (Entropy) 
```bash
python src/state_coverage.py --stamp "<timestamp>" --alg ppo --skills 6
```
### To plot the results of a finetuning experiment
```bash
python plot_results.py --stamp "<timestamp>" --alg ppo --skills 6
```