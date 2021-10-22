# The Scaling Limits of Unsupervised Skill Discovery
## write an introduction here

# Getting Started
### Clone the repository

```bash
git clone https://github.com/FaisalAhmed0/SLUSD
```

### you can setup a new environment and install requirements.txt

```bash
conda create -n slusd
pip3 install -r requirements.txt 
```

### activate the new environment and run train.py

```bash
conda activate vae_env
```
### Run an expirment with ppo
```bash
python diyan_ppo.py --env "MountainCarContinuous-v0"
```

### Run an expirment with sac
```bash
python diyan_sac.py --env "MountainCarContinuous-v0"
```

### fine tune for a specfic environment task
```bash
python finetune.py --env "MountainCarContinuous-v0" --alg ppo
```