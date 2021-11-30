
#xvfb-run -a python record.py --bestskill 5  --stamp "1637662274.157622" --env "Swimmer-v2" --alg ppo --skills 6
xvfb-run -a python record.py --bestskill 5  --stamp "1637526072.801331" --env "MountainCarContinuous-v0" --alg ppo --skills 6
#xvfb-run -a python record.py --bestskill 5  --stamp "1637837247.0089962" --env "HalfCheetah-v2" --alg ppo --skills 6


xvfb-run -a python record.py --bestskill 0  --stamp "1637526150.647815" --env "MountainCarContinuous-v0" --alg sac --skills 6
xvfb-run -a python record.py --bestskill 5  --stamp "1637569210.1555948" --env "Swimmer-v2" --alg sac --skills 6
xvfb-run -a python record.py --bestskill 1  --stamp "1637611096.0250297" --env "HalfCheetah-v2" --alg sac --skills 6
#xvfb-run -a python record.py --bestskill -1  --stamp "1637838633.2888947" --env "Hopper-v2" --alg sac --skills 6
