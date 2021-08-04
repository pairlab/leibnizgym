# Leibniz Gym

![Python](https://img.shields.io/badge/python-3.7-blue.svg)
![Ubuntu 18.04](https://img.shields.io/badge/ubuntu-18.04-orange.svg)
![Ubuntu 20.04](https://img.shields.io/badge/ubuntu-20.04-orange.svg)

This repository provides IsaacGym environment for the [TriFinger robot](https://sites.google.com/view/trifinger/home-page). However,
the development work can be used to base other RL environments in IsaacGym as well.

The project currently uses [RL-Games](https://github.com/Denys88/rl_games) for training agents.
More support will follow based on demand.

For list of contributors, check: [CONTRIBUTORS](CONTRIBUTORS). This code is released under [LICENSE](LICENSE).

# Installation

Details regarding installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym). We currently support the `Preview Release 2` version of IsaacGym.

### Pre-requisites

The code has been tested on Ubuntu 20.04 with Python 3.7. The minimum recommended NVIDIA driver
version for Linux is `460.32` (dictated by support of IsaacGym).

It uses [Anaconda](https://www.anaconda.com/) to create virtual environments.
To install Anaconda, follow instructions [here](https://docs.anaconda.com/anaconda/install/linux/).

### Install IsaacGym in a new conda environment

Run the IsaacGym installation script provided in the downloaded IsaacGym zip):

```bash
cd PATH/TO/isaacgym
bash create_conda_env_rlgpu.sh
```

This will create a new conda env called `rlgpu`, which can be activated by running:

```bash
conda activate rlgpu
```

### Install Leibnizgym

To install `leibnizgym` package and all its dependencies, run:
```bash
cd PATH/TO/leibnizgym
pip install -e .
```

### Install RL toolbox (optional)

Currently, we use [RL-Games](https://github.com/Denys88/rl_games) for reinforcement learning.

**It should be downloaded automatically with the previous step.**

However, if you want to try with a different branch, you can manually install it locally instead:

```bash
# somewhere outside trifinger-manip
git clone https://github.com/Denys88/rl_games.git
pip install -e .
```

# Running

## Training

We use [Hydra](https://hydra.cc/docs/intro/) to keep configuration of runs simple. You can view the main arguments to the scripts by looking in the file [`resources/config/config.yaml`](resources/config/config.yaml).

You can also set the configuration parameters from terminal by doing `{config_variable_name}={value}`. The main ones to be aware of are:

* **`gym`** (string): environment name to use. The current options are the different difficulty levels of trifinger: `trifinger_difficulty_{1,2,3,4}`
* **`num_envs`** (int): number of environment instances to run.
* **`headless`** (bool): whether to run the simulator with/without GUI.

To train an agent on environment difficulty 2, 8192 environment instances and headless mode, run:

```bash
python scripts/rlg_hydra.py gym=trifinger_difficulty_2 args.num_envs=8192 args.headless=True
```


## Inference and Loading Checkpoints

Trained checkpoints from RL-Games are saved within the `nn/` directory within the relevant output directory (`output/<date>/time` by default)

To perform inference and see the results, you can run:

```bash
python scripts/rlg_hydra.py args.num_envs=<INT> args.play=True args.checkpoint=/path/to/checkpoint.pth args.headless=False
```

where:
- `play` tells it not to train
- `checkpoint` is the path to load the checkpoint.

If you want to start training from a trained checkpoint, then just don't specify `play` (or set it to `False`). Usually you'll only want a few environments (say, 256) to get smooth rendering performance during inference.


## Citing

```
***TODO*** BibTeX when arXived.
```
