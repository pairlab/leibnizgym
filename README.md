# Leibniz Gym


This repository provides IsaacGym environment for the tri-finger robot. However, 
the development work can be used to base other RL environments in IsaacGym as well.

The project currently uses [RL-Games](https://github.com/Denys88/rl_games) for training agents. 
More support will follow based on demand.

# Installation

Details regarding installtion of IsaacGym can be found [here](isaacgym/docs/install.html). 
For convinience these are summarized below as well. We currently support the `Preview Release 2` version of IsaacGym.

### Prerequisites

The code has been tested on Ubuntu 20.04 with Python 3.7. The minimum recommended NVIDIA driver
version for Linux is `460.32` (dictated by support of IsaacGym).

It uses [Anaconda](https://www.anaconda.com/) to create virtual environments. 
To install Anaconda, follow instructions [here](https://docs.anaconda.com/anaconda/install/linux/).

### Install IsaacGym in a new conda environment 

Run the IsaacGym installation script (`bash create_conda_env_rlgpu.sh`, provided in the downloaded IsaacGym zip):

This will create a new conda env called `rlgpu`, which can be activated by running:
```bash
conda activate rlgpu
```

### Install Leibnizgym

In the root directory, run:
```bash
pip install -e .
```

This will install the `leibnizgym` package and all its dependencies into the Python environment.

### Install RL toolbox (optional)

Currently, we use RL-games for reinforcement learning. **It should be downloaded automatically with the previous step**. However, if you want to try with a different branch, you can manually install it locally instead:
```bash
# somewhere outside trifinger-manip
git clone https://github.com/Denys88/rl_games.git
pip install -e .
```

# Running


## Training


To train with RL Games:
```bash
python scripts/train_rlgames.py num_envs=<INT>
```

Where num_envs is an integer. For a recent 8G NVIDIA GPU you should go with `8192` envs, or for 16G `16382`.

We use Hydra to keep configuration of runs simple. You can view the main arguments to the scripts by looking in `resources/config/config.yaml` which contains more information. You set them by doing `config_variable=value`. The main ones to be aware of are:

* `leibniz` sets which environment to use. Currently the only one and the default is `trifinger`, other environments will be listed in the `resources/config/leibniz` directory as they are created.
* `lebiniz/reward_terms` sets which set of reward terms to use (environment specific). Currently we have two groups for Trifinger: `keypoints` and `posquat`.
* `leibniz.task_difficulty` (`trifinger`-only) set to 1, 2, 3, or 4, which correspond to training on each of the four difficulty levels from the Real Robot Challenge - see [description here](https://people.tuebingen.mpg.de/felixwidmaier/realrobotchallenge/simulation_phase/tasks.html#difficulty-levels).
* `leibniz.cube_obs_keypoints` - boolean flag, defaults to `True`. When set, pose is represented by keypoints in network inputs, otherwise it is represented by concatenated position and quaternion.
* `wandb` - see `config.yaml` for details, but set `wandb.activate=True` to enable WandB logging. 

(Arguments using the `/` syntax are selecting one option from a config group, whereas those wtih a `.` are setting the value of an individual argument.)

## Inference & Loading Checkpoints

Trained checkpoints from RL Games are saved within the `nn/` directory within the relevant output directory (`output/<date>/time` by default)

To run inference and see the results, you want to go:


```bash
python scripts/train_rlgames.py num_envs=<INT> play=True checkpoint=/path/to/checkpoint.pth
```

Where `play` tells it not to train and `checkpoint` is the path to load the checkpoint. If you want to start training from a trained checkpoint, then just don't specify `play` (or set it to `False`).
Usually you'll only want a few environments (say, 256) to get smooth rendering performance during inference.


The following is one result of training on a V-100GPU for ~24 hours to about 4B timesteps across 16384 environments:

![trifinger_rlg_4](images/training_curve)

## Citing

```
***TODO*** BibTeX when arXived.
```
