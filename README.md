# Leibniz Gym

This repository provides IsaacGym environment for the tri-finger robot. However,
the development work can be used to base other RL environments in IsaacGym as well.

The project currently uses [RL-Games](https://github.com/Denys88/rl_games) for training agents.
More support will follow based on demand.

For list of contributors, check: [CONTRIBUTORS](CONTRIBUTORS). This code is released under [LICENSE](LICENSE).

# Installation

Details regarding installation of IsaacGym can be found [here](isaacgym/docs/install.html).
For convenience these are summarized below as well. We currently support the `Preview Release 2` version of IsaacGym.

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
python scripts/rlg_hydra.py args.num_envs=<INT> args.headless=True
```

Where args.num_envs is an integer. For a recent 8G NVIDIA GPU you should go with `8192` envs, or for 16G `16382`.
`args.headless` tells it to stop rendering.

We use Hydra to keep configuration of runs simple. You can view the main arguments to the scripts by looking in `resources/config/config.yaml` which contains more information. You set them by doing `config_variable=value`. The main ones to be aware of are:

* `gym` sets which environment to use. The current options are the different difficulty levels of trifinger, `trifinger_difficulty_{1,2,3,4}`

## Inference & Loading Checkpoints

Trained checkpoints from RL Games are saved within the `nn/` directory within the relevant output directory (`output/<date>/time` by default)

To run inference and see the results, you want to go:


```bash
python scripts/rlg_hydra.py args.num_envs=<INT> args.play=True args.checkpoint=/path/to/checkpoint.pth args.headless=False
```

Where `play` tells it not to train and `checkpoint` is the path to load the checkpoint. If you want to start training from a trained checkpoint, then just don't specify `play` (or set it to `False`).
Usually you'll only want a few environments (say, 256) to get smooth rendering performance during inference.


## Citing

```
***TODO*** BibTeX when arXived.
```
