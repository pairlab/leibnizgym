# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# isaacgym-rlgpu
from isaacgym import rlgpu
from rlgpu.utils.config import set_np_formatting, set_seed
# leibniz-gym: dump all environments for loading
from leibnizgym.envs.trifinger import TrifingerEnv as Trifinger
# leibnizgym
from leibnizgym.wrappers.vec_task import VecTaskPython
from leibnizgym.utils.config_utils import load_cfg, get_args
from leibnizgym.utils.errors import InvalidTaskNameError
from leibnizgym.utils.message import *
# rl-games
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.common import wrappers
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext
import torch
import numpy as np
# python
import os
import argparse
import yaml
from datetime import datetime


def parse_vec_task(args: argparse.Namespace, cfg: dict) -> VecTaskPython:
    """Parses the configuration parameters for the environment task.

    TODO (@mayank): Remove requirement for args and make this a normal function
                    inside utils.
    Args:
        args: command line arguments.
        cfg: environment configuration dictionary (task)

    Returns:
        TThe vectorized RL-env wrapped around the task.
    """
    # create native task and pass custom config
    if args.task_type == "Python":
        # check device on which to run agent and environment
        if args.device == "CPU":
            print_info("Running using python CPU...")
            # check if agent is on different device
            sim_device = 'cpu'
            ppo_device = 'cuda:0' if args.ppo_device == "GPU" else 'cpu'
        else:
            print_info("Running using python GPU...")
            sim_device = 'cuda:0'
            ppo_device = 'cuda:0'
        # create the IsaacEnvBase defined using leibnizgym
        try:
            task = eval(args.task)(config=cfg, device=sim_device,
                                   visualize=not args.headless,
                                   verbose=args.verbose)
        except NameError:
            raise InvalidTaskNameError(args.task)
        # wrap environment around vec-python wrapper
        env = VecTaskPython(task, rl_device=ppo_device, clip_obs=5, clip_actions=1)
    else:
        raise ValueError(f"No task of type `{args.task_type}` in leibnizgym.")

    return env


def create_rlgpu_env(**kwargs):
    """
    Creates the task from configurations and wraps it using RL-games wrappers if required.
    """
    # TODO (@arthur): leibnizgym parse task
    env = parse_vec_task(cli_args, task_cfg)
    # print the environment information
    print_info(env)
    # save environment config into file
    env.dump_config(os.path.join(logdir, 'env_config.yaml'))
    # wrap around the environment
    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RlGamesGpuEnvAdapter(vecenv.IVecEnv):
    """
    Adapter from VecPythonTask to Rl-Games VecEnv.
    """

    def __init__(self, config_name: str, num_actors: int, **kwargs):
        # this basically calls the `create_rlgpu_env()` function for RLGPU environment.
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        # check if environment is for asymmetric PPO or not
        self.use_global_obs = (self.env.num_states > 0)
        # get initial observations
        self.full_state = {
            "obs": self.env.reset()
        }
        # get state if assymmetric environment
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()

    """
    Properties
    """

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {
            'num_envs': self.env.num_envs,
            'action_space': self.env.action_space,
            'observation_space': self.env.observation_space
        }
        # print the spaces (for debugging)
        print(">> Action space: ", info['action_space'])
        print(">> Observation space: ", info['observation_space'])
        # check if environment is for asymmetric PPO or not
        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(">> State space: ", info['state_space'])
        # return the information about spaces
        return info

    """
    Operations
    """

    def reset(self):
        # reset the environment
        self.full_state["obs"] = self.env.reset()
        # check if environment is for asymmetric PPO or not
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def step(self, action):
        # step through the environment
        next_obs, reward, is_done, info = self.env.step(action)
        # check if environment is for asymmetric PPO or not
        # TODO (@arthur): Improve the return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, [[], info]
        else:
            return self.full_state["obs"], reward, is_done, [[], info]


# register the rl-games adapter to use inside the runner
vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RlGamesGpuEnvAdapter(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'vecenv_type': 'RLGPU',
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
})


class LeibnizAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.game_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if not infos:
            return
        if len(infos) > 0 and isinstance(infos[0], dict):
            for ind in done_indices:
                if len(infos) <= ind // self.algo.num_agents:
                    continue
                info = infos[ind // self.algo.num_agents]
                game_res = None
                if 'battle_won' in info:
                    game_res = info['battle_won']
                if 'scores' in info:
                    game_res = info['scores']

                if game_res is not None:
                    self.game_scores.update(torch.from_numpy(np.asarray([game_res])).to(self.algo.ppo_device))
        if len(infos) > 1 and isinstance(infos[1], dict):  # allow direct logging from env
            self.direct_info = infos[1]

    def after_clear_stats(self):
        self.game_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.game_scores.current_size > 0:
            mean_scores = self.game_scores.get_mean()
            self.writer.add_scalar('scores/mean', mean_scores, frame)
            self.writer.add_scalar('scores/iter', mean_scores, epoch_num)
            self.writer.add_scalar('scores/time', mean_scores, total_time)
        for k, v in self.direct_info.items():
            self.writer.add_scalar(k, v, frame)

def run_rlg_hydra(hydra_cfg):
    global task_cfg, agent_cfg_train, cli_args, logdir, vargs
    from omegaconf import OmegaConf
    task_cfg = OmegaConf.to_container(hydra_cfg.gym)
    agent_cfg_train = OmegaConf.to_container(hydra_cfg.rlg)
    cli_args= hydra_cfg.args
    logdir = cli_args['logdir']
    vargs = OmegaConf.to_container(cli_args)
    run_rlg()


def run_rlg():
    global logdir
    # Create default directories for weights and statistics
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    # set numpy formatting for printing only
    set_np_formatting()

    # append the timestamp to logdir
    now = datetime.now()
    now_dir_name = now.strftime("%m-%d-%Y-%H-%M-%S")
    logdir = os.path.join(logdir, now_dir_name)
    os.makedirs(logdir, exist_ok=True)
    # print the common info
    print_notify(f'Saving logs at: {logdir}')
    print_notify(f'Verbosity     : {cli_args.verbose}')
    print_notify(f'Seed          : {agent_cfg_train["seed"]}')
    # set logdir and seed
    cli_args.logdir = logdir
    set_seed(agent_cfg_train["seed"])
    # print training configuration for debugging
    if cli_args.verbose:
        print_info(f'Agent training configuration: ')
        print_dict(agent_cfg_train)
        print(40 * '-')
    # save agent config into file
    with open(os.path.join(logdir, 'agent_config.yaml'), 'w') as file:
        yaml.dump(agent_cfg_train, file)
    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = Runner(LeibnizAlgoObserver())
    runner.load(agent_cfg_train)
    runner.reset()
    runner.run(vargs)

if __name__ == '__main__':
    # get CLI arguments
    cli_args = get_args(use_rlg_config=True)
    # parse arguments to load configurations
    task_cfg, agent_cfg_train, logdir = load_cfg(cli_args)
    vargs = vars(args)
    run_rlg()

# EOF
