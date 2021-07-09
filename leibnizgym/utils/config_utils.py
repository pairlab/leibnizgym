"""
@author     Arthur Allshire
@email      arthur.allshire@mail.utoronto.ca
@author     Mayank Mittal
@email      mittalma@ethz.ch

@brief Defines argparse utilities for RL-games adapter.
"""

# isaacgym
from isaacgym import gymapi
from isaacgym import gymutil
# leibizgym
from leibnizgym.utils.helpers import update_dict, get_resources_dir
from leibnizgym.utils.errors import InvalidTaskNameError
from leibnizgym.utils.message import *
# python
from typing import Dict
import argparse
import os
import yaml


def join_config_path(config_name: str, base_path: str = None) -> str:
    """Concatenates the configuration for RL games directory.

    Args
        config_name: The name of the configuration file to load.
        base_path: The path where config is stored (default: `resources/config`)

    Returns
        The joined path for RL games config file.
    """
    # check for base path: default 'resources/config'
    if base_path is None:
        base_path = os.path.join(get_resources_dir(), 'config')
    # config file name
    config_filename = os.path.join(base_path, config_name)
    # check if path exists
    if not os.path.exists(config_filename):
        msg = f"Configuration file not found: {config_filename}"
        print_error(msg)
        raise ValueError(msg)

    return config_filename


def retrieve_cfg_paths(args: argparse.Namespace) -> (str, str, str):
    """Returns the config paths for agent and environment based on the args.

    TODO (@mayank): Simplify the passing of args everywhere.

    Args:
        args: command line arguments
    Returns:
        (logdir path, train config dict, task config dict)
    """
    # get task configuration file.
    if args.task == "Trifinger":
        # print task related args input
        print_info(f"[Trifinger]: Task difficulty : {args.task_variant}")
        print_info(f"[Trifinger]: Training type   : {args.training_type}")
        # directory to save logs in (tensorboard>?)
        logdir = os.path.join(args.logdir, "trifinger")
        # path where gym config is stored
        trifinger_path = os.path.join(get_resources_dir(), 'config', 'trifinger')
        # complete file name for yaml config
        task_cfg_file = join_config_path(f"difficulty_{args.task_variant}.yaml",
                                         base_path=os.path.join(trifinger_path, 'gym'))
        # resolve training config for environment
        agent_cfg_file = join_config_path(f"{args.training_type}_ppo.yaml",
                                          base_path=os.path.join(trifinger_path, 'rlg'))
    else:
        raise InvalidTaskNameError(args.task)
    # print the files being loaded
    print_notify(f"Task config : {task_cfg_file}")
    print_notify(f"Agent config: {agent_cfg_file}")
    # return the paths
    return logdir, agent_cfg_file, task_cfg_file


def load_cfg(args: argparse.Namespace) -> (Dict, Dict, str):
    """Loads the config given the args

    Args:
        args: command line arguments
    Returns:
        (environment_config, rl_games_config, logdir)
    """
    # get paths to various logging and configurations
    logdir, cfg_train_pth, cfg_env_pth = retrieve_cfg_paths(args)
    # load task config dictionary
    with open(os.path.join(os.getcwd(), cfg_env_pth), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # load agent config dictionary
    with open(os.path.join(os.getcwd(), cfg_train_pth), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    print(args)

    return update_cfg(cfg, cfg_train, logdir, args)


def update_cfg(cfg, cfg_train, logdir, args):
    # Override settings if passed by the command line
    # Override number of environments
    if args.num_envs > 0:
        cfg["num_instances"] = args.num_envs
    # Override episode length
    if args.episode_length > 0:
        cfg["episode_length"] = args.episode_length
    # Override the physics engine
    cfg["physics_engine"] = 'physx' if args.physics_engine == gymapi.SIM_PHYSX else 'flex'
    # Override the phyiscs settings
    # NOTE: All arg params are overriden if they are provided in the YAML config, except number of threads.
    args_sim_settings = {
        "dt": 1.0 / 60.0,
        "use_gpu_pipeline": args.device == "GPU",
        "num_client_threads": args.slices,
        "flex": {
            "shape_collision_margin": 0.01,
            "num_outer_iterations": 4,
            "num_inner_iterations": 10
        },
        "physx": {
            "solver_type": 1,
            "num_position_iterations": 4,
            "num_velocity_iterations": 0,
            "num_threads": 4,
            "use_gpu": args.use_gpu,
            "num_subscenes": args.subscenes,
            "max_gpu_contact_pairs": 8 * 1024 * 1024,
        }
    }
    # if yaml config has sim settings, override above args settings
    try:
        cfg["sim"] = update_dict(args_sim_settings, cfg["sim"])
    except AttributeError:
        cfg["sim"] = args_sim_settings
    # Override number of threads for physics if passed by CLI
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        cfg["sim"]["physx"]["num_threads"] = args.num_threads
    # Override domain randomization
    # TODO (@arthur): DR for leibnizgym
    # if "task" in cfg:
    #     if "randomize" not in cfg["task"]:
    #         cfg["task"]["randomize"] = args.randomize
    #     else:
    #         cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    # else:
    #     cfg["task"] = {"randomize": False}

    # Set cfg to enable asymmetric training
    if "asym" in args.training_type:
        cfg["asymmetric_obs"] = True

    # %% RLG config
    exp_name = cfg_train["params"]["config"]['name']
    #  Override experiment name
    if args.experiment_name != 'Base':
        exp_name = f"{args.experiment_name}_{args.task_type}_{args.device}_{str(args.physics_engine).split('_')[-1]}"
    # if cfg["task"]["randomize"]:
    #     exp_name += "_DR"
    cfg_train["params"]["config"]['name'] = exp_name
    # Set number of environment instances
    if "num_instances" in cfg:
        cfg_train["params"]["config"]["minibatch_size"] = cfg["num_instances"]
        cfg_train["params"]["config"]["num_actors"] = cfg["num_instances"]
        # Set minibatch size for central value config
        if "central_value_config" in cfg_train["params"]["config"]:
            cfg_train["params"]["config"]["central_value_config"]["minibatch_size"] = cfg["num_instances"]
    else:
        msg = "Must provide num_instances for rl_games to work, either in config or via `num_envs` argument."
        raise ValueError(msg)
    # Check if continue training for loaded checkpoint
    if args.resume > 0:
        cfg_train["params"]["load_checkpoint"] = True
    # Set maximum number of training iterations (epochs)
    if args.max_iterations > 0:
        cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

    # acquire seed from the agent configuration dict
    try:
        seed = cfg_train.get("params").get("seed", 42)
    except AttributeError:
        seed = cfg_train.get("seed", 42)
    # Override the seed for both agent and task
    if args.seed > 0:
        seed = args.seed
    cfg["seed"] = seed
    cfg_train["seed"] = seed

    return cfg, cfg_train, logdir


def get_args(benchmark: bool = False, use_rlg_config: bool = False) -> argparse.Namespace:
    """Defines the custom args for argparsing and parses them.

    Args:
        benchmark: Add additional params for benchmarking the environment.
        use_rlg_config: Add RL-games specfic params or not.

    Returns:
        Parsed CLI arguments.
    """
    custom_parameters = [
        # sets verbosity of the run
        {"name": "--verbose", "action": "store_true", "default": False,
         "help": "Set verbosity of the environment (useful for debugging)."},
        # sets whether to train/test
        {"name": "--test", "action": "store_true", "default": False,
         "help": "Run trained policy, no training"},
        # sets whether to train further or test with provided NN weights
        {"name": "--resume", "type": int, "default": 0,
         "help": "Resume training or start testing from a checkpoint"},
        # sets whether to run GUI for visualization or not.
        {"name": "--headless", "action": "store_true", "default": False,
         "help": "Force display off at all times"},
        # sets the name of the task
        {"name": "--task", "type": str, "default": "Trifinger",
         "help": "Task supported: ['Trifinger'] "},
        # sets the task variant to run
        {"name": "--task_variant", "type": str, "default": "1",
         "help": "Task-specific config, eg for Trifinger allows setting difficulty"},
        # sets the training type such as asymm, LSTM etc.
        {"name": "--training_type", "type": str, "default": "vanilla",
         "help": "choose type of training to be used, allows choosing eg. asymm, LSTM, etc."},
        # sets the task type: that is implementation language
        {"name": "--task_type", "type": str, "default": "Python",
         "help": "Choose Python or C++"},
        # sets the device for the environment
        {"name": "--device", "type": str, "default": "GPU",
         "help": "Choose CPU or GPU device for running physics"},
        # sets the device for the RL agent
        {"name": "--ppo_device", "type": str, "default": "GPU",
         "help": "Choose CPU or GPU device for inferencing PPO network"},
        # sets the directory to log data into
        {"name": "--logdir", "type": str, "default": "logs/"},
        # sets the experiment name
        {"name": "--experiment_name", "type": str, "default": "Base"},
        # sets the training config file
        {"name": "--cfg_train", "type": str, "default": "Base",
         "help": "Agent training config file path"},
        # sets the config environment file
        {"name": "--cfg_env", "type": str, "default": "Base",
         "help": "Environment config file path"},
        # sets the seed of the experiment
        {"name": "--seed", "type": int, "default": -1,
         "help": "Random seed"},
        # sets maximum number of training iterations
        {"name": "--max_iterations", "type": int, "default": 0,
         "help": "Set a maximum number of training iterations"},
        # sets the number of instances to run
        {"name": "--num_envs", "type": int, "default": 0,
         "help": "Number of environments to create - override config file"},
        # sets maximum number of steps in episode
        {"name": "--episode_length", "type": int, "default": 0,
         "help": "Episode length, by default is read from yaml config"},
        # checks applying of physics DR
        {"name": "--randomize", "action": "store_true", "default": False,
         "help": "Apply physics domain randomization"}]

    # configurations specfic for RL-games
    if use_rlg_config:
        # sets whether to train/test
        custom_parameters += [
            {"name": "--play", "action": "store_true", "default": False,
             "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
            # sets the checkpoint point path to load weights from
            {"name": "--checkpoint", "type": str, "default": "",
             "help": "Path to the saved weights, only for rl_games RL library"},
        ]

    # further config for benchmarking the running
    if benchmark:
        custom_parameters += [
            # number of processes to run
            {"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
            # whether to apply random actions or not
            {"name": "--random_actions", "action": "store_true",
             "help": "Run benchmark with random actions instead of inferencing"},
            # number of timing reports for benchmark
            {"name": "--bench_len", "type": int, "default": 10,
             "help": "Number of timing reports"},
            # name of the benchamrking file to dump results into
            {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    # parse arguments
    args = gymutil.parse_arguments(description="RL Policy Training using Leibnizgym",
                                   custom_parameters=custom_parameters)
    # check if run is training/test/play
    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True
    # return parsed arguments
    return args
