import hydra
import os
import logging
from omegaconf import DictConfig, OmegaConf, open_dict
from dataclasses import dataclass, field
from omegaconf import MISSING
from enum import Enum
from typing import Dict, List, Any, Union, Optional
from hydra.core.config_store import ConfigStore
from scripts.rlg_train import run_rlg_hydra
from hydra import utils, slurm_utils
try:
    import wandb
except ImportError:
    pass

@dataclass
class SimConfig:
    """Configuration for the IsaacGym simulator."""
    dt: float = 0.02
    substeps: int =  4
    up_axis: str = "z"
    use_gpu_pipeline: bool = MISSING
    num_client_threads: int = 0
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    physx: Dict[str, Any] = field(default_factory = lambda: {
        "num_threads": 4,
        "solver_type": 1,
        "use_gpu": False, # set to False to run on CPU
        "num_position_iterations": 8,
        "num_velocity_iterations": 0,
        "contact_offset": 0.002,
        "rest_offset": 0.0,
        "bounce_threshold_velocity": 0.5,
        "max_depenetration_velocity": 1000.0,
        "default_buffer_size_multiplier": 5.0,
    })
    flex: Dict[str, Any] = field(default_factory=lambda: {
        "num_outer_iterations": 5,
        "num_inner_iterations": 20,
        "warm_start": 0.8,
        "relaxation": 0.75,
    })

@dataclass
class EnvConfig:
    """Configuration for all instances of `EnvBase`."""
    env_name: str = MISSING
    # general env settings
    num_instances: int = MISSING
    seed: int = MISSING
    spacing: float = 1.0
    aggregate_mode: bool = True
    # command settings
    control_decimation: int = 1
    # physics settings
    physics_engine: str = MISSING
    sim: SimConfig = SimConfig()

@dataclass
class Trifinger(EnvConfig):
    """Configuration for all instances of `TrifingerEnv`."""
    env_name: str = "Trifinger"
    episode_length: int = 750
    task_difficulty: int = MISSING
    enable_ft_sensors: bool = False
    # observation settings
    asymmetric_obs: bool = False
    normalize_obs: bool = True
    # reset distribution settings
    apply_safety_damping: bool = True
    command_mode: str = "torque"
    normalize_action: bool = True
    reset_distribution:  Dict[str, Any] = field(default_factory=lambda:{
      "object_initial_state": {
        "type": "random"
      },
      "robot_initial_state": {
        "dof_pos_stddev": 0.4,
        "dof_vel_stddev": 0.2,
        "type": "default"
      }
    })
    # reward settings
    reward_terms:Dict[str, Any] = field(default_factory=lambda: {
      "finger_move_penalty":{
        "activate": True,
        "weight": -0.1,
      },
      "finger_reach_object_rate": {
        "activate": True,
        "norm_p": 2,
        "weight": -750,
      },
      "object_dist": {
        "activate": True,
        "weight": 2000,
      },
      "object_rot": {
        "activate": False,
        "weight": 300
      },
      "object_rot_delta": {
        "activate": False,
        "weight": -250
      },
      "object_move": {
        "activate": False,
        "weight": -750,
      }
    })
    # # termination settings
    termination_conditions: Dict[str, Any] = field(default_factory=lambda: {
      "success": {
        "activate": False,
        "bonus": 5000.0,
        "orientation_tolerance": 0.1,
        "position_tolerance": 0.01,
      }
    })

@dataclass
class TrifingerDifficulty1(Trifinger):
    """Trifinger Difficulty 1 Configuration."""
    task_difficulty = 1

@dataclass
class TrifingerDifficulty2(Trifinger):
    """Trifinger Difficulty 2 Configuration."""
    task_difficulty = 2

@dataclass
class TrifingerDifficulty3(Trifinger):
    """Trifinger Difficulty 3 Configuration."""
    task_difficulty = 3

@dataclass
class TrifingerDifficulty4(Trifinger):
    task_difficulty = 4
    reward_terms:Dict[str, Any] = field(default_factory=lambda: {
        "finger_move_penalty":{
            "activate": True,
            "weight": -0.1,
        },
        "finger_reach_object_rate": {
            "activate": True,
            "norm_p": 2,
            "weight": -750,
        },
        "object_dist": {
            "activate": True,
            "weight": 2000,
        },
        "object_rot": { # rot must be enabled for difficulty 4!
            "activate": True,
            "weight": 300
        },
        "object_rot_delta": {
            "activate": False,
            "weight": -250
        },
        "object_move": {
            "activate":False,
            "weight": -750,
        }
    })

@dataclass
class TrifingerRotationTest(Trifinger):
    """Mode for testing to try to get the rotation reward up and running."""
    task_difficulty = 5
    episode_length = 75
    reward_terms:Dict[str, Any] = field(default_factory=lambda: {
        "finger_move_penalty":{
            "activate": False,
            "weight": -0.1,
        },
        "finger_reach_object_rate": {
            "activate": True,
            "norm_p": 2,
            "weight": -250,
        },
        "object_dist": {
            "activate": False,
            "weight": 2000,
        },
        "object_rot": {
            "activate": True,
            "weight": 2000,
            "epsilon": 0.01,
            "scale": 3.0,
        },
        "object_rot_delta": {
            "activate": False,
            "weight": -250
        },
        "object_move": {
            "activate": False,
            "weight": -750,
        }
    })
    termination_conditions: Dict[str, Any] = field(default_factory=lambda: {
        "success": {
            "activate": False,
            "bonus": 5000.0,
            "orientation_tolerance": 0.1,
            "position_tolerance": 0.01,
        }
    })



@dataclass
class TrifingerRotationCurriculum(Trifinger):
    """Mode for testing to try to get the rotation reward up and running."""
    task_difficulty = 4
    episode_length = 750
    reward_terms:Dict[str, Any] = field(default_factory=lambda: {
        "finger_move_penalty":{
            "activate": True,
            "weight": -0.1,
        },
        "finger_reach_object_rate": {
            "activate": True,
            "norm_p": 2,
            "weight": -250,
            "thresh_sched_start": 0,
            "thresh_sched_end": 1e7,
        },
        "object_dist": {
            "activate": True,
            "weight": 2000,
            "thresh_sched_start": 0,
            "thresh_sched_end": 10e10,
        },
        "object_rot": {
            "activate": True,
            "weight": 2000,
            "epsilon": 0.01,
            "scale": 3.0,
            "thresh_sched_start": 1e7,
            "thresh_sched_end": 1e10, # dont end!
        },
        "object_rot_delta": {
            "activate": False,
            "weight": -250
        },
        "object_move": {
            "activate": False,
            "weight": -750,
        }
    })
    termination_conditions: Dict[str, Any] = field(default_factory=lambda: {
        "success": {
            "activate": False,
            "bonus": 5000.0,
            "orientation_tolerance": 0.25,
            "position_tolerance": 0.02,
        }
    })

@dataclass
class RLGConfig:
    """Configuration for RLGames."""
    asymmetric_obs: bool = MISSING  # argument specifying whether this config requires asymmetric states
    params: Dict[Any, Any] = MISSING

@dataclass
class RLGArgs:
    verbose: bool = False

@dataclass
class Args:
    """Items which must be propogated down to other areas of configuration or which are expected as args in other places in the config."""

    # configuration about the env creation
    cfg_env:str = 'Base'
    cfg_train:str = 'Base'
    # todo - maybe do some validation on this? should match up to the config selected
    task:str = 'Trifinger'
    task_type:str = 'Python'  # Choose Python or C++
    experiment_name:str = 'Base'  # used in RLGames

    # confuguration about the env itself
    num_envs:int = 256  # overrides the default number of envs
    # todo - investigate interaction between this and pyDR
    randomize:bool = False  # whether to apply physics domain randomisation

    # other misc rags
    seed:int = 7 # random seed
    verbose:bool = False
    logdir:str = 'logs/'  # backs up the configs for this run

    # devcie config
    physics_engine: Any = 'physx' # field(default_factory=lambda: gymapi.SIM_PHYSX) # 'physx' or 'flex'
    device:str = 'GPU'  # CPU or GPU for running physics
    ppo_device:str = 'GPU'  # whether to use GPU for inference with PPO

    # RLGames Arguments
    play:bool = False  # if set runs trained policy (for use with rl games)
    train: bool = MISSING # opposite of play
    checkpoint:str = ''  # used to set checkpoint path

    # Common Gym Arguments
    headless:bool = False  # disables rendering
    compute_device_id:int = 0  # for CUDA
    graphics_deice_id:int = 0  # graphics device id

    wandb_project_name: str = 'trifinger-manip'
    wandb_log: bool = True


@dataclass
class Config:
    """Base config class."""
    gym: EnvConfig = MISSING
    rlg: Dict[str, Any] = MISSING
    # these config items populate those down the tree.
    # TODO is to figure out a better solution using eg specializing configuration in hydra
    args: Args = Args()

    conda: str = MISSING
    max_running: int = MISSING
    max_pending: int = MISSING
    wait: bool = MISSING
    extra: Optional[str] = MISSING
    next_day: Optional[str] = MISSING
    slurm: Dict[str, Any] = MISSING

@dataclass
class ConfigSlurm:
    """Base config class."""
    gym: EnvConfig = MISSING
    rlg: Dict[str, Any] = MISSING
    # these config items populate those down the tree.
    # TODO is to figure out a better solution using eg specializing configuration in hydra
    args: Args = Args()

def update_cfg(cfg):
    """Modifies cfg by copying key arguments to the correct places.

    Args:
        cfg: Hydra config to modify
    """

    cfg.args.train = not cfg.args.play
    # Override settings if passed by the command line
    # Override number of environments
    cfg.gym.num_instances = cfg.args.num_envs
    # Override the phyiscs settings
    cfg.gym.sim.use_gpu_pipeline = cfg.args.device == "GPU"
    cfg.gym.sim.physx.use_gpu = cfg.args.device == "GPU"
    cfg.gym.physics_engine = cfg.args.physics_engine

    # Set cfg to enable asymmetric training
    cfg.gym.asymmetric_obs = cfg.rlg.asymmetric_obs

    # %% RLG config
    #  Override experiment name
    if cfg.args.experiment_name != 'Base':
        cfg.rlg.params.config.name = f"{cfg.args.experiment_name}_{cfg.args.task_type}_{cfg.args.device}_{str(cfg.args.physics_engine).split('_')[-1]}"

    cfg.rlg.params.load_checkpoint = cfg.args.checkpoint != ''
    cfg.rlg.params.load_path = cfg.args.checkpoint

    # Set number of environment instances
    with open_dict(cfg):
        cfg.rlg["params"]["config"]["minibatch_size"] = cfg.args.num_envs
        cfg.rlg["params"]["config"]["num_actors"] = cfg.args.num_envs
        # Set minibatch size for central value config
        if "central_value_config" in cfg.rlg["params"]["config"]:
            cfg.rlg["params"]["config"]["central_value_config"]["minibatch_size"] = cfg.args.num_envs
        cfg.gym.seed = cfg.args.seed
        cfg.rlg.seed = cfg.args.seed

def get_config_store():
    # Instantiates the different configurations in the correct groups.
    cs = ConfigStore.instance()
    cs.store(group="gym", name="trifinger_difficulty_1", node=TrifingerDifficulty1)
    cs.store(group="gym", name="trifinger_difficulty_2", node=TrifingerDifficulty2)
    cs.store(group="gym", name="trifinger_difficulty_3", node=TrifingerDifficulty3)
    cs.store(group="gym", name="trifinger_difficulty_4", node=TrifingerDifficulty4)
    cs.store(group="gym", name="trifinger_rotation_test", node=TrifingerRotationTest)
    cs.store(group="gym", name="trifinger_rotation_curriculum", node=TrifingerRotationCurriculum)

    # Don't need to instantiate the RLG configs as they are still yaml's - see corresponding directory.
    cs.store(name="config", node=Config)

@hydra.main(config_name="config", config_path="../resources/config")
def launch_rlg_hydra(cfg: DictConfig):
    log = logging.getLogger(__name__)

    if 'SLURM_JOB_ID' in os.environ:
        # cfg.slurm.job_name = os.environ['SLURM_JOB_ID']

        slurm_utils.symlink_hydra(cfg, os.getcwd())

        j_dir = slurm_utils.get_j_dir(cfg)
        checkpoint_dir = os.path.join(j_dir, os.environ['SLURM_JOB_ID'])
        checkpoint = os.path.join(checkpoint_dir, f'{cfg.rlg.params.config.name}.pth')
        if os.path.exists(checkpoint):
            log.info(f'Preemption checkpoint detected at {checkpoint}... will restore')
            cfg.rlg.params.load_checkpoint = True
            cfg.rlg.params.load_path = checkpoint
        else:
            log.info(f'No checkpoint detected at {checkpoint}')
        cfg.rlg.params.config.preemption_checkpoint_dir = checkpoint_dir
    else:
        log.warning("Warning - not detected being run as SLURM, assuming we are running locally.")

    if cfg.args.wandb_log:
        wandb.init(
            project=cfg.args.wandb_project_name,
            config=OmegaConf.to_container(cfg),
            sync_tensorboard=True,
            id=os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else None,
            resume='allow',
        )

    update_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))
    run_rlg_hydra(cfg)

if __name__ == "__main__":
    cs = get_config_store()
    launch_rlg_hydra()
