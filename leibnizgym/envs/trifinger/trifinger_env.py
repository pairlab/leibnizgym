"""
@author     Mayank Mittal
@email      mittalma@ethz.ch
@brief      IsaacGym environment for Trifinger robot.
"""

# isaacgym
from isaacgym import gymtorch
from isaacgym import gymapi
# leibnizgym
from leibnizgym.utils import *
from leibnizgym.utils.torch_utils import saturate, unscale_transform, scale_transform, quat_diff_rad
from leibnizgym.envs.env_base import IsaacEnvBase
# leibnizgym - trifinger
from leibnizgym.envs.trifinger.sample import *
from leibnizgym.envs.trifinger.rewards import REWARD_TERMS_MAPPING
from leibnizgym.envs.trifinger.utils import TrifingerDimensions, CuboidalObject
# python
from typing import Union, List, Tuple, Deque
from types import SimpleNamespace
from collections import OrderedDict, deque
import os
import numpy as np
import torch
import torch.nn.functional

# Default configuration for Tri-finger environment
TRIFINGER_DEFAULT_CONFIG_DICT = {
    # Maximum number of steps in an episode.
    "episode_length": 750,
    # Specify difficulty of the task
    # Ref: https://people.tuebingen.mpg.de/felixwidmaier/realrobotchallenge/simulation_phase/tasks.html
    "task_difficulty": 1,
    # Enable force-torque sensor in finger tips or not.
    "enable_ft_sensors": False,
    # Type of low level control of the fingers: ["position", "torque", "position_impedence"]
    "command_mode": "position",
    # Whether to apply safety damping to computed torques
    "apply_safety_damping": True,
    # Whether to fill state buffers or not -- used in Asymmetric PPO implementation.
    # If false: then the state buffers are empty.
    "asymmetric_obs": False,
    # Whether to normalize observations to [-1, 1] or not.
    "normalize_obs": True,
    # Whether to denormalize action from [-1, 1] or not.
    "normalize_action": True,
    # Configuration for resetting the MDP
    "reset_distribution": {
        # Defines how to reset the robot joint state
        "robot_initial_state": {
            # type of distribution: ["default", "random"]
            #  - "default" means that robot is in default configuration.
            #  - "random" means that noise is added to default configuration
            #  - "none" means the no reset is performed between episodes.
            "type": "default",
            "dof_pos_stddev": 0.4,
            "dof_vel_stddev": 0.2
        },
        # Defines how to reset the robot joint state
        "object_initial_state": {
            # type of distribution: ["default", "random", "none"]
            #  - "default": pose is default configuration.
            #  - "random": pose is randomly sampled on the table.
            #  - "none": pose is not reset between episodes - this means that pose
            #            is from the last pose in the previous episode .
            "type": "random",
        }
    },
    "goal_movement": {
        "rotation": {
            "activate": False,
            "rate_magnitude": 0.5, # stdev of angular velocity
        },
    },
    # Reward terms
    "reward_terms": {
        # Reward encouraging movement of fingers towards the cube.
        "finger_reach_object_rate": {
            "activate": True,
            "weight": -750,
            "norm_p": 2,
        },
        # Reward penalising the movement of fingers.
        "finger_move_penalty": {
            "activate": True,
            "weight": -0.1,
        },
        "object_dist": {
            "activate": True,
            "weight": 2000
        },
        "object_rot": {
            "activate": True,
            "weight": 300
        },
        "object_rot_delta": {
            "activate": True,
            "weight": -250,
        },
        "object_move": {
            "activate": True,
            "weight": -750,
        },
    },
    # Termination conditions
    "termination_conditions": {
        # terminates if the object and object goal pose are close
        "success": {
            "activate": True,
            "bonus": 5000.0,
            "position_tolerance": 0.01,  # m
            "orientation_tolerance": 0.2,  # rad
        }
    },
}


class TrifingerEnv(IsaacEnvBase):
    """
    The tri-finger platform comprises of three 3-DOF robot arms for prehensile manipulation [1]. The environment
    is based on the Real Robot Challenge (RRC) organized by Max Plank Institute of Intelligent Systems [2].

    The current environment supports the following objects:
        - cube: object used in phase 1 and 2 of the competition
        - cuboid: object used in the phase 3 of the competition

    References:
    [1] https://sites.google.com/view/trifinger/home-page
    [2] https://real-robot-challenge.com/
    """
    # constants
    # directory where assets for the simulator are present
    _trifinger_assets_dir = os.path.join(get_resources_dir(), "assets", "trifinger")
    # robot urdf (path relative to `_trifinger_assets_dir`)
    _robot_urdf_file = "robot_properties_fingers/urdf/pro/trifingerpro.urdf"
    # stage urdf (path relative to `_trifinger_assets_dir`)
    _stage_urdf_file = "robot_properties_fingers/urdf/high_table_boundary.urdf"
    # object urdf (path relative to `_trifinger_assets_dir`)
    # TODO: Make object URDF configurable.
    _object_urdf_file = "objects/urdf/cube_multicolor_rrc.urdf"
    # physical dimensions of the object
    # TODO: Make object dimensions configurable.
    _object_dims = CuboidalObject(0.065)
    # dimensions of the system
    _dims = TrifingerDimensions
    # Constants for limits
    # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/trifinger_platform.py#L68
    # maximum joint torque (in N-m) applicable on each actuator
    _max_torque_Nm = 0.36
    # maximum joint velocity (in rad/s) on each actuator
    _max_velocity_radps = 10
    # limits of the robot (mapped later: str -> torch.tensor)
    _robot_limits: dict = {
        "joint_position": SimpleNamespace(
            # matches those on the real robot
            low=np.array([-0.33, 0.0, -2.7] * _dims.NumFingers.value, dtype=np.float32),
            high=np.array([1.0, 1.57, 0.0] * _dims.NumFingers.value, dtype=np.float32),
            default=np.array([0.0, 0.9, -1.7] * _dims.NumFingers.value, dtype=np.float32),
        ),
        "joint_velocity": SimpleNamespace(
            low=np.full(_dims.JointVelocityDim.value, -_max_velocity_radps, dtype=np.float32),
            high=np.full(_dims.JointVelocityDim.value, _max_velocity_radps, dtype=np.float32),
            default=np.zeros(_dims.JointVelocityDim.value, dtype=np.float32),
        ),
        "joint_torque": SimpleNamespace(
            low=np.full(_dims.JointTorqueDim.value, -_max_torque_Nm, dtype=np.float32),
            high=np.full(_dims.JointTorqueDim.value, _max_torque_Nm, dtype=np.float32),
            default=np.zeros(_dims.JointTorqueDim.value, dtype=np.float32),
        ),
        "fingertip_position": SimpleNamespace(
            low=np.array([-0.4, -0.4, 0], dtype=np.float32),
            high=np.array([0.4, 0.4, 0.5], dtype=np.float32),
        ),
        "fingertip_orientation": SimpleNamespace(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
        ),
        "fingertip_velocity": SimpleNamespace(
            low=np.full(_dims.VelocityDim.value, -0.2, dtype=np.float32),
            high=np.full(_dims.VelocityDim.value, 0.2, dtype=np.float32),
        ),
        "fingertip_wrench": SimpleNamespace(
            low=np.full(_dims.WrenchDim.value, -1.0, dtype=np.float32),
            high=np.full(_dims.WrenchDim.value, 1.0, dtype=np.float32),
        ),
        # used if we want to have joint stiffness/damping as parameters`
        "joint_stiffness": SimpleNamespace(
            low=np.array([1.0, 1.0, 1.0] * _dims.NumFingers.value, dtype=np.float32),
            high=np.array([50.0, 50.0, 50.0] * _dims.NumFingers.value, dtype=np.float32),
        ),
        "joint_damping": SimpleNamespace(
            low=np.array([0.01, 0.03, 0.0001] * _dims.NumFingers.value, dtype=np.float32),
            high=np.array([1.0, 3.0, 0.01] * _dims.NumFingers.value, dtype=np.float32),
        ),
    }
    # limits of the object (mapped later: str -> torch.tensor)
    _object_limits: dict = {
        "position": SimpleNamespace(
            low=np.array([-0.3, -0.3, 0], dtype=np.float32),
            high=np.array([0.3, 0.3, 0.3], dtype=np.float32),
            default=np.array([0, 0, _object_dims.min_height], dtype=np.float32)
        ),
        "orientation": SimpleNamespace(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
        "velocity": SimpleNamespace(
            low=np.full(_dims.VelocityDim.value, -0.5, dtype=np.float32),
            high=np.full(_dims.VelocityDim.value, 0.5, dtype=np.float32),
            default=np.zeros(_dims.VelocityDim.value, dtype=np.float32)
        ),
    }
    # PD gains for the robot (mapped later: str -> torch.tensor)
    # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/sim_finger.py#L49-L65
    _robot_dof_gains = {
        # The kp and kd gains of the PD control of the fingers.
        # Note: This depends on simulation step size and is set for a rate of 250 Hz.
        "stiffness": [10.0, 10.0, 10.0] * _dims.NumFingers.value,
        "damping": [0.1, 0.3, 0.001] * _dims.NumFingers.value,
        # The kd gains used for damping the joint motor velocities during the
        # safety torque check on the joint motors.
        "safety_damping": [0.08, 0.08, 0.04] * _dims.NumFingers.value
    }
    # History of state: Number of timesteps to save history for
    # Note: Currently used only to manage history of object and frame states.
    #       This can be extended to other observations (as done in ANYmal).
    _state_history_len = 2

    # buffers to store the simulation data
    # goal poses for the object [num. of instances, 7] where 7: (x, y, z, quat)
    _object_goal_poses_buf: torch.Tensor
    # buffer to store the per-timestep rotation of the goal position, if this is enabled
    _object_goal_movement_buf: torch.Tensor
    # DOF state of the system [num. of instances, num. of dof, 2] where last index: pos, vel
    _dof_state: torch.Tensor
    # Rigid body state of the system [num. of instances, num. of bodies, 13] where 13: (x, y, z, quat, v, omega)
    _rigid_body_state: torch.Tensor
    # Root prim states [num. of actors, 13] where 13: (x, y, z, quat, v, omega)
    _actors_root_state: torch.Tensor
    # Force-torque sensor array [num. of instances, num. of bodies * wrench]
    _ft_sensors_values: torch.Tensor
    # DOF position of the system [num. of instances, num. of dof]
    _dof_position: torch.Tensor
    # DOF velocity of the system [num. of instances, num. of dof]
    _dof_velocity: torch.Tensor
    # DOF torque of the system [num. of instances, num. of dof]
    _dof_torque: torch.Tensor
    # Fingertip links state list([num. of instances, num. of fingers, 13]) where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _fingertips_frames_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    # Object prim state [num. of instances, 13] where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _object_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    # keeps track of the number of goal resets
    _successes: torch.Tensor

    def __init__(self, config: dict = None, device: str = 'cpu', verbose: bool = True, visualize: bool = False):
        """Initializes the tri-finger environment configure the buffers.

        Args:
            config: Dictionory containing the configuration (default: TRIFINGER_DEFAULT_CONFIG_DICT).
            device: Torch device to store created buffers at (cpu/gpu).
            verbose: if True, it will display all information.
            visualize: if True, it will open the GUI, otherwise, it will just run the server.
        """
        # load default config
        trifinger_config = TRIFINGER_DEFAULT_CONFIG_DICT
        if config is not None:
            trifinger_config = update_dict(trifinger_config, config)
        # Enable force torque sensor if asymmetric
        if trifinger_config["asymmetric_obs"]:
            trifinger_config["enable_ft_sensors"] = True
        # define spaces for the environment

        # action
        action_dim = self._dims.JointTorqueDim.value if config['command_mode'] != "position_impedance" else self._dims.JointTorqueDim.value * 2

        # observations
        obs_spec = {
            "robot_q": self._dims.GeneralizedCoordinatesDim.value,
            "robot_u": self._dims.GeneralizedVelocityDim.value,
            "object_q": self._dims.ObjectPoseDim.value,
            "object_q_des": self._dims.ObjectPoseDim.value,
            "command": action_dim
        }
        # state
        if trifinger_config["asymmetric_obs"]:
            state_spec = {
                # observations spec
                "robot_q": self._dims.GeneralizedCoordinatesDim.value,
                "robot_u": self._dims.GeneralizedVelocityDim.value,
                "object_q": self._dims.ObjectPoseDim.value,
                "object_q_des": self._dims.ObjectPoseDim.value,
                "command": action_dim,
                # extra observations (added separately to make computations simpler)
                "object_u": self._dims.ObjectVelocityDim.value,
                "fingertip_state": self._dims.NumFingers.value * self._dims.StateDim.value,
                "robot_a": self._dims.GeneralizedVelocityDim.value,
                "fingertip_wrench": self._dims.NumFingers.value * self._dims.WrenchDim.value,
            }
        else:
            state_spec = {}

        # actions
        action_spec = {
            "command":  action_dim
        }
        # define prims present in the scene
        prim_names = ["robot", "stage", "object", "goal_object"]
        # mapping from name to asset instance
        self._gym_assets = dict.fromkeys(prim_names)
        # mapping from name to gym indices
        self._gym_indices = dict.fromkeys(prim_names)
        # mapping from name to gym rigid body handles
        # name of finger tips links i.e. end-effector frames
        fingertips_frames = ["finger_tip_link_0", "finger_tip_link_120", "finger_tip_link_240"]
        self._fingertips_handles = OrderedDict.fromkeys(fingertips_frames, None)
        # mapping from name to gym dof index
        robot_dof_names = list()
        for finger_pos in ['0', '120', '240']:
            robot_dof_names += [f'finger_base_to_upper_joint_{finger_pos}',
                                f'finger_upper_to_middle_joint_{finger_pos}',
                                f'finger_middle_to_lower_joint_{finger_pos}']
        self._robot_dof_indices = OrderedDict.fromkeys(robot_dof_names, None)

        # initialize the super class
        # Note: The base constructor calls the `_setup_sim()` function.
        super().__init__(
            obs_spec, action_spec, state_spec,
            trifinger_config, device=device, verbose=verbose, visualize=visualize
        )
        # initialize the buffers
        self.__initialize()

        self._successes = torch.zeros(self.num_instances, device=self.device, dtype=torch.bool)

        # set the mdp spaces
        self.__configure_mdp_spaces()
        # set the reward term for the MDP
        self._reward_terms = dict()
        print_info("Reward terms: ")
        # add terms to reward
        for reward_term_name, reward_conf in self.config["reward_terms"].items():
            # create instance of reward term
            reward_term = REWARD_TERMS_MAPPING[reward_term_name](reward_term_name, **reward_conf)
            # print reward term
            print(f"\t {reward_term}")
            # add to dictionory if term is active
            self._reward_terms[reward_term_name] = torch.jit.script(reward_term)

    """
    Protected members - Implementation specifics.
    """

    def _setup_sim(self):
        """
        Setup environment and the simulation scene.
        """
        # define ground plane for simulation
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0
        plane_params.dynamic_friction = 0.1
        plane_params.static_friction = 0.1
        # create ground
        self._gym.add_ground(self._sim, plane_params)
        # define scene assets
        self.__create_scene_assets()
        # create environments
        self.__create_envs()

    def _reset_impl(self, instances: torch.Tensor):
        """
        Resets the MDP for given environment instances.

        Args
            instances: The environment instances to reset.
        """

        # A) Reset episode stats buffers
        self._reset_buf[instances] = 0
        self._steps_count_buf[instances] = 0
        self._successes[instances] = 0
        # B) Various randomizations at the start of the episode:
        # -- Default action at start of episode
        self._action_buf[instances] = 0
        # -- Robot base position.
        # -- Stage position.
        # -- Coefficient of restituion and friction for robot, object, stage.
        # -- Mass and size of the object
        # -- Mass of robot links
        # -- Robot joint state
        robot_initial_state_config = self.config["reset_distribution"]["robot_initial_state"]
        self.__sample_robot_state(
            instances,
            distribution=robot_initial_state_config["type"],
            dof_pos_stddev=robot_initial_state_config["dof_pos_stddev"],
            dof_vel_stddev=robot_initial_state_config["dof_vel_stddev"]
        )
        # -- Sampling of initial pose of the object
        object_initial_state_config = self.config["reset_distribution"]["object_initial_state"]
        self.__sample_object_poses(
            instances,
            distribution=object_initial_state_config["type"],
        )
        # -- Sampling of goal pose of the object
        self.__sample_object_goal_poses(
            instances,
            difficulty=self.config["task_difficulty"]
        )
        # C) Extract trifinger indices to reset
        robot_indices = self._gym_indices["robot"][instances].to(torch.int32)
        object_indices = self._gym_indices["object"][instances].to(torch.int32)
        goal_object_indices = self._gym_indices["goal_object"][instances].to(torch.int32)
        all_indices = torch.unique(torch.cat([robot_indices, object_indices, goal_object_indices]))
        # D) Set values into simulator
        # -- DOF
        self._gym.set_dof_state_tensor_indexed(self._sim, gymtorch.unwrap_tensor(self._dof_state),
                                               gymtorch.unwrap_tensor(robot_indices), len(robot_indices))
        # -- actor root states
        self._gym.set_actor_root_state_tensor_indexed(self._sim, gymtorch.unwrap_tensor(self._actors_root_state),
                                                      gymtorch.unwrap_tensor(all_indices), len(all_indices))

    def _goal_reset_impl(self, instances: torch.Tensor):
        # A) Reset episode stats buffers
        self._goal_reset_buf[instances] = 0
        # B) Various randomizations at the start of the episode:
        # -- Sampling of goal pose of the object
        self.__sample_object_goal_poses(
            instances,
            difficulty=self.config["task_difficulty"]
        )
        # C) Extract trifinger indices to reset
        goal_object_indices = self._gym_indices["goal_object"][instances].to(torch.int32)
        all_indices = torch.unique(torch.cat([goal_object_indices]))
        # D) Set values into simulator
        # -- actor root states
        self._gym.set_actor_root_state_tensor_indexed(self._sim, gymtorch.unwrap_tensor(self._actors_root_state),
                                                      gymtorch.unwrap_tensor(all_indices), len(all_indices))

    def _pre_step(self):
        """
        Setting of input actions into simulator before performing the physics simulation step.

        @note The input actions are read from the action buffer variable `_action_buf`.
        """
        # if normalized_action is true, then denormalize them.
        if self.config["normalize_action"]:
            # TODO: Default action should correspond to normalized value of 0.
            action_transformed = unscale_transform(
                self._action_buf,
                lower=self._action_scale.low,
                upper=self._action_scale.high
            )
        else:
            action_transformed = self._action_buf

        # compute command on the basis of mode selected
        if self.config["command_mode"] == 'torque':
            # command is the desired joint torque
            computed_torque = action_transformed
        elif self.config["command_mode"] == 'position':
            # command is the desired joint positions
            desired_dof_position = action_transformed
            # compute torque to apply
            computed_torque = self._robot_dof_gains["stiffness"] * (desired_dof_position - self._dof_position)
            computed_torque -= self._robot_dof_gains["damping"] * self._dof_velocity
        elif self.config["command_mode"] == 'position_impedance':
            # command is the desired joint positions
            desired_dof_position = action_transformed[:, 0:9]
            # compute torque to apply
            computed_torque = action_transformed[:, 9:18] * (desired_dof_position - self._dof_position)
            # computed_torque -= action_transformed[:, 18:27] * self._dof_velocity
            computed_torque -= self._robot_dof_gains["damping"] * self._dof_velocity
        else:
            msg = f"Invalid command mode. Input: {self.config['command_mode']} not in ['torque', 'position']."
            raise ValueError(msg)
        # apply clamping of computed torque to actuator limits
        applied_torque = saturate(
            computed_torque,
            lower=self._robot_limits["joint_torque"].low,
            upper=self._robot_limits["joint_torque"].high
        )
        # apply safety damping and clamping of the action torque if enabled
        if self.config["apply_safety_damping"]:
            # apply damping by joint velocity
            applied_torque -= self._robot_dof_gains["safety_damping"] * self._dof_velocity
            # clamp input
            applied_torque = saturate(
                applied_torque,
                lower=self._robot_limits["joint_torque"].low,
                upper=self._robot_limits["joint_torque"].high
            )
        # set computed torques to simulator buffer.
        self._gym.set_dof_actuation_force_tensor(self._sim, gymtorch.unwrap_tensor(applied_torque))

        self.__update_goal_movement_pre()

    def _post_step(self):
        """
        Setting of buffers after performing the physics simulation step.

        @note Also need to update the reset buffer for the instances that have terminated.
              The termination conditions to check are besides the episode timeout.

        """
        # fill observations buffer
        self._fill_observations_and_states()
        # compute rewards
        self._reward_buf[:] = 0
        terms = self._reward_terms
        rewards = {
            "finger_reach_object_rate": terms["finger_reach_object_rate"].compute(
                self.env_steps_count,
                self._fingertips_frames_state_history[0],
                self._fingertips_frames_state_history[1],
                self._object_state_history[0],
                self._object_state_history[1]
            ),
            "finger_move_penalty": terms["finger_move_penalty"].compute(
                self.config["sim"]["dt"],
                self._fingertips_frames_state_history[0],
                self._fingertips_frames_state_history[1],
            ),
            "object_dist": terms["object_dist"].compute(
                self.config["sim"]["dt"],
                self.env_steps_count,
                self._object_state_history[0],
                self._object_goal_poses_buf
            ),
            "object_rot": terms["object_rot"].compute(
                self.config["sim"]["dt"],
                self.env_steps_count,
                self._object_state_history[0],
                self._object_goal_poses_buf
            ),
            "object_rot_delta": terms["object_rot_delta"].compute(
                self.config["sim"]["dt"],
                self.env_steps_count,
                self._object_state_history[0],
                self._object_state_history[1],
                self._object_goal_poses_buf
            ),
            "object_move": terms["object_move"].compute(
                self._object_state_history[0],
                self._object_state_history[1],
                self._object_goal_poses_buf
            )
        }
        for k, v in rewards.items():
            if self._reward_terms[k].activate:
                self._reward_buf += v
                self._step_info[f"env/rewards/{k}"] = v.mean()

        # check termination conditions (success only)
        self.__check_termination()

        self.__update_goal_movement_post()

    """
    Private functions
    """

    def __initialize(self):
        """Allocate memory to various buffers.
        """
        # change constant buffers from numpy/lists into torch tensors
        # limits for robot
        for limit_name in self._robot_limits:
            # extract limit simple-namespace
            limit_dict = self._robot_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)
        # limits for the object
        for limit_name in self._object_limits:
            # extract limit simple-namespace
            limit_dict = self._object_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)
        # PD gains for actuation
        for gain_name, value in self._robot_dof_gains.items():
            self._robot_dof_gains[gain_name] = torch.tensor(value, dtype=torch.float, device=self.device)

        # store the sampled goal poses for the object: [num. of instances, 7]
        self._object_goal_poses_buf = torch.zeros((self.num_instances, 7), device=self.device, dtype=torch.float)
        # linear vel and angular vel
        self._object_goal_movement_buf = torch.zeros((self.num_instances, 6), device=self.device, dtype=torch.float)
        # get force torque sensor if enabled
        if self.config["enable_ft_sensors"]:
            # joint torques
            dof_force_tensor = self._gym.acquire_dof_force_tensor(self._sim)
            self._dof_torque = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_instances,
                                                                           self._dims.JointTorqueDim.value)
            # force-torque sensor
            num_ft_dims = self._dims.NumFingers.value * self._dims.WrenchDim.value
            sensor_tensor = self._gym.acquire_force_sensor_tensor(self._sim)
            self._ft_sensors_values = gymtorch.wrap_tensor(sensor_tensor).view(self.num_instances, num_ft_dims)
        # get gym GPU state tensors
        actor_root_state_tensor = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        rigid_body_tensor = self._gym.acquire_rigid_body_state_tensor(self._sim)
        # refresh the buffer (to copy memory?)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        # create wrapper tensors for reference (consider everything as pointer to actual memory)
        # DOF
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_instances, -1, 2)
        self._dof_position = self._dof_state[..., 0]
        self._dof_velocity = self._dof_state[..., 1]
        # rigid body
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_instances, -1, 13)
        # root actors
        self._actors_root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        # frames history
        fingertip_handles_indices = list(self._fingertips_handles.values())
        object_indices = self._gym_indices["object"]
        # timestep 0 is current tensor
        curr_history_length = 0
        while curr_history_length < self._state_history_len:
            # add tensors to history list
            self._fingertips_frames_state_history.append(self._rigid_body_state[:, fingertip_handles_indices])
            self._object_state_history.append(self._actors_root_state[object_indices])
            # update current history length
            curr_history_length += 1

    def __configure_mdp_spaces(self):
        """
        Configures the observations, state and action spaces.
        """
        # Action scale for the MDP
        # Note: This is order sensitive.
        if self.config["command_mode"] == "position":
            # action space is joint positions
            self._action_scale.low = self._robot_limits["joint_position"].low
            self._action_scale.high = self._robot_limits["joint_position"].high
        elif self.config["command_mode"] == "torque":
            # action space is joint torques
            self._action_scale.low = self._robot_limits["joint_torque"].low
            self._action_scale.high = self._robot_limits["joint_torque"].high
        elif self.config["command_mode"] == "position_impedance":
            # action space is joint positions
            controls = ['joint_position', 'joint_stiffness']
            self._action_scale.low = torch.cat([self._robot_limits[p].low for p in controls])
            self._action_scale.high = torch.cat([self._robot_limits[p].high for p in controls])
        else:
            msg = f"Invalid command mode. Input: {self.config['command_mode']} not in ['torque', 'position']."
            raise ValueError(msg)

        # Observations scale for the MDP
        # check if policy outputs normalized action [-1, 1] or not.
        if self.config["normalize_action"]:
            obs_action_scale = SimpleNamespace(
                low=torch.full((self.get_action_dim(),), -1, dtype=torch.float, device=self.device),
                high=torch.full((self.get_action_dim(),), 1, dtype=torch.float, device=self.device)
            )
        else:
            obs_action_scale = self._action_scale
        # Note: This is order sensitive.
        self._observations_scale.low = torch.cat([
            self._robot_limits["joint_position"].low,
            self._robot_limits["joint_velocity"].low,
            self._object_limits["position"].low,
            self._object_limits["orientation"].low,
            self._object_limits["position"].low,
            self._object_limits["orientation"].low,
            obs_action_scale.low
        ])
        self._observations_scale.high = torch.cat([
            self._robot_limits["joint_position"].high,
            self._robot_limits["joint_velocity"].high,
            self._object_limits["position"].high,
            self._object_limits["orientation"].high,
            self._object_limits["position"].high,
            self._object_limits["orientation"].high,
            obs_action_scale.high
        ])
        # State scale for the MDP
        if self.config["asymmetric_obs"]:
            # finger tip scaling
            fingertip_state_scale = SimpleNamespace(
                low=torch.cat([
                    self._robot_limits["fingertip_position"].low,
                    self._robot_limits["fingertip_orientation"].low,
                    self._robot_limits["fingertip_velocity"].low,
                ]),
                high=torch.cat([
                    self._robot_limits["fingertip_position"].high,
                    self._robot_limits["fingertip_orientation"].high,
                    self._robot_limits["fingertip_velocity"].high,
                ])
            )
            # Note: This is order sensitive.
            self._states_scale.low = torch.cat([
                self._observations_scale.low,
                self._object_limits["velocity"].low,
                fingertip_state_scale.low.repeat(self._dims.NumFingers.value),
                self._robot_limits["joint_torque"].low,
                self._robot_limits["fingertip_wrench"].low.repeat(self._dims.NumFingers.value),
            ])
            self._states_scale.high = torch.cat([
                self._observations_scale.high,
                self._object_limits["velocity"].high,
                fingertip_state_scale.high.repeat(self._dims.NumFingers.value),
                self._robot_limits["joint_torque"].high,
                self._robot_limits["fingertip_wrench"].high.repeat(self._dims.NumFingers.value),
            ])
        # check that dimensions of scalings are correct
        # count number of dimensions
        state_dim = sum(self.state_spec.values())
        obs_dim = sum(self.obs_spec.values())
        action_dim = sum(self.action_spec.values())
        # check that dimensions match
        # observations
        if self._observations_scale.low.shape[0] != obs_dim or self._observations_scale.high.shape[0] != obs_dim:
            msg = f"Observation scaling dimensions mismatch. " \
                  f"\tLow: {self._observations_scale.low.shape[0]}, " \
                  f"\tHigh: {self._observations_scale.high.shape[0]}, " \
                  f"\tExpected: {obs_dim}."
            raise AssertionError(msg)
        # state
        if self._states_scale.low.shape[0] != state_dim or self._states_scale.high.shape[0] != state_dim:
            msg = f"States scaling dimensions mismatch. " \
                  f"\tLow: {self._states_scale.low.shape[0]}, " \
                  f"\tHigh: {self._states_scale.high.shape[0]}, " \
                  f"\tExpected: {state_dim}."
            raise AssertionError(msg)
        # actions
        if self._action_scale.low.shape[0] != action_dim or self._action_scale.high.shape[0] != action_dim:
            msg = f"Actions scaling dimensions mismatch. " \
                  f"\tLow: {self._action_scale.low.shape[0]}, " \
                  f"\tHigh: {self._action_scale.high.shape[0]}, " \
                  f"\tExpected: {action_dim}."
            raise AssertionError(msg)
        # print the scaling
        if self.verbose:
            print_info(f'MDP Raw observation bounds\n'
                       f'\tLow: {self._observations_scale.low}\n'
                       f'\tHigh: {self._observations_scale.high}')
            print_info(f'MDP Raw state bounds\n'
                       f'\tLow: {self._states_scale.low}\n'
                       f'\tHigh: {self._states_scale.high}')
            print_info(f'MDP Raw action bounds\n'
                       f'\tLow: {self._action_scale.low}\n'
                       f'\tHigh: {self._action_scale.high}')

    def __create_scene_assets(self):
        """ Define Gym assets for stage, robot and object.
        """
        # define assets
        self._gym_assets["robot"] = self.__define_robot_asset()
        self._gym_assets["stage"] = self.__define_stage_asset()
        self._gym_assets["object"] = self.__define_object_asset()
        self._gym_assets["goal_object"] = self.__define_goal_object_asset()
        # display the properties (only for debugging)
        # robot
        print_info("Trifinger Robot Asset: ")
        print(f'\t Number of bodies: {self._gym.get_asset_rigid_body_count(self._gym_assets["robot"])}')
        print(f'\t Number of shapes: {self._gym.get_asset_rigid_shape_count(self._gym_assets["robot"])}')
        print(f'\t Number of dofs: {self._gym.get_asset_dof_count(self._gym_assets["robot"])}')
        print(f'\t Number of actuated dofs: {self._dims.JointTorqueDim.value}')
        # stage
        print_info("Trifinger Stage Asset: ")
        print(f'\t Number of bodies: {self._gym.get_asset_rigid_body_count(self._gym_assets["stage"])}')
        print(f'\t Number of shapes: {self._gym.get_asset_rigid_shape_count(self._gym_assets["stage"])}')

    def __create_envs(self):
        """Create various instances for the environment.
        """
        # define the dof properties for the robot
        robot_dof_props = self._gym.get_asset_dof_properties(self._gym_assets["robot"])
        # set dof properites based on the control mode
        for k, dof_index in enumerate(self._robot_dof_indices.values()):
            # note: since safety checks are employed, the simulator PD controller is not
            #       used. Instead the torque is computed manually and applied, even if the
            #       command mode is 'position'.
            robot_dof_props['driveMode'][dof_index] = gymapi.DOF_MODE_EFFORT
            robot_dof_props['stiffness'][dof_index] = 0.0
            robot_dof_props['damping'][dof_index] = 0.0
            # set dof limits
            robot_dof_props['effort'][dof_index] = self._max_torque_Nm
            robot_dof_props['velocity'][dof_index] = self._max_velocity_radps
            robot_dof_props['lower'][dof_index] = float(self._robot_limits["joint_position"].low[k])
            robot_dof_props['upper'][dof_index] = float(self._robot_limits["joint_position"].high[k])
        # define lower and upper region bound for each environment
        env_lower_bound = gymapi.Vec3(-self.config["spacing"], -self.config["spacing"], 0.0)
        env_upper_bound = gymapi.Vec3(self.config["spacing"], self.config["spacing"], self.config["spacing"])
        num_instances_per_row = int(np.sqrt(self.num_instances))
        # initialize gym indices buffer as a list
        # note: later the list is converted to torch tensor for ease in interfacing with IsaacGym.
        for asset_name in self._gym_indices.keys():
            self._gym_indices[asset_name] = list()
        # count number of shapes and bodies
        max_agg_bodies = 0
        max_agg_shapes = 0
        for asset in self._gym_assets.values():
            max_agg_bodies += self._gym.get_asset_rigid_body_count(asset)
            max_agg_shapes += self._gym.get_asset_rigid_shape_count(asset)
        # iterate and create environment instances
        for env_index in range(self.num_instances):
            # create environment
            env_ptr = self._gym.create_env(self._sim, env_lower_bound, env_upper_bound, num_instances_per_row)
            # begin aggregration mode if enabled
            # TODO: What does aggregation mode mean?
            if self.config["aggregate_mode"]:
                self._gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            # add trifinger robot to environment
            trifinger_actor = self._gym.create_actor(env_ptr, self._gym_assets["robot"], gymapi.Transform(),
                                                     "robot", env_index, 0, 0)
            trifinger_idx = self._gym.get_actor_index(env_ptr, trifinger_actor, gymapi.DOMAIN_SIM)
            # add stage to environment
            stage_handle = self._gym.create_actor(env_ptr, self._gym_assets["stage"], gymapi.Transform(),
                                                  "stage", env_index, 1, 0)
            stage_idx = self._gym.get_actor_index(env_ptr, stage_handle, gymapi.DOMAIN_SIM)
            # add object to environment
            object_handle = self._gym.create_actor(env_ptr, self._gym_assets["object"], gymapi.Transform(),
                                                   "object", env_index, 0, 0)
            object_idx = self._gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            # add goal object to environment
            goal_handle = self._gym.create_actor(env_ptr, self._gym_assets["goal_object"], gymapi.Transform(),
                                                 "goal_object", env_index + self.num_instances, 0, 0)
            goal_object_idx = self._gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            # add force-torque sensor to fingertips
            if self.config["enable_ft_sensors"]:
                # enable joint force sensors
                self._gym.enable_actor_dof_force_sensors(env_ptr, trifinger_actor)
                # add force-torque sensor to finger tips
                for fingertip_handle in self._fingertips_handles.values():
                    self._gym.create_force_sensor(env_ptr, fingertip_handle, gymapi.Transform())
            # change settings of DOF
            self._gym.set_actor_dof_properties(env_ptr, trifinger_actor, robot_dof_props)
            # add color to instances
            stage_color = gymapi.Vec3(0.73, 0.68, 0.72)
            self._gym.set_rigid_body_color(env_ptr, stage_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, stage_color)
            # end aggregation mode if enabled
            if self.config["aggregate_mode"]:
                self._gym.end_aggregate(env_ptr)
            # add instances to list
            self._envs.append(env_ptr)
            self._gym_indices["robot"].append(trifinger_idx)
            self._gym_indices["stage"].append(stage_idx)
            self._gym_indices["object"].append(object_idx)
            self._gym_indices["goal_object"].append(goal_object_idx)
        # convert gym indices from list to tensor
        for asset_name, asset_indices in self._gym_indices.items():
            self._gym_indices[asset_name] = torch.tensor(asset_indices, dtype=torch.long, device=self.device)

    """
    Helper functions - define assets
    """

    def __define_robot_asset(self):
        """ Define Gym asset for robot.
        """
        # define tri-finger asset
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.flip_visual_attachments = False
        robot_asset_options.fix_base_link = True
        robot_asset_options.collapse_fixed_joints = False
        robot_asset_options.disable_gravity = False
        robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        robot_asset_options.thickness = 0.001
        robot_asset_options.angular_damping = 0.01
        if self._physics_engine == gymapi.SIM_PHYSX:
            robot_asset_options.use_physx_armature = True
        # load tri-finger asset
        trifinger_asset = self._gym.load_asset(self._sim, self._trifinger_assets_dir,
                                               self._robot_urdf_file, robot_asset_options)
        # set the link properties for the robot
        # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/sim_finger.py#L563
        trifinger_props = self._gym.get_asset_rigid_shape_properties(trifinger_asset)
        for p in trifinger_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.8
        self._gym.set_asset_rigid_shape_properties(trifinger_asset, trifinger_props)
        # extract the frame handles
        for frame_name in self._fingertips_handles.keys():
            self._fingertips_handles[frame_name] = self._gym.find_asset_rigid_body_index(trifinger_asset,
                                                                                         frame_name)
            # check valid handle
            if self._fingertips_handles[frame_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid handle received for frame: `{frame_name}`."
                print_error(msg)
        # extract the dof indices
        # Note: need to write actuated dofs manually since the system contains fixed joints as well which show up.
        for dof_name in self._robot_dof_indices.keys():
            self._robot_dof_indices[dof_name] = self._gym.find_asset_dof_index(trifinger_asset, dof_name)
            # check valid handle
            if self._robot_dof_indices[dof_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid index received for DOF: `{dof_name}`."
                print_error(msg)
        # return the asset
        return trifinger_asset

    def __define_stage_asset(self):
        """ Define Gym asset for stage.
        """
        # define stage asset
        stage_asset_options = gymapi.AssetOptions()
        stage_asset_options.disable_gravity = True
        stage_asset_options.fix_base_link = True
        stage_asset_options.thickness = 0.001
        # load stage asset
        stage_asset = self._gym.load_asset(self._sim, self._trifinger_assets_dir,
                                           self._stage_urdf_file, stage_asset_options)
        # set stage properties
        stage_props = self._gym.get_asset_rigid_shape_properties(stage_asset)
        # iterate over each mesh
        for p in stage_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
        self._gym.set_asset_rigid_shape_properties(stage_asset, stage_props)
        # return the asset
        return stage_asset

    def __define_object_asset(self):
        """ Define Gym asset for object.
        """
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        # load object asset
        object_asset = self._gym.load_asset(self._sim, self._trifinger_assets_dir,
                                            self._object_urdf_file, object_asset_options)
        # set object properties
        # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/collision_objects.py#L96
        object_props = self._gym.get_asset_rigid_shape_properties(object_asset)
        for p in object_props:
            p.friction = 1.0
            p.torsion_friction = 0.001
            p.restitution = 0.0
        self._gym.set_asset_rigid_shape_properties(object_asset, object_props)
        # return the asset
        return object_asset

    def __define_goal_object_asset(self):
        """ Define Gym asset for goal object.
        """
        # define object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        object_asset_options.fix_base_link = not self.config['goal_movement']['rotation']['activate']
        object_asset_options.thickness = 0.001
        # load object asset
        goal_object_asset = self._gym.load_asset(self._sim, self._trifinger_assets_dir,
                                                 self._object_urdf_file, object_asset_options)
        # return the asset
        return goal_object_asset

    """
    Helper functions - MDP
    """

    def _fill_observations_and_states(self):
        """
        Fills observation and state buffer with the current state of the system.
        """
        # refresh memory buffers
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        if self.config["enable_ft_sensors"] or self.config["asymmetric_obs"]:
            self._gym.refresh_dof_force_tensor(self._sim)
            self._gym.refresh_force_sensor_tensor(self._sim)
        # extract frame handles
        fingertip_handles_indices = list(self._fingertips_handles.values())
        object_indices = self._gym_indices["object"]
        # update state histories
        self._fingertips_frames_state_history.appendleft(self._rigid_body_state[:, fingertip_handles_indices])
        self._object_state_history.appendleft(self._actors_root_state[object_indices])
        # fill the observations and states buffer
        self.__fill_observations()
        self.__fill_states()
        # TODO (@mayank): add observation noise
        # normalize observations if flag is enabled
        if self.config["normalize_obs"]:
            # for normal obs
            self._obs_buf = scale_transform(
                self._obs_buf,
                lower=self._observations_scale.low,
                upper=self._observations_scale.high
            )
            # for asymmetric obs
            if self.config["asymmetric_obs"]:
                self._states_buf = scale_transform(
                    self._states_buf,
                    lower=self._states_scale.low,
                    upper=self._states_scale.high
                )

    def __fill_observations(self):
        """
        Fills observation buffer with the current state of the system.
        """
        # generalized coordinates
        start_offset = 0
        end_offset = start_offset + self._dims.GeneralizedCoordinatesDim.value
        self._obs_buf[:, start_offset:end_offset] = self._dof_position
        # generalized velocities
        start_offset = end_offset
        end_offset = start_offset + self._dims.GeneralizedVelocityDim.value
        self._obs_buf[:, start_offset:end_offset] = self._dof_velocity
        # object pose
        start_offset = end_offset
        end_offset = start_offset + self._dims.ObjectPoseDim.value
        self._obs_buf[:, start_offset:end_offset] = self._object_state_history[0][:, 0:7]
        # object desired pose
        start_offset = end_offset
        end_offset = start_offset + self._dims.ObjectPoseDim.value
        self._obs_buf[:, start_offset:end_offset] = self._object_goal_poses_buf
        # previous action from policy
        start_offset = end_offset
        end_offset = start_offset + self.get_action_dim()
        self._obs_buf[:, start_offset:end_offset] = self._action_buf

    def __fill_states(self):
        """
        Fills states buffer with the current state of the system.
        """
        # if states spec is empty then return
        if not self.config["asymmetric_obs"]:
            return
        # copy common observations to state
        start_offset = 0
        end_offset = self.get_obs_dim()
        self._states_buf[:, start_offset:end_offset] = self._obs_buf
        # object velcity
        start_offset = end_offset
        end_offset = start_offset + self._dims.ObjectVelocityDim.value
        self._states_buf[:, start_offset:end_offset] = self._object_state_history[0][:, 7:13]
        # fingertip state
        num_fingerip_states = self._dims.NumFingers.value * self._dims.StateDim.value
        start_offset = end_offset
        end_offset = start_offset + num_fingerip_states
        self._states_buf[:, start_offset:end_offset] = \
            self._fingertips_frames_state_history[0].reshape(self.num_instances, num_fingerip_states)
        # add force-torque measurements
        if self.config["enable_ft_sensors"]:
            # joint torque
            start_offset = end_offset
            end_offset = start_offset + self._dims.JointTorqueDim.value
            self._states_buf[:, start_offset:end_offset] = self._dof_torque
            # force-torque sensors
            start_offset = end_offset
            end_offset = start_offset + self._dims.NumFingers.value * self._dims.WrenchDim.value
            self._states_buf[:, start_offset:end_offset] = self._ft_sensors_values

    def __check_termination(self):
        """Check whether the episode is done per environment.
        """
        # Extract configuration for termination conditions
        termination_config = self.config["termination_conditions"]
        # Termination condition - successful completition
        # Calculate distance between current object and goal
        object_goal_position_dist = torch.norm(
            self._object_goal_poses_buf[:, 0:3] - self._object_state_history[0][:, 0:3],
            p=2, dim=-1
        )
        # log theoretical number of r eseats
        goal_position_reset = torch.le(object_goal_position_dist,
                 termination_config["success"]["position_tolerance"])
        current_position_goal = torch.sum(goal_position_reset)
        self._step_info['env/current_position_goal/count'] = current_position_goal
        # For task with difficulty 4, we need to check if orientation matches as well.
        # Compute the difference in orientation between object and goal pose
        object_goal_orientation_dist = quat_diff_rad(self._object_state_history[0][:, 3:7],
                                                     self._object_goal_poses_buf[:, 3:7])
        # Check for distance within tolerance
        goal_orientation_reset = torch.le(object_goal_orientation_dist,
                                          termination_config["success"]["orientation_tolerance"])
        self._step_info['env/current_orientation_goal/count'] = torch.sum(goal_orientation_reset)

        if self.config['task_difficulty'] < 4:
            # Check for task completion if position goal is within a threshold
            task_completion_reset = goal_position_reset
        elif self.config['task_difficulty'] == 4:
            # Check for task completion if both position + orientation goal is within a threshold
            task_completion_reset = torch.logical_and(goal_position_reset, goal_orientation_reset)
        else:
            # Check for task completion if both orientation goal is within a threshold
            task_completion_reset = goal_orientation_reset

        if termination_config["success"]["activate"]:
            # add termination bonus
            env_ids = torch.nonzero(task_completion_reset).squeeze()
            # add the reward bonus for episodes terminated
            self._reward_buf[env_ids] += termination_config["success"]["bonus"]
            self._goal_reset_buf = task_completion_reset
            self._successes = self._successes + self._goal_reset_buf
        else:
            self._successes = torch.logical_and(self._goal_reset_buf, self._successes)

        successes_mean = np.mean((self._successes).cpu().numpy())
        self._step_info['env/average_consecutive_success'] = successes_mean

    def __sample_robot_state(self, instances: torch.Tensor, distribution: str = 'default',
                             dof_pos_stddev: float = 0.0, dof_vel_stddev: float = 0.0):
        """Samples the robot DOF state based on the settings.

        Type of robot initial state distribution: ["default", "random"]
             - "default" means that robot is in default configuration.
             - "random" means that noise is added to default configuration
             - "none" means that robot is configuration is not reset between episodes.

        Args:
            instances: A tensor contraining indices of environment instances to reset.
            distribution: Name of distribution to sample initial state from: ['default', 'random']
            dof_pos_stddev: Noise scale to DOF position (used if 'type' is 'random')
            dof_vel_stddev: Noise scale to DOF velocity (used if 'type' is 'random')
        """
        # number of samples to generate
        num_samples = instances.size()[0]
        # sample dof state based on distribution type
        if distribution == "none":
            return
        elif distribution == "default":
            # set to default configuration
            self._dof_position[instances] = self._robot_limits["joint_position"].default
            self._dof_velocity[instances] = self._robot_limits["joint_velocity"].default
        elif distribution == "random":
            # sample uniform random from (-1, 1)
            dof_state_dim = self._dims.JointPositionDim.value + self._dims.JointVelocityDim.value
            dof_state_noise = 2 * torch.rand((num_samples, dof_state_dim,), dtype=torch.float,
                                             device=self.device) - 1
            # set to default configuration
            self._dof_position[instances] = self._robot_limits["joint_position"].default
            self._dof_velocity[instances] = self._robot_limits["joint_velocity"].default
            # add noise
            # DOF position
            start_offset = 0
            end_offset = self._dims.JointPositionDim.value
            self._dof_position[instances] += dof_pos_stddev * dof_state_noise[:, start_offset:end_offset]
            # DOF velocity
            start_offset = end_offset
            end_offset += self._dims.JointVelocityDim.value
            self._dof_velocity[instances] += dof_vel_stddev * dof_state_noise[:, start_offset:end_offset]
        else:
            msg = f"Invalid robot initial state distribution. Input: {distribution} not in [`default`, `random`]."
            raise ValueError(msg)
        # reset robot fingertips state history
        for idx in range(1, self._state_history_len):
            self._fingertips_frames_state_history[idx][instances] = 0.0

    def __sample_object_poses(self, instances: torch.Tensor, distribution: str):
        """Sample poses for the cube.

        Type of distribution: ["default", "random", "none"]
             - "default" means that pose is default configuration.
             - "random" means that pose is randomly sampled on the table.
             - "none" means no resetting of object pose between episodes.

        Args:
            instances: A tensor contraining indices of environment instances to reset.
            distribution: Name of distribution to sample initial state from: ['default', 'random']
        """
        # number of samples to generate
        num_samples = instances.size()[0]
        # sample poses based on distribution type
        if distribution == "none":
            return
        elif distribution == "default":
            pos_x, pos_y, pos_z = self._object_limits["position"].default
            orientation = self._object_limits["orientation"].default
        elif distribution == "random":
            # For initialization
            pos_x, pos_y = random_xy(num_samples, self._object_dims.max_com_distance_to_center, self.device)
            pos_z = self._object_dims.size[2] / 2
            orientation = random_yaw_orientation(num_samples, self.device)
        else:
            msg = f"Invalid object initial state distribution. Input: {distribution} " \
                  "not in [`default`, `random`, `none`]."
            raise ValueError(msg)
        # set buffers into simulator
        # extract indices for goal object
        object_indices = self._gym_indices["object"][instances]
        # set values into buffer
        # object buffer
        self._object_state_history[0][instances, 0] = pos_x
        self._object_state_history[0][instances, 1] = pos_y
        self._object_state_history[0][instances, 2] = pos_z
        self._object_state_history[0][instances, 3:7] = orientation
        self._object_state_history[0][instances, 7:13] = 0
        # reset object state history
        for idx in range(1, self._state_history_len):
            self._object_state_history[idx][instances] = 0.0
        # root actor buffer
        self._actors_root_state[object_indices] = self._object_state_history[0][instances]

    def __sample_object_goal_poses(self, instances: torch.Tensor, difficulty: int):
        """Sample goal poses for the cube and sets them into the desired goal pose buffer.

        Args:
            instances: A tensor contraining indices of environment instances to reset.
            difficulty: Difficulty level. The higher, the more difficult is the goal.

        Possible levels are:
            - -1:  Random goal position on the table, including yaw orientation.
            - 1: Random goal position on the table, no orientation.
            - 2: Fixed goal position in the air with x,y = 0.  No orientation.
            - 3: Random goal position in the air, no orientation.
            - 4: Random goal pose in the air, including orientation.
        """
        # number of samples to generate
        num_samples = instances.size()[0]
        # sample poses based on task difficulty
        if difficulty == -1:
            # For initialization
            pos_x, pos_y = random_xy(num_samples, self._object_dims.max_com_distance_to_center, self.device)
            pos_z = self._object_dims.size[2] / 2
            orientation = random_yaw_orientation(num_samples, self.device)
        elif difficulty == 1:
            # Random goal position on the table, no orientation.
            pos_x, pos_y = random_xy(num_samples, self._object_dims.max_com_distance_to_center, self.device)
            pos_z = self._object_dims.size[2] / 2
            orientation = default_orientation(num_samples, self.device)
        elif difficulty == 2:
            # Fixed goal position in the air with x,y = 0.  No orientation.
            pos_x, pos_y = 0.0, 0.0
            pos_z = self._object_dims.min_height + 0.05
            orientation = default_orientation(num_samples, self.device)
        elif difficulty == 3:
            # Random goal position in the air, no orientation.
            pos_x, pos_y = random_xy(num_samples, self._object_dims.max_com_distance_to_center, self.device)
            pos_z = random_z(num_samples, self._object_dims.min_height, self._object_dims.max_height, self.device)
            orientation = default_orientation(num_samples, self.device)
        elif difficulty == 4 or difficulty == 5:
            # Random goal pose in the air, including orientation.
            # Note: Set minimum height such that the cube does not intersect with the
            #       ground in any orientation
            pos_x, pos_y = random_xy(num_samples, self._object_dims.max_com_distance_to_center, self.device)
            pos_z = random_z(num_samples, self._object_dims.radius_3d, self._object_dims.max_height, self.device)
            orientation = random_orientation(num_samples, self.device)
        elif difficulty == 6:
            # Fixed goal position in the air with x,y = 0.  No orientation.
            pos_x, pos_y = 0.0, 0.0
            pos_z = self._object_dims.min_height + 0.05
            # random orientation
            orientation = random_orientation(num_samples, self.device)
        else:
            msg = f"Invalid difficulty index for task: {difficulty}."
            raise ValueError(msg)

        if self.config['goal_movement']['rotation']['activate']:
            angular_vel = random_angular_vel(num_samples, self.device, self.config['goal_movement']['rotation']['rate_magnitude'])
            print(angular_vel)
            self._object_goal_movement_buf[instances, 3:6] = angular_vel
        else:
            self._object_goal_movement_buf[instances, 3:6] = 0.0

        # extract indices for goal object
        goal_object_indices = self._gym_indices["goal_object"][instances]
        # set values into buffer
        # object goal buffer
        self._object_goal_poses_buf[instances, 0] = pos_x
        self._object_goal_poses_buf[instances, 1] = pos_y
        self._object_goal_poses_buf[instances, 2] = pos_z
        self._object_goal_poses_buf[instances, 3:7] = orientation
        # root actor buffer
        self._actors_root_state[goal_object_indices, 0:7] = self._object_goal_poses_buf[instances]
        self._actors_root_state[goal_object_indices, 7:13] = self._object_goal_movement_buf[instances]

    def __update_goal_movement_pre(self):
        """Updates the angular velocity of the cube to be correct based on the movement velocity."""
        if self.config['goal_movement']['rotation']['activate']:
            goal_object_indices = self._gym_indices["goal_object"]
            self._actors_root_state[goal_object_indices, 10:13] = self._object_goal_movement_buf[:, 3:6]

            goal_object_indices = self._gym_indices["goal_object"].to(torch.int32)
            # D) Set values into simulator
            # -- actor root states
            self._gym.set_actor_root_state_tensor_indexed(self._sim, gymtorch.unwrap_tensor(self._actors_root_state),
                                                          gymtorch.unwrap_tensor(goal_object_indices), len(goal_object_indices))

    def __update_goal_movement_post(self):
        """Updates the goal pose buffer based on the movement that the cube has undergone."""

        if self.config['goal_movement']['rotation']['activate']:
            goal_object_indices = self._gym_indices["goal_object"]
            self._object_goal_poses_buf[:] = self._actors_root_state[goal_object_indices, 0:7]

# EOF
