"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Modifications (02.02.2020)
@author     Mayank Mittal
@email      mittalma@ethz.ch
@brief      Base environment for reinforcement learning using IsaacGym.
"""

# isaacgym
from isaacgym import gymapi
# leibnitzgym
from leibnizgym.utils.message import *
from leibnizgym.utils.helpers import update_dict
# python
from typing import Dict, Tuple, Union, List
from types import SimpleNamespace
import sys
import numpy as np
import random
import torch
import yaml

# default configuration dictionory for simulator
ISAACGYM_DEFAULT_CONFIG_DICT = {
    # Seed of the experiment
    "seed": 0,
    # Number of environment instances
    "num_instances": 1,
    # Spacing between each environment instance
    "spacing": 1.0,
    # Number of physics simulation steps to perform.
    "control_decimation": 1,
    # Maximum number of steps in an episode.
    # None implies no termination of episode via step count.
    "episode_length": None,
    # Aggregate simulation bodies for batch processing
    "aggregate_mode": True,
    # Physics engine to use: ["physx", "flex"]
    "physics_engine": "physx",
    # for physics settings
    "sim": {
        # simulation timestep
        "dt": 0.02,
        # number of substeps to perform
        "substeps": 2,
        # axis pointing upwards (robotics: "z", graphics: "y")
        "up_axis": "z",
        # acceleration due to gravity
        "gravity": [0.0, 0.0, -9.81],
        # number of client threads that process env slices
        "num_client_threads": 0,
        # use GPU pipeline or not
        "use_gpu_pipeline": False,
        # for physics configuration
        "physx": {
            # type of solver to use: {0: pgs, 1: tgs}
            "solver_type": 1,
            "num_position_iterations": 4,
            "num_velocity_iterations": 0,
            "num_threads": 4,
            "use_gpu": False,
            "num_subscenes": 0,
            "max_gpu_contact_pairs": 8 * 1024 * 1024,
        },
        "flex": {
            "shape_collision_margin": 0.01,
            "num_outer_iterations": 4,
            "num_inner_iterations": 10
        }
    },
}

class IsaacEnvBase:
    """
    IsaacGym provides high-fiedlity 3D world simulator that accelerates RL research [1]. This class provides the base
    interfaces to initialize the gym simulator. By using Isaac SDK [2], different robots can be setted up by inheriting
    this class.

    The implementation borrows inspiration from [3].

    [1] https://developer.nvidia.com/isaac-sdk
    [2] "GPU-Accelerated Robotic Simulation for Distributed Reinforcement Learning", Liang et al., 2018
    [3] https://github.com/robotlearn/pyrobolearn/blob/master/pyrobolearn/simulators/isaac.py
    """

    # buffers to store the simulation data
    # state information: [num. of instances, state dim]
    _states_buf: torch.Tensor
    # observations information : [num. of instances, obs dim]
    _obs_buf: torch.Tensor
    # action information: [num. of instances, action dim]
    _action_buf: torch.Tensor
    # termination: [num. of instances,]
    _reset_buf: torch.Tensor
    # termination: [num. of instances,]
    _goal_reset_buf: torch.Tensor
    # rewards: [num. of instances,]
    _reward_buf: torch.Tensor
    # number of steps in each instance: [num. of instances,]
    _steps_count_buf: torch.Tensor
    # list of environment simulation pointers to store
    _envs = list()
    # scalars logged on each step. reset at the start of step method
    _step_info: Dict[str, float]

    def __init__(self, obs_spec: Dict[str, int], action_spec: Dict[str, int], state_spec: Dict[str, int],
                 config: dict = None, device: str = 'cpu', verbose: bool = True, visualize: bool = False):
        """
        Initialize the IsaacGym simulator.

        Args:
            obs_spec: Observations names to size mapping.
            action_spec: Action names to size mapping.
            state_spec: State names to size mapping.
            config: Dictionory containing the configuration for the environment (default: ISAACGYM_DEFAULT_CONFIG_DICT).
            device: Torch device to store created buffers at (cpu/gpu).
            verbose: if True, it will display all information.
            visualize: if True, it will open the GUI, otherwise, it will just run the server.
        """
        # copy input arguments into class members
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.state_spec = state_spec
        self.device = device
        self.verbose = verbose
        self.visualize = visualize
        # load the config if provided
        self.config = ISAACGYM_DEFAULT_CONFIG_DICT
        if config is not None:
            self.config = update_dict(self.config, config)
        # display config for debug
        if self.verbose:
            print_info("Environment configuration: ")
            print_dict(self.config, nesting=0)
            print('-' * 40)
        # extract commonly used parameters from config
        self.num_instances = self.config["num_instances"]
        self.control_decimation = self.config["control_decimation"]
        self.episode_length = self.config["episode_length"]

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        # acquire gym handle
        self._gym = gymapi.acquire_gym()
        # define gym variables
        # simulation
        self._sim_params = None
        self._sim = None
        # visualization
        self._enable_viewer_sync = True
        self._viewer = None
        # define config variables for creating simulation handle
        self._physics_engine = None
        self._graphics_deivce = None
        # define variables to store ranges for observations, states, and action
        self._observations_scale = SimpleNamespace(low=None, high=None)
        self._states_scale = SimpleNamespace(low=None, high=None)
        self._action_scale = SimpleNamespace(low=None, high=None)
        # setup the simulation
        self.__setup()
        # set seed of the environment
        self.seed(self.config["seed"])

    """
    Configurations
    """

    def set_gravity(self, gravity: Tuple[float, float, float] = (0, 0, -9.81)):
        """Set the gravity in the simulator with the given acceleration.
        By default, there is no gravitational force enabled in the simulator.

        Args:
            gravity (list, tuple of 3 floats): acceleration in the x, y, z directions.
        """
        gravity = gymapi.Vec3(*gravity)
        self._sim_params.gravity = gravity
        self._gym.set_sim_params(self._sim, self._sim_params)

    def set_sim_params(self, params: gymapi.SimParams):
        """Set the simulator physics parameters.

        Args:
            params: Input simulator physics parameters datatype.
        """
        self._sim_params = params
        self._gym.set_sim_params(self._sim, self._sim_params)

    def set_camera_lookat(self, pos: Union[List[float], Tuple[float, float, float]],
                          target: Union[List[float], Tuple[float, float, float]]):
        """Sets the viewer camera position and orientation.

        Args:
            pos: The camera eye's position in world coordinates.
            target: The camera eye's target position in world coordinates.
        """
        cam_pos = gymapi.Vec3(*pos)
        cam_target = gymapi.Vec3(*target)
        self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

    """
    Properties
    """

    def get_gravity(self) -> np.ndarray:
        """Returns the gravity set in the simulator.
        """
        gravity = self._sim_params.gravity  # return gymapi.Vec3
        return np.asarray([gravity[0], gravity[1], gravity[2]])

    def get_sim_params(self) -> gymapi.SimParams:
        """Returns the simulator physics parameters.
        """
        return self._sim_params

    def get_state_shape(self) -> torch.Size:
        """Returns the size of the state buffer: [num. instances, num. of state]
        """
        return self._states_buf.size()

    def get_obs_shape(self) -> torch.Size:
        """Returns the size of the observation buffer: [num. instances, num. of obs]
        """
        return self._obs_buf.size()

    def get_action_shape(self) -> torch.Size:
        """Returns the size of the action buffer: [num. instances, num. of action]
        """
        return self._action_buf.size()

    def get_num_instances(self) -> int:
        """Returns number of environment instances
        """
        return self.num_instances

    def get_state_dim(self) -> int:
        """Returns imensions of state specfication.
        """
        return self.get_state_shape()[1]

    def get_obs_dim(self) -> int:
        """Returns imensions of obs specfication.
        """
        return self.get_obs_shape()[1]

    def get_action_dim(self) -> int:
        """Returns imensions of action specfication.
        """
        return self.get_action_shape()[1]

    """
    Properties - Access to internal buffers. 
    """

    @property
    def states_buf(self) -> torch.Tensor:
        """Returns state information."""
        return self._states_buf

    @property
    def obs_buf(self) -> torch.Tensor:
        """Returns observation information."""
        return self._obs_buf

    @property
    def action_buf(self) -> torch.Tensor:
        """Returns action information."""
        return self._action_buf

    @property
    def reward_buf(self) -> torch.Tensor:
        """Returns rewards information."""
        return self._reward_buf

    @property
    def dones_buf(self) -> torch.Tensor:
        """Returns episode termination information."""
        return self._reset_buf

    @property
    def env_steps_count(self) -> int:
        """Returns the total number of environment steps aggregated across parallel environments."""
        return self._gym.get_frame_count(self._sim) * self.num_instances

    """
    Operations
    """

    def dump_config(self, filename: str):
        """Dumps the environment configuration into a YAML file.

        Args:
            filename: The path to the file to save configuration into.
        """
        # check ending
        if not filename.endswith('.yaml'):
            filename += '.yaml'
        # check if directory exists
        dir_name = os.path.dirname(filename)
        os.makedirs(dir_name, exist_ok=True)
        # save yaml file
        with open(filename, 'w') as file:
            yaml.dump(self.config, file)

    @staticmethod
    def seed(seed: int = None):
        """ Set the seed of the environment.

        Args:
            seed: Seed number.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self) -> torch.Tensor:
        """
        Reset at the start of the episode.

        @note: The returned tensors are on the device set in to the environment config.

        Returns:
            Returns the observation tensor [num. instances, obs. dim]
        """
        # resets the episode and fills the observation buffers
        self._reset_impl(torch.arange(0, self.num_instances, device=self.device))
        # apply action
        self._pre_step()
        # step physics for control decimation
        self._gym.simulate(self._sim)
        # fetch results
        if self.device == 'cpu':
            self._gym.fetch_results(self._sim, True)
        # Compute observations
        self._fill_observations_and_states()
        # return observations
        return self._obs_buf.clone().detach()

    def step(self, action: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Apply input action on the environment; reset any which have reached the episode length.

        @note: The returned tensors are on the device set in to the environment config.

        Args:
            action: Action to apply on the simulator. It is a tensor of  shape [N, A] where N is number
                of instances and A is action dimension.

        Returns:
            Tuple containing tensors for  observation buffer, rewards, termination state, along with a dictionory
            with extra information.
        """
        self._step_info = {} # reset the step info
        # if input command is numpy array, convert to torch tensor
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float, device=self.device)
        # check input tensor spec
        action_shape = (self.num_instances, self.get_action_dim())
        if tuple(action.size()) != action_shape:
            msg = f"Invalid shape for tensor `action`. Input: {tuple(action.size())} != {action_shape}."
            raise ValueError(msg)
        # copy input action command into buffer
        self._action_buf = action.clone().to(self.device)
        # Note: The reset is performed before pre-step instead of after post-step
        # The reason is that for certain cases, the observations may include buffers (force sensors, kinematics)
        # which are not updated after reset since they require stepping through physics.
        # reset environment that have been terminated
        env_ids = torch.nonzero(self._reset_buf).view(-1)
        if len(env_ids) > 0:
            self._reset_impl(env_ids)
        goal_env_ids = torch.nonzero(self._goal_reset_buf).view(-1)
        if len(goal_env_ids) > 0:
            self._goal_reset_impl(goal_env_ids)
        # apply action
        self._pre_step()
        # step physics for control decimation
        for _ in range(self.control_decimation):
            self._gym.simulate(self._sim)
        # fetch results
        if self.device == 'cpu':
            self._gym.fetch_results(self._sim, True)
        # compute observations, rewards, resets, ...
        self._post_step()
        # increment number of steps in all instances
        self._steps_count_buf += 1
        # set reset flags for episodes which have reached maximum number of steps.
        if self.episode_length is not None:
            timeout_reset = torch.greater_equal(self._steps_count_buf, self.episode_length)
            self._reset_buf = torch.logical_or(self._reset_buf, timeout_reset)
        # extract the buffers to return
        obs = self._obs_buf
        rewards = self._reward_buf
        dones = torch.logical_and(self._reset_buf, self._goal_reset_buf)
        # return MDP
        return obs, rewards, dones, self._step_info

    def render(self):
        """
        Render the viewer for visualization.
        """
        if self._viewer is not None:
            # check for window closed
            if self._gym.query_viewer_has_closed(self._viewer):
                sys.exit(1)
            # check for keyboard events
            for event in self._gym.query_viewer_action_events(self._viewer):
                if event.action == "QUIT" and event.value > 0:
                    sys.exit(0)
                elif event.action == "toggle_viewer_sync" and event.value > 0:
                    self._enable_viewer_sync = not self._enable_viewer_sync
            # fetch results
            if self.device != 'cpu':
                self._gym.fetch_results(self._sim, True)
            # step graphics
            if self._enable_viewer_sync:
                self._gym.step_graphics(self._sim)
                self._gym.draw_viewer(self._viewer, self._sim, True)
            else:
                self._gym.poll_viewer_events(self._viewer)
        else:
            print_warn("The function `render()` called without visualization enabled.")

    def close(self):
        """
        Cleanup after exiting.
        """
        # viewer
        if self._viewer is not None:
            self._gym.destroy_viewer(self._viewer)
        # sim
        if self._sim is not None:
            self._gym.destroy_sim(self._sim)

    """
    Protected members - Implementation specifics.
    """

    def _setup_sim(self):
        """
        Setup environment and the simulation scene.
        """
        raise NotImplementedError

    def _fill_observations_and_states(self):
        """
        Fills observations and states buffer with the current state of the system.
        """
        raise NotImplementedError

    def _reset_impl(self, instances: torch.Tensor):
        """
        Resets the MDP for given environment instances.

        Args
            instances: A tensor containing indices of environment instances to reset.
        """
        raise NotImplementedError

    def _goal_reset_impl(self, instances: torch.Tensor):
        """
        Resets the goal position for the given environment instances.

        Args
            instances: A tensor containing indices of environment instances to reset.
        """
        raise NotImplementedError

    def _pre_step(self):
        """
        Setting of input actions into simulator before performing the physics simulation step.

        @note The input actions are read from the action buffer variable `_action_buf`.
        """
        raise NotImplementedError

    def _post_step(self)->Dict[str, float]:
        """
        Setting of buffers after performing the physics simulation step.

        @note Also need to update the reset buffer for the instances that have terminated.
              The termination conditions to check are besides the episode timeout.

        """
        raise NotImplementedError

    """
    Private members
    """

    def __parse_sim_params(self) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()
        # check correct up-axis
        if self.config["sim"]["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {self.config['sim']['up_axis']}"
            print_error(msg)
            raise ValueError(msg)
        # assign general sim parameters
        sim_params.dt = self.config["sim"]["dt"]
        sim_params.num_client_threads = self.config["sim"]["num_client_threads"]
        sim_params.use_gpu_pipeline = self.config["sim"]["use_gpu_pipeline"]
        # assign up-axis
        if self.config["sim"]["up_axis"] == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y
        # assign gravity
        sim_params.gravity = gymapi.Vec3(*self.config["sim"]["gravity"])
        # configure physics parameters
        if self.config["physics_engine"] == 'physx':
            # set the parameters
            for opt in self.config["sim"]["physx"].keys():
                setattr(sim_params.physx, opt, self.config["sim"]["physx"][opt])
        else:
            # set the parameters
            for opt in self.config["sim"]["flex"].keys():
                setattr(sim_params.flex, opt, self.config["sim"]["flex"][opt])
        # display the parameters for physics
        print_info(f"Simulation physics parameters: \n{sim_params}")
        # return the configured params
        return sim_params

    def __initialize(self):
        """Allocate memory to various buffers.
        """
        # count number of dimensions
        state_dim = sum(self.state_spec.values())
        obs_dim = sum(self.obs_spec.values())
        action_dim = sum(self.action_spec.values())
        # print for debugging mostly
        print_info(f"Observations: {obs_dim}")
        print_dict(self.obs_spec, nesting=0)
        print_info(f"States: {state_dim}")
        print_dict(self.state_spec, nesting=0)
        print_info(f"Action: {action_dim}")
        print_dict(self.action_spec, nesting=0)
        print('-' * 20)
        # allocate memory to ranges for spaces
        # state
        self._states_scale.low = torch.full((state_dim,), -float('inf'), dtype=torch.float)
        self._states_scale.high = torch.full((state_dim,), float('inf'), dtype=torch.float)
        # observations
        self._observations_scale.low = torch.full((obs_dim,), -float('inf'), dtype=torch.float)
        self._observations_scale.high = torch.full((obs_dim,), float('inf'), dtype=torch.float)
        # action
        self._action_scale.low = torch.full((action_dim,), -float('inf'), dtype=torch.float)
        self._action_scale.high = torch.full((action_dim,), float('inf'), dtype=torch.float)
        # allocate memory to buffers
        # state
        self._states_buf = torch.zeros((self.num_instances, state_dim), device=self.device, dtype=torch.float)
        # observations
        self._obs_buf = torch.zeros((self.num_instances, obs_dim), device=self.device, dtype=torch.float)
        # action
        self._action_buf = torch.zeros((self.num_instances, action_dim), device=self.device, dtype=torch.float)
        # termination
        self._reset_buf = torch.zeros(self.num_instances, device=self.device, dtype=torch.bool)
        # termination
        self._goal_reset_buf = torch.zeros(self.num_instances, device=self.device, dtype=torch.bool)
        # reward
        self._reward_buf = torch.zeros(self.num_instances, device=self.device, dtype=torch.float)
        # number of steps taken
        self._steps_count_buf = torch.zeros(self.num_instances, device=self.device, dtype=torch.long)

    def __setup(self):
        """
        Setup environment and the simulation scene.
        """
        # allocate buffers
        self.__initialize()
        # get the gymapi backend constant
        if self.config["physics_engine"] == 'physx':
            self._physics_engine = gymapi.SIM_PHYSX
        elif self.config["physics_engine"] == 'flex':
            self._physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.config['physics_engine']}"
            raise ValueError(msg)
        # get device ID for running graphics
        self._graphics_device = -1 if not self.visualize else 0
        # parse parameters
        self._sim_params = self.__parse_sim_params()
        # create sim instance
        self._sim = self._gym.create_sim(0, self._graphics_device, self._physics_engine,
                                         self._sim_params)
        # setup the simulation scene
        self._setup_sim()
        # check creation of sim
        self._gym.prepare_sim(self._sim)
        # create the viewer
        if self.visualize:
            # synchronize rendering and physics?
            self._enable_viewer_sync = True
            # create viewer
            self._viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
            if self._viewer is None:
                print_error("*** Failed to create viewer.")
                sys.exit(1)
            # bind keys for behaviors
            self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_ESCAPE, "QUIT")
            self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_V, "toggle_viewer_sync")
            # set camera pose
            self.set_camera_lookat(pos=(1.0, 1.0, 1.0), target=(0.0, 0.0, 0.0))

# EOF
