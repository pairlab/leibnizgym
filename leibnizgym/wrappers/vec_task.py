"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

@author     Arthur Allshire
@email      arthur.allshire@mail.utoronto.ca
@author     Mayank Mittal
@email      mittalma@ethz.ch

@brief Defines vec-environment wrapper for environments in isaacgym simulator.
"""

# leibnizgym
from leibnizgym.envs.env_base import IsaacEnvBase
# python
from gym import spaces
from typing import Tuple
import torch
import numpy as np


class VecTask:
    """
    A wrapper around a base environment for RL training using IsaacGym simulator..

    TODO (@mayank): All wrappers should still inherit from env_base class to ensure consistency!
                    Since we don't support C++ environments, the VecTask and VecTaskPython can be merged.
    """

    def __init__(self, task: IsaacEnvBase, rl_device: str, clip_obs: float = 5.0, clip_actions: float = 1.0):
        """Initialize the wrapper for RL training.

        Args:
            task: An instance of the IsaacEnvBase class to wrap around.
            rl_device: The device on which RL agent is present.
            clip_obs: Clipping for the observations.
            clip_actions: Clipping for the actions.
        """
        # check instance of input
        assert isinstance(task, IsaacEnvBase)
        # copy inputs to class members
        self._task = task
        self._clip_obs = float(clip_obs)
        self._clip_actions = float(clip_actions)
        self._rl_device = rl_device
        # set gym spaces for the environment
        self._obs_space = spaces.Box(np.full(self.num_obs, -self._clip_obs),
                                     np.full(self.num_obs, self._clip_obs))
        self._state_space = spaces.Box(np.full(self.num_states, -self._clip_obs),
                                       np.full(self.num_states, self._clip_obs))
        self._act_space = spaces.Box(np.full(self.num_actions, -self._clip_actions),
                                     np.full(self.num_actions, self._clip_actions))

    def __str__(self) -> str:
        msg = f"Vectorized Environment around task: {type(self._task).__name__} \n" \
              f"\t Number of instances   : {self.num_envs} \n" \
              f"\t Number of observations: {self.num_obs} \n" \
              f"\t Number of states      : {self.num_states} \n" \
              f"\t Number of actions     : {self.num_actions} \n" \
              f"\t Observation clipping  : {self._clip_obs} \n" \
              f"\t Actions clipping      : {self._clip_actions} \n"
        return msg

    """
    Properties
    """

    def get_number_of_agents(self) -> int:
        """Returns number of agents in the environment (used for multi-agent environments)"""
        if hasattr(self._task, 'get_number_of_agents'):
            return self._task.get_number_of_agents()
        else:
            return 1

    @property
    def num_envs(self) -> int:
        return self._task.get_num_instances()

    @property
    def num_states(self) -> int:
        return self._task.get_state_dim()

    @property
    def num_obs(self) -> int:
        return self._task.get_obs_dim()

    @property
    def num_actions(self) -> int:
        return self._task.get_action_dim()

    @property
    def observation_space(self) -> spaces.Box:
        return self._obs_space

    @property
    def state_space(self) -> spaces.Box:
        return self._state_space

    @property
    def action_space(self) -> spaces.Box:
        return self._act_space

    """
    Operations - Implementation specfic.
    """

    def dump_config(self, filename: str):
        """Dumps the environment configuration into a YAML file.

        Args:
            filename: The path to the file to save configuration into.
        """
        self._task.dump_config(filename)

    def reset(self) -> torch.Tensor:
        raise NotImplementedError

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        raise NotImplementedError


class VecTaskPython(VecTask):
    """
    Wrapper for Python CPU/GPU environment on IsaacGym simulator.
    """

    def __init__(self, task: IsaacEnvBase, rl_device: str, clip_obs: float = 5.0, clip_actions: float = 1.0):
        """Initialize the wrapper for RL training.

        Args:
            task: An instance of the IsaacEnvBase class to wrap around.
            rl_device: The device on which RL agent is present.
            clip_obs: Clipping for the observations.
            clip_actions: Clipping for the actions.
        """
        super().__init__(task, rl_device, clip_obs, clip_actions)

    """
    Properties
    """

    def get_state(self) -> torch.Tensor:
        return torch.clamp(self._task.states_buf, -self._clip_obs, self._clip_obs).to(self._rl_device)

    """
    Operations - Implementation specfic.
    """

    def reset(self) -> torch.Tensor:
        obs = self._task.reset()
        return torch.clamp(obs, -self._clip_obs, self._clip_obs).to(self._rl_device)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # render the GUI
        if self._task.visualize:
            self._task.render()
        # clip input actions
        actions_tensor = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        # take step into environment
        obs, rew, is_done, info = self._task.step(actions_tensor)
        # copy tensors to RL device
        obs = torch.clamp(obs, -self._clip_obs, self._clip_obs).to(self._rl_device)
        rew = rew.to(self._rl_device)
        is_done = is_done.to(self._rl_device)
        # return
        return obs, rew, is_done, info

# EOF
