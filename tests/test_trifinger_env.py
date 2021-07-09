#!/usr/bin/env python3

# leibnizgym
from leibnizgym.utils import *
from leibnizgym.envs import TrifingerEnv
# python
import torch
import unittest


class TestTrifingerEnv(unittest.TestCase):
    """Test the Trifinger gym environment."""

    """
    Reset environment tests
    """

    def test_default_reset(self):
        # configure the environment
        env_config = {
            'num_instances': 1,
            'control_decimation': 5,
            'command_mode': 'torque',
            'sim': {
                "use_gpu_pipeline": False,
                "physx": {
                    "use_gpu": False
                }
            },
            # Configuration for resetting the MDP
            "reset_distribution": {
                # Defines how to reset the robot joint state
                "robot_initial_state": {
                    "type": "default",
                },
                # Defines how to reset the robot joint state
                "object_initial_state": {
                    "type": "default",
                }
            }
        }
        # create environment
        env = TrifingerEnv(config=env_config, device='cpu', verbose=True, visualize=True)

        # check reset
        for step_num in range(3000):
            # reset every certain number of steps
            if step_num % 100 == 0:
                _ = env.reset()
            # render the env
            env.render()

    def test_random_reset(self):
        # configure the environment
        env_config = {
            'num_instances': 1,
            'control_decimation': 5,
            'command_mode': 'torque',
            'sim': {
                "use_gpu_pipeline": False,
                "physx": {
                    "use_gpu": False
                }
            },
            # Configuration for resetting the MDP
            "reset_distribution": {
                # Defines how to reset the robot joint state
                "robot_initial_state": {
                    "type": "random",
                    "dof_pos_scale": 0.4,
                    "dof_vel_scale": 0.05
                },
                # Defines how to reset the robot joint state
                "object_initial_state": {
                    "type": "random",
                }
            }
        }
        # create environment
        env = TrifingerEnv(config=env_config, device='cpu', verbose=True, visualize=True)

        # check reset
        for step_num in range(3000):
            # reset every certain number of steps
            if step_num % 100 == 0:
                _ = env.reset()
            # render the env
            env.render()

    """
    Step environment test
    """

    def test_zero_action_agent(self):
        # configure the environment
        env_config = {
            'num_instances': 1,
            'control_decimation': 5,
            'command_mode': 'torque',
            'sim': {
                "use_gpu_pipeline": False,
                "physx": {
                    "use_gpu": False
                }
            }
        }
        # create environment
        env = TrifingerEnv(config=env_config, device='cpu', verbose=True, visualize=True)

        # check reset
        for step_num in range(3000):
            # reset every certain number of steps
            if step_num % 100 == 0:
                _ = env.reset()
            else:
                # random action agent
                action = torch.zeros(env.get_action_shape(), dtype=torch.float, device=env.device)
                # step through physics
                _, _, _, _ = env.step(action)
            # render the env
            env.render()

    def test_random_agent(self):
        # configure the environment
        env_config = {
            'num_instances': 1,
            'control_decimation': 5,
            'command_mode': 'torque',
            'sim': {
                "use_gpu_pipeline": False,
                "physx": {
                    "use_gpu": False
                }
            }
        }
        # create environment
        env = TrifingerEnv(config=env_config, device='cpu', verbose=True, visualize=True)

        # check reset
        for step_num in range(3000):
            # reset every certain number of steps
            if step_num % 100 == 0:
                _ = env.reset()
            else:
                # random action agent
                action = 2 * torch.rand(env.get_action_shape(), dtype=torch.float, device=env.device) - 1
                # step through physics
                _, _, _, _ = env.step(action)
            # render the env
            env.render()

    """
    Asymmetric observations environment test
    """

    def test_asymm_zero_action_agent(self):
        # configure the environment
        env_config = {
            'num_instances': 1,
            "asymmetric_obs": True,
            'control_decimation': 5,
            'command_mode': 'torque',
            'sim': {
                "use_gpu_pipeline": False,
                "physx": {
                    "use_gpu": False
                }
            }
        }
        # create environment
        env = TrifingerEnv(config=env_config, device='cpu', verbose=True, visualize=True)

        # check reset
        for step_num in range(3000):
            # reset every certain number of steps
            if step_num % 100 == 0:
                _ = env.reset()
            else:
                # random action agent
                action = torch.zeros(env.get_action_shape(), dtype=torch.float, device=env.device)
                # step through physics
                _, _, _, _ = env.step(action)
            # render the env
            env.render()


if __name__ == "__main__":
    unittest.main()
