"""
@brief      Demo script for checking tri-finger environment.
"""

# leibnizgym
from leibnizgym.utils import *
from leibnizgym.envs import TrifingerEnv
# python
import torch

if __name__ == '__main__':
    # configure the environment
    env_config = {
        'num_instances': 8192,
        'aggregrate_mode': True,
        'control_decimation': 1,
        'command_mode': 'torque',
        'sim': {
            "use_gpu_pipeline": True,
            "physx": {
                "use_gpu": False
            }
        }
    }
    # create environment
    env = TrifingerEnv(config=env_config, device='cuda:0', verbose=True, visualize=False)
    _ = env.reset()
    print_info("Trifinger environment creation successful.")

    # sample run
    while True:
        # zero action agent
        action = 2 * torch.rand(env.get_action_shape(), dtype=torch.float, device=env.device) - 1
        # step through physics
        _, _, _, _ = env.step(action)
        # render environment
        env.render()

# EOF
