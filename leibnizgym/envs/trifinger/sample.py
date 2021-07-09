"""
@author     Mayank Mittal
@email      mittalma@ethz.ch
@brief      Defines sampling stratergies.

# TODO: These functions are generic. Can put in leibnizgym.utils.torch_utils module.
"""

# leibnizgym
from leibnizgym.utils.torch_utils import quaternion_from_euler_xyz
# python
from typing import Union, List, Tuple
import numpy as np
import torch
import torch.nn.functional

"""
Sampling of cuboidal object
"""


@torch.jit.script
def random_xy(num: int, max_com_distance_to_center: float, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns sampled uniform positions in circle (https://stackoverflow.com/a/50746409)"""
    # sample radius of circle
    radius = torch.sqrt(torch.rand(num, dtype=torch.float, device=device))
    radius *= max_com_distance_to_center
    # sample theta of point
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)
    # x,y-position of the cube
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    return x, y


@torch.jit.script
def random_z(num: int, min_height: float, max_height: float, device: str) -> torch.Tensor:
    """Returns sampled height of the goal object."""
    z = torch.rand(num, dtype=torch.float, device=device)
    z = (max_height - min_height) * z + min_height

    return z


@torch.jit.script
def default_orientation(num: int, device: str) -> torch.Tensor:
    """Returns identity rotation transform."""
    quat = torch.zeros((num, 4,), dtype=torch.float, device=device)
    quat[..., -1] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation in 3D as quaternion.
    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html
    """
    # sample random orientation from normal distribution
    quat = torch.randn((num, 4,), dtype=torch.float, device=device)
    # normalize the quaternion
    quat = torch.nn.functional.normalize(quat, p=2., dim=-1, eps=1e-12)

    return quat

@torch.jit.script
def random_angular_vel(num: int, device: str, magnitude_stdev: float) -> torch.Tensor:
    """Samples a random angular velocity with standard deviation `magnitude_stdev`"""

    axis = torch.randn((num, 3,), dtype=torch.float, device=device)
    axis /= torch.norm(axis, p=2, dim=-1).view(-1, 1)
    magnitude = torch.randn((num, 1,), dtype=torch.float, device=device)
    magnitude *= magnitude_stdev
    return magnitude * axis

@torch.jit.script
def random_yaw_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation around z-axis."""
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    return quaternion_from_euler_xyz(roll, pitch, yaw)

# EOF
