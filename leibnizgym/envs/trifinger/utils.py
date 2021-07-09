"""
@author     Mayank Mittal
@email      mittalma@ethz.ch
@brief      Dimensions for Trifinger robot.
"""

# python
import enum
from typing import Tuple, Union
import numpy as np

# ################### #
# Dimensions of robot #
# ################### #


class TrifingerDimensions(enum.Enum):
    """
    Dimensions of the tri-finger robot.

    Note: While it may not seem necessary for tri-finger robot since it is fixed base, for floating
    base systems having this dimensions class is useful.
    """
    # general state
    # cartesian position + quaternion orientation
    PoseDim = 7,
    # linear velocity + angular velcoity
    VelocityDim = 6
    # state: pose + velocity
    StateDim = 13
    # force + torque
    WrenchDim = 6
    # for robot
    # number of fingers
    NumFingers = 3
    # for three fingers
    JointPositionDim = 9
    JointVelocityDim = 9
    JointTorqueDim = 9
    # generalized coordinates
    GeneralizedCoordinatesDim = JointPositionDim
    GeneralizedVelocityDim = JointVelocityDim
    # for objects
    ObjectPoseDim = 7
    ObjectVelocityDim = 6


# ################# #
# Different objects #
# ################# #


# readius of the area
ARENA_RADIUS = 0.195


class CuboidalObject:
    """
    Fields for a cuboidal object.

    @note Motivation for this class is that if domain randomization is performed over the
          size of the cuboid, then its attributes are automatically updated as well.
    """
    # 3D radius of the cuboid
    radius_3d: float
    # distance from wall to the center
    max_com_distance_to_center: float
    # minimum and mximum height for spawning the object
    min_height: float
    max_height = 0.1

    def __init__(self, size: Union[float, Tuple[float, float, float]]):
        """Initialize the cuboidal object.

        Args:
            size: The size of the object along x, y, z in meters. If a single float is provided, then it is assumed that
                  object is a cube.
        """
        # decide the size depedning on input type
        if isinstance(size, float):
            self._size = (size, size, size)
        else:
            self._size = size
        # compute remaning attributes
        self.__compute()

    """
    Properties
    """

    @property
    def size(self) -> Tuple[float, float, float]:
        """
        Returns the dimensions of the cuboid object (x, y, z) in meters.
        """
        return self._size

    """
    Configurations
    """

    @size.setter
    def size(self, size: Union[float, Tuple[float, float, float]]):
        """ Set size of the object.

        Args:
            size: The size of the object along x, y, z in meters. If a single float is provided, then it is assumed
                  that object is a cube.
        """
        # decide the size depedning on input type
        if isinstance(size, float):
            self._size = (size, size, size)
        else:
            self._size = size
        # compute attributes
        self.__compute()

    """
    Private members
    """

    def __compute(self):
        """Compute the attributes for the object.
        """
        # compute 3D radius of the cuboid
        max_len = max(self._size)
        self.radius_3d = max_len * np.sqrt(3) / 2
        # compute distance from wall to the center
        self.max_com_distance_to_center = ARENA_RADIUS - self.radius_3d
        # minimum height for spawning the object
        self.min_height = self._size[2] / 2

# EOF
