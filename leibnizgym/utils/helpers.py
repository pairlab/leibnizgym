"""
@author     Mayank Mittal
@email      mittalma@ethz.ch
@brief      General helper functions
"""

# python
import collections.abc
import os
from typing import Union


def get_resources_dir() -> Union[bytes, str]:
    """
    :return: Returns path to the resources directory
    """
    resources_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../resources')
    resources_dir = os.path.abspath(resources_dir)
    # check path exists
    assert os.path.exists(resources_dir), f"Resources directory not found: {resources_dir}"
    # return the path
    return resources_dir


def update_dict(orig_dict: dict, new_dict: collections.abc.Mapping) -> dict:
    """Updates exisitng dictionary with values from a new dictionory.

    This function mimics the dict.update() function. However, it works for
    nested dictionories as well.

    Ref: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    Args:
        orig_dict: The original dictionory to insert items to.
        new_dict: The new dictionory to insert items from.

    Returns:
        The updated dictionory.
    """
    for keyname, value in new_dict.items():
        if isinstance(value, collections.abc.Mapping):
            orig_dict[keyname] = update_dict(orig_dict.get(keyname, {}), value)
        else:
            orig_dict[keyname] = value
    return orig_dict

# EOF
