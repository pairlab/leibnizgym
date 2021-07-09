"""
@author     Mayank Mittal
@email      mittalma@ethz.ch
@brief      Custom datatypes to use.
"""

# python
import torch


class RewardTerm(torch.nn.Module):
    """
    Class for defining various reward terms.
    """
    name: str = ''
    weight: float = 0.0

    def __init__(self, name: str, activate: bool, weight: float, **kwargs):
        """Initialize the members of the reward term.

        Args:
            name: Name of the reward term.
            activate: Whether to use the reward term or not.
            weight: Weight of the reward term.
            **kwargs: Miscellaneous arguments for the reward.
        """
        super(RewardTerm, self).__init__()
        self.name = name
        self.activate = activate
        self.weight = weight

    def __str__(self) -> str:
        if self.activate:
            return f"Reward name: {self.name}, enable: {self.activate}, weight: {self.weight}"
        else:
            return f"Reward name: {self.name}, enable: {self.activate}"

    """
    Operations
    """

    def compute(self, *args, **kwargs) -> torch.Tensor:
        """
        Function to compute the reward term.

        Args:
            *args: Inputs to the computation function.
            **kwargs: Miscellaneous inputs to the computation function.

        Returns:
            The computed reward term.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Function to compute the reward term.

        Args:
            *args: Inputs to the computation function.
            **kwargs: Miscellaneous inputs to the computation function.

        Returns:
            The computed reward term.
        """
        return self.compute(args, kwargs)

# EOF
