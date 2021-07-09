"""
@author     Mayank Mittal
@email      mittalma@ethz.ch

@brief  Custom errors for simulation.
"""


class InvalidTaskNameError(Exception):
    """
    Error for invalid task that has not been implemented.
    """

    def __init__(self, task_name):
        """ Checks if a given task name exists or not in leibizgym.

        Args:
            task_name: Name of the task.
        """
        # valid task names
        valid_tasks = ['Trifinger']
        # create message
        msg = f"Unrecognized task: `{task_name}`. Task should be in: {valid_tasks}"
        super().__init__(msg)

# EOF
