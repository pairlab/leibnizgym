"""
@author     Mayank Mittal
@email      mittalma@ethz.ch

@brief      Defines message logging and console output.
"""

# python
import time
import inspect
import os
from termcolor import colored


def __message(level, color, *msg):
    """Appends date-time to message and applies colored formatting."""
    module = inspect.getmodule(inspect.stack()[2][0])
    out = "[%s] [%s]: [%s]: " % (level, time.strftime("%Y.%m.%d::%H-%M-%S"),
                                 os.path.splitext(os.path.basename(module.__file__))[0])
    for sub_msg in msg:
        out += f'{sub_msg[0]}'
    print(colored(out, color))
    # TODO use python.logging to actually log messages and output them to appropriate files


def print_info(*msg):
    """Output an INFO (general-purpose) message."""
    __message("INFO", None, msg)


def print_debug(*msg):
    """Output an DEBUG message. Useful for developers."""
    __message("DEBUG", None, msg)


def print_notify(*msg):
    """Output a NOTIFICATION message. Indicates correct operation, but should capture users attention."""
    __message("NOTIFY", 'blue', msg)


def print_warn(*msg):
    """Output a WARNING message. Indicates possible problem or misbehavior."""
    __message("WARN", 'yellow', msg)


def print_error(*msg):
    """Output an ERROR message."""
    __message("ERROR", 'red', msg)


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionory."""
    if type(val) == dict:
        if not start:
            print('')
        nesting += 4
        for k in val:
            print(nesting * ' ', end='')
            print(k, end=': ')
            print_dict(val[k], nesting, start=False)
    else:
        print(val)

# EOF
