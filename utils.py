import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union, cast






def generate_cases(params: Dict[str, Union[List[Any], Any]]) -> List[str]:
    '''
    Generate cases for benchmarking by iterating the parameter values
    '''
    commands = ['']
    for param, values in params.items():
        if isinstance(values, list):
            prev_len = len(commands)
            commands *= len(values)
            dashes = '-' if len(param) == 1 else '--'
            for command_num in range(prev_len):
                for idx, val in enumerate(values):
                    commands[prev_len * idx + command_num] += ' ' + \
                        dashes + param + ' ' + str(val)
        else:
            dashes = '-' if len(param) == 1 else '--'
            for command_num, _ in enumerate(commands):
                commands[command_num] += ' ' + \
                    dashes + param + ' ' + str(values)
    return commands



