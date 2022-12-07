from torch import Tensor
from enum import Enum
import json

def serialize_autodisc_space(space):
    """
    brief:  transform space into serializable object for json (tensor to list)
    param:  space: one of possible outputs we want save
    """
    serialized_space = {}
    if isinstance(space, Tensor):
        serialized_space = space.tolist()
    elif isinstance(space, list):
        for i in range(len(space)):
            space[i] = serialize_autodisc_space(space[i])
        serialized_space = space
    elif isinstance(space, dict):
        for key in space:
            serialized_space[key] = serialize_autodisc_space(space[key])
    else:
        serialized_space = space
    return serialized_space

def is_json_serializable(object):
    try:
        json.dumps(object)
        return True
    except (TypeError, OverflowError):
        return False

class SavableOutputs(Enum):
    raw_run_parameters = "Parameters sent by the explorer before input wrappers"
    run_parameters = "Parameters sent by the explorer after input wrappers"
    raw_output = "Raw system output"
    output = "Representation of system output"
    rendered_output = "Rendered system output"
