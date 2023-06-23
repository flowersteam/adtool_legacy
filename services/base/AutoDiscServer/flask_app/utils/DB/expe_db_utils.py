import json
from enum import Enum


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
