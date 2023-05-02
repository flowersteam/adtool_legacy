import json
from utils.DB import AppDBMethods


def _get_module_parameters(app_db_caller, module, filter):
    response = app_db_caller("/{}?{}".format(module, filter),
                             AppDBMethods.GET, {}
                             )
    modules = json.loads(response.content)

    parameters = []
    for module in modules:
        parameters.append({
            'name': module['name'],
            'config': module['config']
        })

    return parameters


def reconstruct_parameters(experiment_id, app_db_caller):
    parameters = {}

    # Experiment
    parameters['experiment'] = _get_module_parameters(
        app_db_caller, 'experiments', f'id=eq.{experiment_id}')[0]

    # System
    parameters['system'] = _get_module_parameters(
        app_db_caller, 'systems', f'experiment_id=eq.{experiment_id}')[0]

    # Explorer
    parameters['explorer'] = _get_module_parameters(
        app_db_caller, 'explorers', f'experiment_id=eq.{experiment_id}')[0]

    # Input Wrappers
    parameters['input_wrappers'] = []
    parameters['input_wrappers'].extend(_get_module_parameters(
        app_db_caller, 'input_wrappers', f'experiment_id=eq.{experiment_id}'))

    # Output Representations
    parameters['output_representations'] = []
    parameters['output_representations'].extend(_get_module_parameters(
        app_db_caller, 'output_representations', f'experiment_id=eq.{experiment_id}'))

    parameters['callbacks'] = []
    parameters['loggers'] = []

    return parameters
