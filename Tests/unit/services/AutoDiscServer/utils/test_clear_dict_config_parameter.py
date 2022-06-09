#region import
import os 
import sys
from copy import deepcopy
from unittest import mock

classToTestFolderPath = os.path.abspath(__file__)
classToTestFolderPath = classToTestFolderPath.split('/')
classToTestFolderPath = classToTestFolderPath[0:classToTestFolderPath.index("AutomatedDiscoveryTool")+1]
AutoDiscServerPath = "/".join(classToTestFolderPath) + "/services/AutoDiscServer"

sys.path.insert(0, os.path.dirname(AutoDiscServerPath))

from AutoDiscServer.utils import clear_dict_config_parameter
#endregion

#region test clear_dict_config_parameter

def test_clear_dict_config_parameter():
    ## init
    experiment_config = {
        "experiment": {
            "name": "create-test-local",
            "config": {
                "host": "local",
                "nb_seeds": 1, 
                "nb_iterations": 4, 
                "save_frequency": 1, 
                "discovery_saving_keys": [
                    "Parameters sent by the explorer before input wrappers",
                    "Parameters sent by the explorer after input wrappers",
                    "Raw system output", "Representation of system output",
                    "Rendered system output"
                ]
            }
        }, 
        "system": {
            "name": "PythonLenia",
            "config": {
                "SX": 256,
                "SY": 256,
                "version": "pytorch_fft", 
                "mon_dict": {}, 
                "final_step": 200, 
                "scale_init_state": 1, 
                "mon_decimal_parameter": 5.2
            }
        }, 
        "explorer": {
            "name": "IMGEPExplorer", 
            "config": {
                "goal_selection_type": "random", 
                "use_exandable_goal_space": True, 
                "num_of_random_initialization": 10, 
                "source_policy_selection_type": "optimal"
            }
        }, 
        "input_wrappers": [
            {
                "name": "generic.CPPN", 
                "config": {
                    "n_passes": 2, 
                    "wrapped_output_space_key": "init_state"
                }, 
                "index": 0
            }
        ], 
        "output_representations": [
            {
                "name": "specific.LeniaFlattenImage", 
                "config": {
                    "SX": 256, 
                    "SY": 256, 
                    "distance_function": "L2", 
                    "wrapped_input_space_key": "states"
                }, 
                "index": 0
            }
        ], 
        "callbacks": [], 
        "logger_handlers": []
    }
    experiment_config['callbacks'] = {
                'on_discovery': [
                    {
                        'name' : 'base',
                        'config': {
                            'to_save_outputs': experiment_config['experiment']['config']['discovery_saving_keys']
                        }
                    }
                ],
                'on_save_finished': [
                    {
                        'name' : 'base',
                        'config': {}
                    }
                ],
                'on_cancelled': [
                    {
                        'name' : 'base',
                        'config': {}
                    }
                ],
                'on_finished': [
                    {
                        'name' : 'base',
                        'config': {}
                    }
                ],
                'on_error': [
                    {
                        'name' : 'base',
                        'config': {}
                    }
                ],
                'on_saved': [
                    {
                        'name' : 'base',
                        'config': {}
                    }
                ],
            }
    ## exec
    cleared_config = clear_dict_config_parameter(deepcopy(experiment_config))
    ## check
    assert cleared_config["experiment"] == {}
    assert cleared_config["callbacks"]["on_discovery"][0]["config"]["to_save_outputs"]== ['raw_run_parameters', 'run_parameters', 'raw_output', 'output', 'rendered_output']

#endregion