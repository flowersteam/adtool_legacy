#region import
import os 
import sys
from copy import deepcopy
from unittest import mock
import pytest
import pytest_mock
from requests import patch

classToTestFolderPath = os.path.abspath(__file__)
classToTestFolderPath = classToTestFolderPath.split('/')
classToTestFolderPath = classToTestFolderPath[0:classToTestFolderPath.index("AutomatedDiscoveryTool")+1]
AutoDiscServerPath = "/".join(classToTestFolderPath) + "/services/AutoDiscServer"
auto_discFolderPath = "/".join(classToTestFolderPath) + "/libs/auto_disc"

sys.path.insert(0, os.path.dirname(auto_discFolderPath))
sys.path.insert(0, os.path.dirname(AutoDiscServerPath))

from AutoDiscServer.experiments import BaseExperiment
#endregion

#region experiment example
id = 14
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
on_progress_callback = lambda id, current_progress:print("on_progress_callback : id {}, progress {}".format(id, current_progress))
on_checkpoint_needed_callback = lambda id, x : 0
on_checkpoint_finished_callback = lambda id, current_checkpoint_id :print("on_checkpoint_finished_callback")
on_checkpoint_update_callback = lambda current_checkpoint_id, error :print("on_checkpoint_update_callback")
on_experiment_update_callback = lambda id, status :print("on_experiment_update_callback")
#endregion

#region mock
mocked___initialize_checkpoint_history = lambda : None
mocked__clear_dict_config_parameter = lambda x : None
Object = lambda **kwargs: type("Object", (), kwargs)()
mocked__AppDBCaller_call = lambda self, x, y : Object(content="[]".encode())
#endregion

#region test __init__
@mock.patch("AutoDiscServer.experiments.BaseExperiment._initialize_checkpoint_history", side_effect=mocked___initialize_checkpoint_history)
@mock.patch("AutoDiscServer.experiments.base_experiment.clear_dict_config_parameter", side_effect=mocked__clear_dict_config_parameter)
def test__init__(mocker_initialize_checkpoint_history, mocker_clear_dict_config_parameter):
    """ Test BaseExperiment.__init__ behavior

        Parameters
        -----------
            mocker : pytest mocker to replace the function calls made in the function being tested

        Return
        -------
    """
    ## init
    # do in experiment example region
    experiement_ref = deepcopy(experiment_config)

    ## exec
    baseExperiment = BaseExperiment(
        id, 
        deepcopy(experiment_config), 
        deepcopy(on_progress_callback),
        deepcopy(on_checkpoint_needed_callback),
        deepcopy(on_checkpoint_finished_callback),
        deepcopy(on_checkpoint_update_callback),
        deepcopy(on_experiment_update_callback)
    )
    ## check
    assert baseExperiment.id == 14

    experiement_ref['experiment']['id'] = id
    experiement_ref['experiment']['save_frequency'] = experiement_ref['experiment']['config']['save_frequency']
    del experiement_ref['experiment']['config']['save_frequency']
    del experiement_ref["callbacks"]
    del baseExperiment.experiment_config["callbacks"]
    assert baseExperiment.experiment_config == experiement_ref

#endregion

#region test _initialize_checkpoint_history

@mock.patch("AutoDiscServer.experiments.base_experiment.AppDBCaller.__call__", side_effect=mocked__AppDBCaller_call)
@mock.patch("AutoDiscServer.experiments.base_experiment.clear_dict_config_parameter", side_effect=mocked__clear_dict_config_parameter)
def test__initialize_checkpoint_history(mocker_AppDBCaller, mocker_clear_dict_config_parameter):
    ## exec
    baseExperiment = BaseExperiment(
        id, 
        deepcopy(experiment_config), 
        deepcopy(on_progress_callback),
        deepcopy(on_checkpoint_needed_callback),
        deepcopy(on_checkpoint_finished_callback),
        deepcopy(on_checkpoint_update_callback),
        deepcopy(on_experiment_update_callback)
    )

    ## check
    assert baseExperiment.progresses == {0:0}
    assert baseExperiment.current_progress == 0
    assert baseExperiment.checkpoints_history == {0: {'seeds_status': {}, 'parent_id': None}}

#endregion

#region test on_progress
@mock.patch("AutoDiscServer.experiments.base_experiment.AppDBCaller.__call__", side_effect=mocked__AppDBCaller_call)
@mock.patch("AutoDiscServer.experiments.base_experiment.clear_dict_config_parameter", side_effect=mocked__clear_dict_config_parameter)
def test_on_progress(mocker_AppDBCaller, mocker_clear_dict_config_parameter):
    ## init
    baseExperiment = BaseExperiment(
        id, 
        deepcopy(experiment_config), 
        deepcopy(on_progress_callback),
        deepcopy(on_checkpoint_needed_callback),
        deepcopy(on_checkpoint_finished_callback),
        deepcopy(on_checkpoint_update_callback),
        deepcopy(on_experiment_update_callback)
    )
    # Init without _initialize_checkpoint_history
    baseExperiment.progresses = dict(zip([i for i in range(baseExperiment.experiment_config['experiment']['config']["nb_seeds"])], 
                                   [0 for _ in range(baseExperiment.experiment_config['experiment']['config']["nb_seeds"])]))
    baseExperiment.current_progress = 0
    baseExperiment.checkpoints_history = {}
    
    ## exec
    baseExperiment.on_progress(0)

    ## check
    assert baseExperiment.progresses[0] == 1
    assert baseExperiment.current_progress == 1

#endregion

#region on_save

@mock.patch("AutoDiscServer.experiments.base_experiment.AppDBCaller.__call__", side_effect=mocked__AppDBCaller_call)
@mock.patch("AutoDiscServer.experiments.base_experiment.clear_dict_config_parameter", side_effect=mocked__clear_dict_config_parameter)
def test_on_save(mocker_AppDBCaller, mocker_clear_dict_config_parameter):
    ## init
    baseExperiment = BaseExperiment(
        id, 
        deepcopy(experiment_config), 
        deepcopy(on_progress_callback),
        deepcopy(on_checkpoint_needed_callback),
        deepcopy(on_checkpoint_finished_callback),
        deepcopy(on_checkpoint_update_callback),
        deepcopy(on_experiment_update_callback)
    )

    ## exec
    baseExperiment.on_save(0, 0)
    ## check
    assert baseExperiment.checkpoints_history[0]["seeds_status"][0] == 0


#endregion

#region test on_error

@mock.patch("AutoDiscServer.experiments.base_experiment.AppDBCaller.__call__", side_effect=mocked__AppDBCaller_call)
@mock.patch("AutoDiscServer.experiments.base_experiment.clear_dict_config_parameter", side_effect=mocked__clear_dict_config_parameter)
def test_on_error(mocker_AppDBCaller, mocker_clear_dict_config_parameter):
    ## init
    baseExperiment = BaseExperiment(
        id, 
        deepcopy(experiment_config), 
        deepcopy(on_progress_callback),
        deepcopy(on_checkpoint_needed_callback),
        deepcopy(on_checkpoint_finished_callback),
        deepcopy(on_checkpoint_update_callback),
        deepcopy(on_experiment_update_callback)
    )
    seed = 0
    current_checkpoint_id = 0
    ## exec
    baseExperiment.on_error(seed, current_checkpoint_id)
    ## check
    assert baseExperiment.progresses == {}
    assert baseExperiment.checkpoints_history[current_checkpoint_id]["seeds_status"][seed] == 3

#endregion

#region test on_finished

@mock.patch("AutoDiscServer.experiments.base_experiment.AppDBCaller.__call__", side_effect=mocked__AppDBCaller_call)
@mock.patch("AutoDiscServer.experiments.base_experiment.clear_dict_config_parameter", side_effect=mocked__clear_dict_config_parameter)
def test_on_finished(mocker_AppDBCaller, mocker_clear_dict_config_parameter):
    ## init
    baseExperiment = BaseExperiment(
        id, 
        deepcopy(experiment_config), 
        deepcopy(on_progress_callback),
        deepcopy(on_checkpoint_needed_callback),
        deepcopy(on_checkpoint_finished_callback),
        deepcopy(on_checkpoint_update_callback),
        deepcopy(on_experiment_update_callback)
    )
    seed = 0
    ## exec
    baseExperiment.on_finished(seed)
    ## check
    assert baseExperiment.progresses == {}


#endregion


#region test on_cancelled

@mock.patch("AutoDiscServer.experiments.base_experiment.AppDBCaller.__call__", side_effect=mocked__AppDBCaller_call)
@mock.patch("AutoDiscServer.experiments.base_experiment.clear_dict_config_parameter", side_effect=mocked__clear_dict_config_parameter)
def test_on_cancelled(mocker_AppDBCaller, mocker_clear_dict_config_parameter):
    ## init
    baseExperiment = BaseExperiment(
        id, 
        deepcopy(experiment_config), 
        deepcopy(on_progress_callback),
        deepcopy(on_checkpoint_needed_callback),
        deepcopy(on_checkpoint_finished_callback),
        deepcopy(on_checkpoint_update_callback),
        deepcopy(on_experiment_update_callback)
    )
    seed = 0
    ## exec
    baseExperiment.on_cancelled(seed)
    ## check
    assert baseExperiment.progresses == {}


#endregion


#region test _get_current_checkpoint_id

@mock.patch("AutoDiscServer.experiments.base_experiment.AppDBCaller.__call__", side_effect=mocked__AppDBCaller_call)
@mock.patch("AutoDiscServer.experiments.base_experiment.clear_dict_config_parameter", side_effect=mocked__clear_dict_config_parameter)
def test_get_current_checkpoint_id(mocker_AppDBCaller, mocker_clear_dict_config_parameter):
    ## init
    baseExperiment = BaseExperiment(
        id, 
        deepcopy(experiment_config), 
        deepcopy(on_progress_callback),
        deepcopy(on_checkpoint_needed_callback),
        deepcopy(on_checkpoint_finished_callback),
        deepcopy(on_checkpoint_update_callback),
        deepcopy(on_experiment_update_callback)
    )
    seed = 0
    ## exec
    current_checkpoint_id = baseExperiment._get_current_checkpoint_id(seed)
    ## check
    assert current_checkpoint_id == 0


#endregion

#region test callback_to_all_running_seeds

@mock.patch("AutoDiscServer.experiments.base_experiment.AppDBCaller.__call__", side_effect=mocked__AppDBCaller_call)
@mock.patch("AutoDiscServer.experiments.base_experiment.clear_dict_config_parameter", side_effect=mocked__clear_dict_config_parameter)
def test_callback_to_all_running_seeds(mocker_AppDBCaller, mocker_clear_dict_config_parameter):
    ## init
    baseExperiment = BaseExperiment(
        id, 
        deepcopy(experiment_config), 
        deepcopy(on_progress_callback),
        deepcopy(on_checkpoint_needed_callback),
        deepcopy(on_checkpoint_finished_callback),
        deepcopy(on_checkpoint_update_callback),
        deepcopy(on_experiment_update_callback)
    )
    seed = 0
    current_checkpoint_id = 0
    ## exec
    baseExperiment.callback_to_all_running_seeds(lambda seed, current_checkpoint_id : baseExperiment.on_error(seed,current_checkpoint_id ))
    ## check
    assert baseExperiment.checkpoints_history == {0: {'seeds_status' : {0:3}, 'parent_id':None}}


#endregion