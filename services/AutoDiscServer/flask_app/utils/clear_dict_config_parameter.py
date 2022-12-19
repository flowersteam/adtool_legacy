from copy import deepcopy
from utils.DB.expe_db_utils import SavableOutputs

def clear_dict_config_parameter(experiment_config):
    """
    Brief: Remove all unnecessary entries from dict config to pass them to the run.py
    experiment_config: dict which contains experiment config
    return : dict with the experiment config needed by the run.py to start an experiment
    """
    config = deepcopy(experiment_config)
    del config['experiment']['config']
    del config['experiment']['name']
    # TODO find a way to select savable outputs without the hard coding value of self.experiment_config['callbacks'] in base_experiment
    to_save_outputs_list = [member.name for member in list(SavableOutputs) if member.value in config["callbacks"]["on_discovery"][0]["config"]["to_save_outputs"]]
    config["callbacks"]["on_discovery"][0]["config"]["to_save_outputs"] = to_save_outputs_list
    if 'id' in config['experiment']:
        del config['experiment']['id']
    if 'host' in config['experiment']:
        del config['experiment']['host']
    return config
