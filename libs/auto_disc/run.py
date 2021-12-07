import sys
import argparse
import json
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../"))
from auto_disc import REGISTRATION
from auto_disc import ExperimentPipeline
from auto_disc.utils.logger import AutoDiscLogg

import numpy as np
import random
import torch

def create(parameters, experiment_id, seed, additional_callbacks = None):
    _set_seed(seed)
    checkpoint_id = parameters['experiment']['checkpoint_id']
    save_frequency = parameters['experiment']['save_frequency']

    # Get explorer
    explorer_class = REGISTRATION['explorers'][parameters['explorer']['name']]
    explorer = explorer_class(**parameters['explorer']['config'])

    # Get system
    system_class = REGISTRATION['systems'][parameters['system']['name']]
    system = system_class(**parameters['system']['config'])

    # Get input wrappers
    input_wrappers = []
    for _input_wrapper in parameters['input_wrappers']:
        input_wrapper_class = REGISTRATION['input_wrappers'][_input_wrapper['name']]
        input_wrappers.append(
            input_wrapper_class(**_input_wrapper['config'])
        )

    # Get output representations
    output_representations = []
    for _output_representation in parameters['output_representations']:
        output_representation_class = REGISTRATION['output_representations'][_output_representation['name']]
        output_representations.append(
            output_representation_class(**_output_representation['config'])
        )

    # Get callbacks
    callbacks = {
        'on_discovery': [],
        'on_save_finished': [],
        'on_finished': [],
        'on_error': [],
        'on_cancelled': [],
        'on_saved':[]
    }

    for callback_key in callbacks:
        if(additional_callbacks):
            callbacks[callback_key].extend(additional_callbacks[callback_key])
        for _callback in parameters['callbacks'][callback_key]:
            callback_class = REGISTRATION['callbacks'][callback_key][_callback['name']]
            callbacks[callback_key].append(
                callback_class(**_callback['config'])
            )
    
    # Get logger
    logger_key = parameters['logger']['name']
    logger_class = REGISTRATION['logger'][logger_key]
    logger = logger_class(**parameters['logger']['config'], experiment_id=experiment_id)

    logger = AutoDiscLogg(seed, checkpoint_id, logger)

    # Create experiment pipeline
    experiment = ExperimentPipeline(
        experiment_id=experiment_id,
        seed=seed,
        checkpoint_id=checkpoint_id,
        save_frequency=save_frequency,
        system=system,
        explorer=explorer,
        input_wrappers=input_wrappers,
        output_representations=output_representations,
        on_discovery_callbacks=callbacks['on_discovery'],
        on_save_finished_callbacks=callbacks['on_save_finished'],
        on_finished_callbacks=callbacks['on_finished'],
        on_cancelled_callbacks=callbacks['on_cancelled'],
        on_save_callbacks=callbacks['on_saved'],
        on_error_callbacks=callbacks['on_error'],
        logger = logger
    )

    return experiment

def start(experiment, nb_iterations):
    experiment.run(nb_iterations)

def _set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--experiment_id', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--nb_iterations', type=int, required=True)
    
    args = parser.parse_args()
    
    with open(args.config_file) as json_file:
        config = json.load(json_file)
        
    experiment = create(config, args.experiment_id, args.seed)
    start(experiment, args.nb_iterations)  