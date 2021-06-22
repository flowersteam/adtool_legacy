import sys
from auto_disc import REGISTRATION
from auto_disc import ExperimentPipeline

import numpy as np
import random
import torch

def create(parameters, additional_callbacks):
    seed = parameters['experiment']['seed']
    _set_seed(seed)
    experiment_id = parameters['experiment']['id'] 
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
        callbacks[callback_key].extend(additional_callbacks[callback_key])
        for _callback in parameters['callbacks'][callback_key]:
            callback_class = REGISTRATION['callbacks'][callback_key][_callback['name']]
            callbacks[callback_key].append(
                callback_class(**_callback['config'])
            )
    
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
        on_save_callbacks=callbacks['on_saved']
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
    if len(sys.argv) != 2:
        raise SyntaxError("One single dictionnary parameter must be provided to set the experiment.")

    experiment = create(sys.argv[1])
    start(experiment, sys.argv[1]['nb_iterations'])