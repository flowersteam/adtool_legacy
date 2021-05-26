import sys
from auto_disc import REGISTRATION
from auto_disc import ExperimentPipeline

def create(parameters, additional_callbacks):
    seed = parameters['experiment']['seed']
    experiment_id = parameters['experiment']['id'] 
    checkpoint_id = parameters['experiment']['checkpoint_id']

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

    # TODO : Register callbacks and automatically load them
    on_discovery_callbacks = []
    on_discovery_callbacks.extend(additional_callbacks['on_discovery_callbacks'])
    on_save_finished_callbacks = []
    on_save_finished_callbacks.extend(additional_callbacks['on_save_finished_callbacks'])
    on_finish_callbacks = []
    on_finish_callbacks.extend(additional_callbacks['on_finish_callbacks'])
    on_cancel_callbacks = []
    on_cancel_callbacks.extend(additional_callbacks['on_cancel_callbacks'])
    
    # Create experiment pipeline
    experiment = ExperimentPipeline(
        experiment_id=experiment_id,
        seed=seed,
        checkpoint_id=checkpoint_id,
        system=system,
        explorer=explorer,
        input_wrappers=input_wrappers,
        output_representations=output_representations,
        on_discovery_callbacks=on_discovery_callbacks,
        on_save_finished_callbacks=on_save_finished_callbacks,
        on_finish_callbacks=on_finish_callbacks,
        on_cancel_callbacks=on_cancel_callbacks
    )

    return experiment

def start(experiment, nb_iterations):
    experiment.run(nb_iterations)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SyntaxError("One single dictionnary parameter must be provided to set the experiment.")

    experiment = create(sys.argv[1])
    start(experiment, sys.argv[1]['nb_iterations'])