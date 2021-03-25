from AutoDiscServer.utils import list_classes
from libs.auto_disc import ExperimentPipeline
from libs.auto_disc.utils.callbacks import CustomPrintCallback
import asyncio

class ExperimentsHandler():
    def __init__(self):
        self._experiments = []

    def add_experiment(self, parameters):
        # TODO: Add entry in DB to obtain the id
        id = 0
        try:
            # Get explorer
            explorer_class = next(filter(lambda dict: dict['name'] == parameters['explorer']['name'], list_classes('libs.auto_disc.explorers')))['class']
            explorer = explorer_class(
                config_kwargs=parameters['explorer']['config']
                )

            # Get system
            system_class = next(filter(lambda dict: dict['name'] == parameters['system']['name'], list_classes('libs.auto_disc.systems')))['class']
            system = system_class(
               config_kwargs=parameters['system']['config'],
               input_space_kwargs=parameters['system']['input_space'], 
               output_space_kwargs=parameters['system']['output_space'], 
               step_output_space_kwargs=parameters['system']['step_output_space']
            )

            # Get input wrappers
            input_wrappers = []
            for _input_wrapper in parameters['input_wrappers']:
                input_wrapper_class = next(filter(lambda dict: dict['name'] == _input_wrapper['name'], list_classes('libs.auto_disc.input_wrappers')))['class']
                input_wrappers.append(
                    input_wrapper_class(
                        config_kwargs=_input_wrapper['config'],
                        input_space_kwargs=_input_wrapper['input_space']
                    )
                )

            # Get output representations
            output_representations = []
            for _output_representation in parameters['input_wrappers']:
                output_representation_class = next(filter(lambda dict: dict['name'] == _output_representation['name'], list_classes('libs.auto_disc.output_representations')))['class']
                output_representations.append(
                    output_representation_class(
                        config_kwargs=_output_representation['config'],
                        outputspace_kwargs=_output_representation['output_space']
                    )
                )

            # Create experiment
            experiment = ExperimentPipeline(
                system=system,
                explorer=explorer,
                input_wrappers=input_wrappers,
                output_representations=output_representations,
                on_exploration_callbacks=[CustomPrintCallback('New discovery for experiment n°{}'.format(id))] # TODO
            )

            ## launch async expe
            task = asyncio.create_task(experiment.run(parameters['nb_iterations']))
            
            ## Add it in the list 
            self._experiments.append({
                "ID": id,
                "experiment": experiment,
                "task": task
            })

            return id
        except Exception as err:
            raise Exception("Error when creating experiment:") from err

    def remove_experiment(self, id):
        # Check if experiment is in the list
        enumerated_experiment_entry = next(filter(lambda element: element[1]["ID"] == id, enumerate(self._experiments)), None)
        if enumerated_experiment_entry is None:
            raise Exception("Unknown experiment ID !")
        
        index = enumerated_experiment_entry[0]

        # Pop from list
        experiment = self._experiments.pop(index)

        # Cancel experiment
        experiment['task'].cancel() # Need to check if cancelled() == True ??

        # Notify the DB
        # TODO           

    def list_running_experiments(self):
        return [expe['ID'] for expe in self._experiments]