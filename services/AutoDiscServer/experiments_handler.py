from AutoDiscServer.utils import list_classes
from auto_disc import ExperimentPipeline
from auto_disc.utils.callbacks import CustomPrintCallback
import threading

class ExperimentsHandler():
    def __init__(self):
        self._experiments = []
        self._id = 0

    def add_experiment(self, parameters):
        # TODO: Add entry in DB to obtain the id
        id = self._id
        self._id += 1
        try:
            # Get explorer
            explorer_class = next(filter(lambda dict: dict['name'] == parameters['explorer']['name'], list_classes('auto_disc.explorers')))['class']
            explorer = explorer_class(**parameters['explorer']['parameters'])

            # Get system
            system_class = next(filter(lambda dict: dict['name'] == parameters['system']['name'], list_classes('auto_disc.systems')))['class']
            system = system_class(**parameters['system']['parameters'])

            # Get input wrappers
            input_wrappers = []
            for _input_wrapper in parameters['input_wrappers']:
                input_wrapper_class = next(filter(lambda dict: dict['name'] == _input_wrapper['name'], list_classes('auto_disc.input_wrappers')))['class']
                input_wrappers.append(
                    input_wrapper_class(**_input_wrapper['parameters'])
                )

            # Get output representations
            output_representations = []
            for _output_representation in parameters['output_representations']:
                output_representation_class = next(filter(lambda dict: dict['name'] == _output_representation['name'], list_classes('auto_disc.output_representations')))['class']
                output_representations.append(
                    output_representation_class(**_output_representation['parameters'])
                )

            # Create experiment
            experiment = ExperimentPipeline(
                system=system,
                explorer=explorer,
                input_wrappers=input_wrappers,
                output_representations=output_representations,
                on_exploration_callbacks=[CustomPrintCallback('New discovery for experiment n°{}'.format(id))], # TODO
                on_finish_callbacks=[CustomPrintCallback('Experiment n°{} finished !'.format(id))], # TODO
                on_cancel_callbacks=[CustomPrintCallback('Experiment n°{} cancelled !'.format(id))], # TODO
            )

            ## launch async expe
            # coroutine = experiment.run
            
            task = threading.Thread(target=experiment.run, args=(parameters['nb_iterations'], ))
            task.start()
            
            ## Add it in the list 
            self._experiments.append({
                "ID": id,
                "experiment_object": experiment,
                "task": task,
            })

            return id
        except Exception as err:
            message = "Error when creating experiment:" + str(err)
            raise Exception(message)

    def remove_experiment(self, id):
        # Check if experiment is in the list
        enumerated_experiment_entry = next(filter(lambda element: element[1]["ID"] == id, enumerate(self._experiments)), None)
        if enumerated_experiment_entry is None:
            raise Exception("Unknown experiment ID !")
        
        index = enumerated_experiment_entry[0]

        # Pop from list
        experiment = self._experiments.pop(index)

        # Cancel experiment
        experiment['experiment_object'].cancellation_token.trigger()
        experiment['task'].join() # Need to check if cancelled() == True ??

        # Notify the DB
        # TODO           

    def list_running_experiments(self):
        return [expe['ID'] for expe in self._experiments]