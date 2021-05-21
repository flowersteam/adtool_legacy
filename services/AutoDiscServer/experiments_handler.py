from numpy.lib.function_base import _diff_dispatcher
from auto_disc import REGISTRATION
from auto_disc import ExperimentPipeline
from auto_disc.utils.callbacks import CustomPrintCallback, CustomSaveCallback
from AutoDiscServer.utils import CheckpointsStatusEnum
import threading
import requests
import json
import datetime


# def merge_dict(ditc1, dict2):
#     for key in ditc1:
#         if key not in dict2:
#             dict2[key] = ditc1[key]
#     return dict2


class ExperimentsHandler():
    def __init__(self):
        self._experiments = []
        self._id = 0

    def add_experiment(self, parameters):
        # TODO: Add entry in DB to obtain the id
        # id = self._id
        # self._id += 1
        try:
            # Get explorer
            explorer_class = REGISTRATION['explorers'][parameters['explorer']['name']]
            explorer = explorer_class(**parameters['explorer']['parameters'])

            # Get system
            system_class = REGISTRATION['systems'][parameters['system']['name']]
            system = system_class(**parameters['system']['parameters'])

            # Get input wrappers
            input_wrappers = []
            for _input_wrapper in parameters['input_wrappers']:
                input_wrapper_class = REGISTRATION['input_wrappers'][_input_wrapper['name']]
                input_wrappers.append(
                    input_wrapper_class(**_input_wrapper['parameters'])
                )

            # Get output representations
            output_representations = []
            for _output_representation in parameters['output_representations']:
                output_representation_class = REGISTRATION['output_representations'][_output_representation['name']]
                output_representations.append(
                    output_representation_class(**_output_representation['parameters'])
                )

            #Add experiment in DB and obtain the id
            exp_date = datetime.datetime.now()
            exp_date = exp_date.strftime("%Y")+ exp_date.strftime("%m")+ exp_date.strftime("%d")
            response = requests.post("http://127.0.0.1:3000" + "/experiments",
                                     json={
                                        "name": parameters['name'],
                                        "created_on":exp_date,
                                        "config":json.dumps(parameters['parameters']) #TODO
                                     })
            id = response.headers["Location"].split(".")
            id = int(id[1])

            # Create experiment
            experiment = ExperimentPipeline(
                system=system,
                explorer=explorer,
                input_wrappers=input_wrappers,
                output_representations=output_representations,
                on_exploration_callbacks=[
                    CustomPrintCallback('New discovery for experiment n°{}'.format(id)), 
                    CustomSaveCallback("../experiment_results/")], # TODO
                on_finish_callbacks=[CustomPrintCallback('Experiment n°{} finished !'.format(id))], # TODO
                on_cancel_callbacks=[CustomPrintCallback('Experiment n°{} cancelled !'.format(id))], # TODO
            )

            ## launch async expe
            task = threading.Thread(target=experiment.run, args=(parameters['nb_iterations'], ))
            task.start()
            
            ## Add it in the list 
            self._experiments.append({
                "ID": id,
                "experiment_object": experiment,
                "task": task,
            })

                        
            response = requests.post("http://127.0.0.1:3000" + "/systems",
                                     json={
                                        "experiment_id": id,
                                        "name": parameters['system']['name'],
                                        "config":json.dumps(system.config)
                                     })
            response = requests.post("http://127.0.0.1:3000" + "/explorers",
                                     json={
                                        "experiment_id": id,
                                        "name": parameters['explorer']['name'],
                                        "config":json.dumps(explorer.config)
                                     })
            # response = requests.post("http://127.0.0.1:3000" + "/input_wrappers",
            #                          json=
            #                             [{"experiment_id": id, 
            #                               "name":parameters['input_wrappers'][i]['name'], 
            #                               "config":input_wrappers[i].config, #TODO
            #                               "index": i} 
            #                               for i in range(len(input_wrappers))]
            #                          )
            response = requests.post("http://127.0.0.1:3000" + "/input_wrappers",
                                     json=
                                        [{"experiment_id": id, 
                                          "name":parameters['input_wrappers'][i]['name'], 
                                          "config":parameters['input_wrappers'][i]['parameters'], #TODO
                                          "index": i} 
                                          for i in range(len(input_wrappers))]
                                     )
            # response = requests.post("http://127.0.0.1:3000" + "/input_wrappers",
            #                          json=
            #                             [{"experiment_id": id, 
            #                               "name":parameters['input_wrappers'][i]['name'], 
            #                               "config":merge_dict(parameters['input_wrappers'][i]['parameters'], input_wrappers[i].config), #TODO
            #                               "index": i} 
            #                               for i in range(len(input_wrappers))]
            #                          )
            response = requests.post("http://127.0.0.1:3000" + "/output_representations",
                                     json=
                                        [{"experiment_id": id, 
                                          "name":parameters['output_representations'][i]['name'], 
                                          "config":output_representations[i].config, #TODO
                                          "index": i} 
                                          for i in range(len(output_representations))]
                                     )

            return id
        except Exception as err:
            message = "Error when creating experiment:" + str(err)
            raise Exception(message)

    def _get_experiment(self, id):
        # Check if experiment is in the list
        enumerated_experiment_entry = next(filter(lambda element: element[1]["ID"] == id, enumerate(self._experiments)), None)
        if enumerated_experiment_entry is None:
            raise Exception("Unknown experiment ID !")
        
        index = enumerated_experiment_entry[0]
        return self._experiments[index], index


    def remove_experiment(self, id):
        # Get index in list
        _, index = self._get_experiment(id)
        
        # Pop from list
        experiment = self._experiments.pop(index)

        # Cancel experiment
        experiment['experiment_object'].cancellation_token.trigger()
        experiment['task'].join()

        # Notify the DB
        # TODO through callback ?
        # response = requests.patch("http://127.0.0.1:3000" + "/checkpoints?experiment_id=eq." + str(12),
        response = requests.patch("http://127.0.0.1:3000" + "/checkpoints?experiment_id=eq." + str(experiment["ID"]),
                                     json=
                                        {"status" : int(CheckpointsStatusEnum.CANCELLED)} 
                                     )        

    def list_running_experiments(self):
        return [expe['ID'] for expe in self._experiments]