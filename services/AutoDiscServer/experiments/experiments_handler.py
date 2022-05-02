import traceback
from AutoDiscServer.utils.experiment_status_enum import ExperimentStatusEnum
from AutoDiscServer.experiments import LocalExperiment, RemoteExperiment
from AutoDiscServer.utils import CheckpointsStatusEnum, reconstruct_parameters
from AutoDiscServer.utils.DB import  AppDBCaller, AppDBMethods
import datetime
import traceback
import threading


class ExperimentsHandler():
    def __init__(self):
        self._experiments = []
        self._app_db_caller = AppDBCaller("http://127.0.0.1:3000")

#region utils
    def _get_experiment(self, id):
        # Check if experiment is in the list
        experiment = next(iter(filter(lambda element: element.id == id, self._experiments)), None)
        if experiment is None:
            raise Exception("Unknown experiment ID !")
        
        return experiment

    def _create_experiment(self, id, parameters):
        # Create experiment
        if parameters["experiment"]["config"]["host"] == "local":
            experiment = LocalExperiment(id, parameters, 
                                        self.on_progress_callback,
                                        self.on_checkpoint_needed_callback,
                                        self.on_checkpoint_finished_callback,
                                        self.on_checkpoint_update_callback,
                                        self.on_experiment_update_callback)
        else:
            experiment = RemoteExperiment(parameters["experiment"]["config"]["host"], 
                                        id, parameters, 
                                        self.on_progress_callback,
                                        self.on_checkpoint_needed_callback,
                                        self.on_checkpoint_finished_callback,
                                        self.on_checkpoint_update_callback,
                                        self.on_experiment_update_callback)
        
        # Add it in the list 
        self._experiments.append(experiment)

        return experiment

    def _reload_experiment(self, id, parameters):
        try:
            experiment = self._create_experiment(id, parameters)
            experiment.reload()
        except:
            message = "Error when reloading experiment: {}".format(traceback.format_exc())
            raise Exception(message)
#endregion

#region main functions

    def prepare_and_start_experiment_async(self, experiment):
        # Prepare for start
        experiment.prepare()

        # Start the experiment
        experiment.start()

    def reload_running_remote_experiments(self):
        import json

        response = self._app_db_caller("/experiments?exp_status=eq.{}".format(ExperimentStatusEnum.RUNNING), 
                                        AppDBMethods.GET, {}
                                      )
        running_experiments = json.loads(response.content)
        for experiment in running_experiments:
            id = experiment['id']
            try:
                parameters = reconstruct_parameters(id, self._app_db_caller)
                self._reload_experiment(id, parameters)
            except:
                message = "Error when reloading experiment {}: {}".format(id, traceback.format_exc())
                try:
                    self.remove_experiment(id, CheckpointsStatusEnum.ERROR, ExperimentStatusEnum.ERROR)
                except:
                    message += "\nError when removing experiment: {}".format(traceback.format_exc())
                finally:
                    print(message) # TODO log in file

    def add_experiment(self, parameters):
        try:
            # Add experiment in DB and obtain the id
            exp_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z")
            response = self._app_db_caller("/experiments", AppDBMethods.POST, {
                                        "name": parameters['experiment']['name'],
                                        "created_on": exp_date,
                                        "config": parameters['experiment']['config'],
                                        "progress": 0,
                                        "exp_status": int(ExperimentStatusEnum.PREPARING),
                                        "archived": False,
                                        "checkpoint_saves_archived": False,
                                        "discoveries_archived": False
                                     })
            id = response.headers["Location"].split(".")
            id = int(id[1])

            experiment = self._create_experiment(id, parameters)

            prepare_and_start_experiment_async = threading.Thread(target=self.prepare_and_start_experiment_async, args=(experiment,))
            prepare_and_start_experiment_async.start()

            self._app_db_caller("/systems", AppDBMethods.POST, {
                                "experiment_id": id,
                                "name": parameters['system']['name'],
                                "config": parameters['system']['config']
                                })
            self._app_db_caller("/explorers", AppDBMethods.POST,{
                                "experiment_id": id,
                                "name": parameters['explorer']['name'],
                                "config": parameters['explorer']['config']
                                })
            self._app_db_caller("/input_wrappers", AppDBMethods.POST,
                                [{"experiment_id": id, 
                                    "name": parameters['input_wrappers'][i]['name'], 
                                    "config": parameters['input_wrappers'][i]['config'],
                                    "index": i} 
                                    for i in range(len(parameters['input_wrappers']))]
                                )
            self._app_db_caller("/output_representations", AppDBMethods.POST,
                                [{"experiment_id": id, 
                                    "name": parameters['output_representations'][i]['name'], 
                                    "config": parameters['output_representations'][i]['config'],
                                    "index": i} 
                                    for i in range(len(parameters['output_representations']))]
                                )

            return id
        except Exception as err:
            message = "Error when creating experiment: {}".format(traceback.format_exc())
            try:
                self.remove_experiment(id, CheckpointsStatusEnum.ERROR, ExperimentStatusEnum.ERROR)
            except Exception as second_err:
                message += "\nError when removing experiment: {}".format(traceback.format_exc())
            finally:
                self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":id,
                                    "message": message
                                }
                            )
                raise Exception(message)

    def remove_experiment(self, id, checkpoint_status=CheckpointsStatusEnum.CANCELLED, experiment_status=ExperimentStatusEnum.CANCELLED):
        try:
            # Get index in list
            experiment = self._get_experiment(id)
            
            # Pop from list
            if experiment in self._experiments:
                self._experiments.remove(experiment)

        # Cancel experiment
            experiment.stop()

        except Exception as err:
            message = "Error when removing experiment: {}".format(traceback.format_exc())
            raise Exception(message)
        finally:
            # Save in DB
            self._app_db_caller("/checkpoints?experiment_id=eq.{}&status=eq.{}".format(id, int(CheckpointsStatusEnum.RUNNING)), 
                                AppDBMethods.PATCH, 
                                {"status": int(checkpoint_status)} 
                        )
            self._app_db_caller("/experiments?id=eq.{}".format(id), 
                                AppDBMethods.PATCH, 
                                {"exp_status": int(experiment_status)} 
                        )        

    def list_running_experiments(self):
        return [expe.id for expe in self._experiments]

#endregion

#region callbacks
    def on_progress_callback(self, experiment_id, progress):
        self._app_db_caller("/experiments?id=eq.{}".format(experiment_id), 
                            AppDBMethods.PATCH, 
                            {"progress": progress})

    def on_checkpoint_needed_callback(self, experiment_id, previous_checkpoint_id):
        response = self._app_db_caller("/checkpoints", 
                            AppDBMethods.POST, 
                            {
                                "experiment_id": experiment_id,
                                "parent_id": previous_checkpoint_id,
                                "status": int(CheckpointsStatusEnum.RUNNING)
                            })
        id = response.headers["Location"].split(".")[1]
        return int(id)

    def on_checkpoint_finished_callback(self, experiment_id, checkpoint_id):
        self._app_db_caller("/checkpoints?id=eq.{}&experiment_id=eq.{}".format(checkpoint_id, experiment_id), 
                            AppDBMethods.PATCH, 
                            {"status": int(CheckpointsStatusEnum.DONE)})
    
    def on_checkpoint_update_callback(self, checkpoint_id, error):
        if error:
            self._app_db_caller("/checkpoints?id=eq.{}".format(checkpoint_id), 
                                AppDBMethods.PATCH, 
                                {"status":int(CheckpointsStatusEnum.ERROR)})
    
    def on_experiment_update_callback(self, experiement_id, param_to_update):
        """update experiment in db
        
        Args:
            param1 (ExperimentsHandler): class instance
            param (int): the id of experiment
            param2 (dict): dict of each value we want update in experiment table in db (key example : "name", "created_on", "config", "progress", "exp_status")

        Returns:
            response: The response of the request to the database
        """
        try:
            response = self._app_db_caller("/experiments?id=eq.{}".format(experiement_id), AppDBMethods.PATCH, param_to_update)
            return response
        except Exception as err:
            message = "Error when update experiment: {}".format(traceback.format_exc())
            raise Exception(message)
#endregion