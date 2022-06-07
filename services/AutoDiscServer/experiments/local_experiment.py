from AutoDiscServer.experiments import BaseExperiment
from AutoDiscServer.utils.DB import AppDBLoggerHandler, AppDBMethods
from AutoDiscServer.utils.DB.expe_db_utils import serialize_autodisc_space, is_json_serializable
from AutoDiscServer.utils import ExperimentStatusEnum, CheckpointsStatusEnum

from auto_disc.run import create, start as start_pipeline

import threading
import pickle
import logging
from copy import copy

class LocalExperiment(BaseExperiment):
    '''
        Local Python experiment. Pipelines are directly launched from this class with additional callbacks provided to handle DB storage. 
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.threading_lock = threading.Lock()

        self._additional_callbacks = {
            "on_discovery": [self.on_progress, self.save_discovery_to_expe_db],
            "on_save_finished": [self.on_save],
            "on_finished": [self.on_finished],
            "on_cancelled": [self.on_cancelled],
            "on_error": [self.on_error],
            "on_saved": [self.save_modules_to_expe_db],
        }
        
        self._additional_handlers = [AppDBLoggerHandler('http://{}:{}'.format(self.autoDiscServerConfig.APPDB_CALLER_HOST, self.autoDiscServerConfig.APPDB_CALLER_PORT), self.id, self._get_current_checkpoint_id)]

        self._pipelines = []

#region public launching
    def prepare(self):
        self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "experimence in preparation"
                                }
                            )
        for i in range(self.experiment_config['experiment']['config']['nb_seeds']):
            seed = i
            experiment_id = self.experiment_config['experiment']['id']
            self._pipelines.append(create(self.cleared_config, experiment_id, seed, self._additional_callbacks, self._additional_handlers))

    def start(self):
        print("Starting local experiment with id {} and {} seeds".format(self.id, self.experiment_config['experiment']['config']['nb_seeds']))
        self._running_tasks = []
        response = self._app_db_caller("/experiments?id=eq.{}".format(self.id), 
                                AppDBMethods.PATCH, 
                                {"exp_status": ExperimentStatusEnum.RUNNING})
        self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "the experiment start"
                                }
                            )
        for pipeline in self._pipelines:
            task = threading.Thread(target=start_pipeline, args=(pipeline, self.experiment_config['experiment']['config']['nb_iterations'], ))
            task.start()
            self._running_tasks.append(task)

    def stop(self):
        print("Stopping {} seeds of local experiment with id {}".format(self.experiment_config['experiment']['config']['nb_seeds'], self.id))
        for i in range(self.experiment_config['experiment']['config']['nb_seeds']):
            self._pipelines[i].cancellation_token.trigger()
            self._running_tasks[i].join()

    def reload(self):
        self._app_db_caller("/checkpoints?experiment_id=eq.{}&status=eq.{}".format(self.id, int(CheckpointsStatusEnum.RUNNING)), 
                                AppDBMethods.PATCH, 
                                {"status": int(CheckpointsStatusEnum.CANCELLED)} 
                        )
        self._app_db_caller("/experiments?id=eq.{}".format(self.id), 
                            AppDBMethods.PATCH, 
                            {"exp_status": int(CheckpointsStatusEnum.CANCELLED)} 
                    ) 
#endregion

#region callbacks
    def on_progress(self, **kwargs):
        super().on_progress(kwargs["seed"], kwargs["run_idx"] + 1)

    def on_save(self, **kwargs):
        self.threading_lock.acquire()
        super().on_save(kwargs["seed"], self._get_current_checkpoint_id(kwargs["seed"]))
        self.threading_lock.release()

    def on_error(self, **kwargs):
        self.threading_lock.acquire()
        super().on_error(kwargs["seed"], self._get_current_checkpoint_id(kwargs["seed"]))
        self.threading_lock.release()

    def on_finished(self, **kwargs):
        self.threading_lock.acquire()
        super().on_finished(kwargs["seed"])
        self.threading_lock.release()
    
    def on_cancelled(self, **kwargs):
        self.threading_lock.acquire()
        super().on_cancelled(kwargs["seed"])
        self.threading_lock.release()
#endregion

#region saving
    def save_discovery_to_expe_db(self, **kwargs):
        """
        brief:      callback saves the discoveries outputs we want to save on database.
        comment:    always saved : run_idx(json), experiment_id(json)
                    saved if key in self.to_save_outputs: raw_run_parameters(json)
                                                        run_parameters,(json)
                                                        raw_output(file),
                                                        output(json),
                                                        rendered_output(file),
                                                        step_observations(file)
        """
        saves={}
        files_to_save={}
        to_save_outputs = copy(self.cleared_config["callbacks"]["on_discovery"][0]["config"]["to_save_outputs"])
        to_save_outputs.extend(["run_idx", "experiment_id", "seed"])

        for save_item in to_save_outputs:
            if save_item == "step_observations":
                kwargs[save_item] = serialize_autodisc_space(kwargs[save_item])

            if save_item == "raw_output" or save_item == "step_observations":
                files_to_save[save_item] = ('{}_{}_{}'.format(save_item, kwargs["experiment_id"], kwargs["run_idx"]), pickle.dumps(kwargs[save_item]), 'application/json')
            elif save_item == "rendered_output":
                filename = "exp_{}_idx_{}".format(kwargs["experiment_id"], kwargs["run_idx"])
                filename=filename+"."+kwargs["rendered_output"][1]
                files_to_save["rendered_output"] = (filename, kwargs["rendered_output"][0].getbuffer())
            else:
                serialized_object = serialize_autodisc_space(kwargs[save_item]) 
                if is_json_serializable(serialized_object):
                    saves[save_item] = serialized_object
                else:
                    files_to_save[save_item] = ('{}_{}_{}'.format(save_item, kwargs["experiment_id"], kwargs["run_idx"]), pickle.dumps(kwargs[save_item]), 'application/json')
        
        discovery_id = self._expe_db_caller("/discoveries", request_dict=saves)["ID"]
        self._expe_db_caller("/discoveries/" + discovery_id + "/files", files=files_to_save)

    def save_modules_to_expe_db(self, **kwargs):
        to_save_modules = ["system","explorer","input_wrappers","output_representations","in_memory_db"]

        files_to_save={} 
        for module in to_save_modules:
            if isinstance(kwargs[module], list):
                to_pickle = []
                for element in kwargs[module]:
                    to_pickle.append(element.save())
            else:
                to_pickle = kwargs[module].save()

            module_to_save = pickle.dumps(to_pickle)
            files_to_save[module] = module_to_save
        
        module_id = self._expe_db_caller("/checkpoint_saves", 
                                    request_dict={
                                        "checkpoint_id": self._get_current_checkpoint_id(kwargs["seed"]),
                                        "run_idx": kwargs["run_idx"],
                                        "seed": kwargs["seed"]
                                    }
                                )["ID"]
        self._expe_db_caller("/checkpoint_saves/" + module_id + "/files", files=files_to_save)
#endregion

#region utils
    def clean_after_experiment(self, **kwargs):
        super().clean_after_experiment()
        ad_logger = logging.getLogger("ad_tool_logger")
        ad_logger.handlers = [handler for handler in ad_logger.handlers if handler.experiment_id != self.id]
#endregion
    