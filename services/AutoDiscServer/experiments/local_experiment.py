import requests
from AutoDiscServer.experiments import BaseExperiment
from AutoDiscServer.utils.DB import ExpeDBCaller, AppDBLoggerHandler
from AutoDiscServer.utils.DB.expe_db_utils import serialize_autodisc_space

from auto_disc.run import create, start as start_pipeline

import threading
import pickle
import logging
from copy import copy

class LocalExperiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.threading_lock = threading.Lock()

        additional_callbacks = {
            "on_discovery": [self.on_progress, self.save_discovery_to_expe_db],
            "on_save_finished": [self.on_save],
            "on_finished": [self.on_finished],
            "on_cancelled": [self.on_cancelled],
            "on_error": [self.on_error],
            "on_saved": [self.save_modules_to_expe_db],
        }
        
        
        additional_handlers = [AppDBLoggerHandler('http://127.0.0.1:3000', self.id, self.__get_current_checkpoint_id)]

        self._expe_db_caller = ExpeDBCaller('http://127.0.0.1:5001')

        self._pipelines = []
        for i in range(self.experiment_config['experiment']['config']['nb_seeds']):
            seed = i
            experiment_id = self.experiment_config['experiment']['id']
            self._pipelines.append(create(self.cleared_config, experiment_id, seed, additional_callbacks, additional_handlers))

    def start(self):
        print("Starting local experiment with id {} and {} seeds".format(self.id, self.experiment_config['experiment']['config']['nb_seeds']))
        self._running_tasks = []
        for pipeline in self._pipelines:
            task = threading.Thread(target=start_pipeline, args=(pipeline, self.experiment_config['experiment']['config']['nb_iterations'], ))
            task.start()
            self._running_tasks.append(task)

    def stop(self):
        print("Stopping {} seeds of local experiment with id {}".format(self.experiment_config['experiment']['config']['nb_seeds'], self.id))
        for i in range(self.experiment_config['experiment']['config']['nb_seeds']):
            self._pipelines[i].cancellation_token.trigger()
            self._running_tasks[i].join()

    def on_progress(self, **kwargs):
        super().on_progress(kwargs["seed"])

    def on_save(self, **kwargs):
        self.threading_lock.acquire()
        super().on_save(kwargs["seed"], self.__get_current_checkpoint_id(kwargs["seed"]))
        self.threading_lock.release()

    def on_error(self, **kwargs):
        self.threading_lock.acquire()
        res =  super().on_error(kwargs["seed"], kwargs["checkpoint_id"])
        self.threading_lock.release()
        return res

    def on_finished(self, **kwargs):
        self.threading_lock.acquire()
        super().on_finished(kwargs["seed"])
        self.threading_lock.release()
    
    def on_cancelled(self, **kwargs):
        self.threading_lock.acquire()
        super().on_cancelled(kwargs["seed"])
        self.threading_lock.release()

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
                saves[save_item] = serialize_autodisc_space(kwargs[save_item])
        
        discovery_id = self._expe_db_caller("/discoveries", request_dict=saves)["ID"]
        self._expe_db_caller("/discoveries/" + discovery_id + "/files", files=files_to_save)

    def save_modules_to_expe_db(self, **kwargs):
        #TODO convert to_save_modules --> self.to_save_modules (like on_discovery_*_callback)
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
                                        "checkpoint_id": self.__get_current_checkpoint_id(kwargs["seed"]),
                                        "run_idx": kwargs["run_idx"],
                                        "seed": kwargs["seed"]
                                    }
                                )["ID"]
        self._expe_db_caller("/checkpoint_saves/" + module_id + "/files", files=files_to_save)

    def __get_current_checkpoint_id(self, seed):
        current_checkpoint_id = next(
            checkpoint_id 
            for checkpoint_id, history in self.checkpoints_history.items()
            if seed not in history["seeds_status"]
            )
        return current_checkpoint_id

    def clean_after_experiment(self, **kwargs):
        super().clean_after_experiment()
        ad_logger = logging.getLogger("ad_tool_logger")
        ad_logger.handlers = [handler for handler in ad_logger.handlers if handler.experiment_id != self.id]
    