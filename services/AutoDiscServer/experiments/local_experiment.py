from AutoDiscServer.experiments import BaseExperiment
from auto_disc.run import create, start as start_pipeline
import threading
from copy import copy

class LocalExperiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.threading_lock = threading.Lock()
        
        self.experiment_config['callbacks']['on_discovery'][0]['name'] = 'expeDB'
        self.experiment_config['callbacks']['on_discovery'][0]['config']['base_url'] = 'http://127.0.0.1:5001'

        self.experiment_config['callbacks']['on_saved'][0]['name'] = 'expeDB'
        self.experiment_config['callbacks']['on_saved'][0]['config']['base_url'] = 'http://127.0.0.1:5001'

        additional_callbacks = {
            "on_discovery": [self.on_progress],
            "on_save_finished": [self.on_save],
            "on_finished": [self.on_finished],
            "on_cancelled": [self.on_cancelled],
            "on_error": [self.on_error],
            "on_saved": [],
        }

        self._pipelines = []
        for i in range(self.experiment_config['experiment']['config']['nb_seeds']):
            config = copy(self.experiment_config)
            config['experiment']["seed"] = i
            self._pipelines.append(create(config, additional_callbacks))

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
        res = super().on_save(kwargs["seed"], kwargs["checkpoint_id"])
        self.threading_lock.release()
        return {"checkpoint_id": res}

    def on_error(self, **kwargs):
        self.threading_lock.acquire()
        res =  super().on_error(kwargs["seed"], kwargs["checkpoint_id"], kwargs["message"])
        self.threading_lock.release()
        return res

    def on_finished(self, **kwargs):
        self.threading_lock.acquire()
        super().on_finished()
        self.threading_lock.release()
    
    def on_cancelled(self, **kwargs):
        self.threading_lock.acquire()
        super().on_cancelled()
        self.threading_lock.release()