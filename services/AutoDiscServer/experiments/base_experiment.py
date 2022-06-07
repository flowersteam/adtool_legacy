import json
from AutoDiscServer.utils import SeedStatusEnum, ExperimentStatusEnum, CheckpointsStatusEnum
from AutoDiscServer.utils import clear_dict_config_parameter, AutoDiscServerConfig
from AutoDiscServer.utils.DB import ExpeDBCaller, AppDBCaller, AppDBMethods

class BaseExperiment():
    def __init__(self, id, experiment_config, on_progress_callback, on_checkpoint_needed_callback, on_checkpoint_finished_callback,
                on_checkpoint_update_callback, on_experiment_update_callback):
        self.id = id
        self.experiment_config = experiment_config
        self._on_progress_callback = on_progress_callback
        self._on_checkpoint_needed_callback = on_checkpoint_needed_callback
        self._on_checkpoint_finished_callback = on_checkpoint_finished_callback
        self._on_checkpoint_update_callback = on_checkpoint_update_callback
        self._on_experiment_update_callback = on_experiment_update_callback
        self.autoDiscServerConfig = AutoDiscServerConfig()
        self._expe_db_caller = ExpeDBCaller('http://{}:{}'.format(self.autoDiscServerConfig.EXPEDB_CALLER_HOST, self.autoDiscServerConfig.EXPEDB_CALLER_PORT))
        self._app_db_caller = AppDBCaller("http://{}:{}".format(self.autoDiscServerConfig.APPDB_CALLER_HOST, self.autoDiscServerConfig.APPDB_CALLER_PORT))

        # Progress handling
        self._initialize_checkpoint_history()

        self.experiment_config['experiment']['id'] = id

        # TODO: Find a way to avoid this
        self.experiment_config['experiment']['save_frequency'] = self.experiment_config['experiment']['config']['save_frequency']
        del self.experiment_config['experiment']['config']['save_frequency']

        #TODO when the user can choose callbacks delete this
        if self.experiment_config['callbacks'] == []:
            self.experiment_config['callbacks'] = {
                'on_discovery': [
                    {
                        'name' : 'base',
                        'config': {
                            'to_save_outputs': self.experiment_config['experiment']['config']['discovery_saving_keys']
                        }
                    }
                ],
                'on_save_finished': [
                    {
                        'name' : 'base',
                        'config': {}
                    }
                ],
                'on_cancelled': [
                    {
                        'name' : 'base',
                        'config': {}
                    }
                ],
                'on_finished': [
                    {
                        'name' : 'base',
                        'config': {}
                    }
                ],
                'on_error': [
                    {
                        'name' : 'base',
                        'config': {}
                    }
                ],
                'on_saved': [
                    {
                        'name' : 'base',
                        'config': {}
                    }
                ],
            }

        self.cleared_config = clear_dict_config_parameter(self.experiment_config)

#region public launching
    def prepare(self):
        raise NotImplementedError()
    
    def start(self):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()
    
    def reload(self):
        raise NotImplementedError()
#endregion

#region callbacks
    def on_progress(self, seed, progress_value=None):
        self.progresses[seed] = self.progresses[seed] + 1 if progress_value is None else progress_value
        min_progress = min(self.progresses.values())
        if min_progress > self.current_progress:
            self.current_progress = min_progress
            self._on_progress_callback(self.id, self.current_progress)

    def on_save(self, seed, current_checkpoint_id):
        checkpoint_id = current_checkpoint_id
        # Create new checkpoint if needed
        # if the seed is the first to have not crashed has arrived in this checkpoint
        # AND 
        # if we need to make a new backup afterwards then we create a new checkpoint
        if not any(value == int(SeedStatusEnum.DONE) for value in self.checkpoints_history[current_checkpoint_id]["seeds_status"].values()):   
                checkpoint_id = self._on_checkpoint_needed_callback(self.id, current_checkpoint_id)
                self.checkpoints_history[checkpoint_id] = {
                    "seeds_status": {},
                    "parent_id": current_checkpoint_id
                }               
        elif self.progresses[seed] <= self.experiment_config['experiment']['config']["nb_iterations"] - self.experiment_config['experiment']['save_frequency']:            
            list_index = list(self.checkpoints_history)
            checkpoint_id = list_index[list_index.index(current_checkpoint_id)+1]
        
        # Update list of seeds for precedent checkpoint
        self.checkpoints_history[current_checkpoint_id]["seeds_status"][seed] = int(SeedStatusEnum.DONE)

        # Put precedent checkpoint to done if all seeds finished it
        if len(self.checkpoints_history[current_checkpoint_id]["seeds_status"]) >= len(self.progresses) \
            and \
            all([value == int(SeedStatusEnum.DONE) or value == int(SeedStatusEnum.ERROR) 
                 for value in list(self.checkpoints_history[current_checkpoint_id]["seeds_status"].values())]) == True:
            self._on_checkpoint_finished_callback(self.id, current_checkpoint_id)

        return checkpoint_id

    def on_error(self, seed, current_checkpoint_id):
        del self.progresses[seed]
        
        # Update list of seeds for precedent checkpoint
        self.checkpoints_history[current_checkpoint_id]["seeds_status"][seed] = int(SeedStatusEnum.ERROR)

        error = len(self.progresses) == 0
        # Put precedent checkpoint to error
        self._on_checkpoint_update_callback(current_checkpoint_id, error)
        if error:
            self._on_experiment_update_callback(self.id, {"exp_status": int(ExperimentStatusEnum.ERROR)})
            self.clean_after_experiment()

    def on_finished(self, seed):
        del self.progresses[seed]
        if len(self.progresses) == 0:
            self._on_experiment_update_callback(self.id, {"exp_status": int(ExperimentStatusEnum.DONE)})
            self.clean_after_experiment()
    
    def on_cancelled(self, seed):
        del self.progresses[seed]
        if len(self.progresses) == 0:
            # self._on_experiment_update_callback(self.id, {"exp_status": int(ExperimentStatusEnum.CANCELLED)})
            self.clean_after_experiment()
#endregion

#region utils
    def _get_current_checkpoint_id(self, seed):
        current_checkpoint_id = next(
            checkpoint_id 
            for checkpoint_id, history in self.checkpoints_history.items()
            if seed not in history["seeds_status"] or history["seeds_status"][seed] != int(SeedStatusEnum.DONE)
        )
        return current_checkpoint_id

    def clean_after_experiment(self):
        pass

    def callback_to_all_running_seeds(self, callback):
        for i in range(self.experiment_config['experiment']['config']['nb_seeds']):
            current_seed_chekpoint_id = self._get_current_checkpoint_id(i)
            current_seed_checkpoint_history = self.checkpoints_history[current_seed_chekpoint_id]
            # If runnning apply callback
            if i not in current_seed_checkpoint_history["seeds_status"] or current_seed_checkpoint_history["seeds_status"][i] == int(SeedStatusEnum.RUNNING):
                callback(i, current_seed_chekpoint_id)

    def _initialize_checkpoint_history(self):
        # Get checkpoints from DB
        response = self._app_db_caller("/checkpoints?experiment_id=eq.{}&order=id".format(self.id), 
                                        AppDBMethods.GET, {}
                                      )
        checkpoints_from_db = json.loads(response.content)

        # Init progress
        self.progresses = dict(zip([i for i in range(self.experiment_config['experiment']['config']["nb_seeds"])], 
                                   [0 for _ in range(self.experiment_config['experiment']['config']["nb_seeds"])]))
        self.current_progress = 0
        self.checkpoints_history = {}

        # Set checkpoint history and progress based on DB 
        if len(checkpoints_from_db) == 0: # Experiment's first start
            checkpoint_id = self._on_checkpoint_needed_callback(self.id, None)
            self.checkpoints_history[checkpoint_id] = {
                "seeds_status": {},
                "parent_id": None
            }
        else: # Experiment's reload
            nb_seeds = self.experiment_config['experiment']['config']["nb_seeds"]
            checkpoint_frequency = self.experiment_config['experiment']['config']['save_frequency']
            for checkpoint in checkpoints_from_db:
                self.checkpoints_history[checkpoint['id']] = {
                    "seeds_status": dict(zip([i for i in range(nb_seeds)], 
                                             [SeedStatusEnum(checkpoint['status']) for _ in range(nb_seeds)])),
                    "parent_id": checkpoint['parent_id']
                }

                if CheckpointsStatusEnum(checkpoint['status']) == CheckpointsStatusEnum.DONE:
                    self.current_progress += checkpoint_frequency
                    self.progresses = dict.fromkeys(self.progresses, self.current_progress)
#endregion