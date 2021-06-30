from AutoDiscServer.utils import SeedStatusEnum, ExperimentStatusEnum

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
        self.progresses = [0 for _ in range(self.experiment_config['experiment']['config']["nb_seeds"])]
        self.current_progress = 0
        self.checkpoints_history = {}
        checkpoint_id = self._on_checkpoint_needed_callback(id, None)
        self.nb_active_seed = self.experiment_config['experiment']['config']["nb_seeds"]
        self.checkpoints_history[checkpoint_id] = {
            "seeds_status": {},
            "parent_id": None
        }
        self.experiment_config['experiment']['id'] = id
        self.experiment_config['experiment']['checkpoint_id'] = checkpoint_id

        # TODO: Find a way to avoid this
        self.experiment_config['experiment']['save_frequency'] = self.experiment_config['experiment']['config']['save_frequency']
        del self.experiment_config['experiment']['config']['save_frequency']

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

    
    def start():
        raise NotImplementedError()

    def stop():
        raise NotImplementedError()

    def on_progress(self, seed):
        self.progresses[seed] += 1
        if min(self.progresses) > self.current_progress:
            self.current_progress += 1
            self._on_progress_callback(self.id, self.current_progress)

    def on_save(self, seed, current_checkpoint_id):
        checkpoint_id = current_checkpoint_id
        # Create new checkpoint if needed
        # if the seed is the first to have not crashed has arrived in this checkpoint
        # AND 
        # if we need to make a new backup afterwards then we create a new checkpoint
        if ((len(self.checkpoints_history[current_checkpoint_id]["seeds_status"]) == 0 
            or all(value == int(SeedStatusEnum.ERROR) for value in self.checkpoints_history[current_checkpoint_id]["seeds_status"].values()))
            and 
            self.progresses[seed] <= self.experiment_config['experiment']['config']["nb_iterations"] 
                                    - self.experiment_config['experiment']['save_frequency']):   
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
        if ((len(self.checkpoints_history[current_checkpoint_id]["seeds_status"]) >= self.nb_active_seed)
            and
            (all([value == int(SeedStatusEnum.DONE) or value == int(SeedStatusEnum.ERROR) 
                for value in list(self.checkpoints_history[current_checkpoint_id]["seeds_status"].values())]) 
                == True)):
                self._on_checkpoint_finished_callback(self.id, current_checkpoint_id)

        return checkpoint_id
    


    def on_error(self, seed, current_checkpoint_id, message):
        self.nb_active_seed -= 1
        
        # Update list of seeds for precedent checkpoint
        self.checkpoints_history[current_checkpoint_id]["seeds_status"][seed] = int(SeedStatusEnum.ERROR)

        error = (self.nb_active_seed == 0)
        # Put precedent checkpoint to error
        self._on_checkpoint_update_callback(seed, current_checkpoint_id, message, error)
        if error:
            self._on_experiment_update_callback(self.id, {"exp_status": int(ExperimentStatusEnum.ERROR)})


    def on_finished(self):
        if self.nb_active_seed > 0:
            self._on_experiment_update_callback(self.id, {"exp_status": int(ExperimentStatusEnum.DONE)})
    
    def on_cancelled(self):
        if self.nb_active_seed > 0:
            self._on_experiment_update_callback(self.id, {"exp_status": int(ExperimentStatusEnum.CANCELLED)})