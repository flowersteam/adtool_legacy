from AutoDiscServer.utils import SeedStatusEnum, ExperimentStatusEnum
from AutoDiscServer.utils import clear_dict_config_parameter

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
        self.progresses = dict(zip([i for i in range(self.experiment_config['experiment']['config']["nb_seeds"])], 
                                   [0 for _ in range(self.experiment_config['experiment']['config']["nb_seeds"])]))
        
        self.current_progress = 0
        self.checkpoints_history = {}
        checkpoint_id = self._on_checkpoint_needed_callback(id, None)
        self.checkpoints_history[checkpoint_id] = {
            "seeds_status": {},
            "parent_id": None
        }
        self.experiment_config['experiment']['id'] = id
        self.experiment_config['experiment']['checkpoint_id'] = checkpoint_id

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
    
    def start():
        raise NotImplementedError()

    def stop():
        raise NotImplementedError()

    def on_progress(self, seed):
        self.progresses[seed] += 1
        if min(self.progresses.values()) > self.current_progress:
            self.current_progress += 1
            self._on_progress_callback(self.id, self.current_progress)

    def on_save(self, seed, current_checkpoint_id):
        checkpoint_id = current_checkpoint_id
        # Create new checkpoint if needed
        # if the seed is the first to have not crashed has arrived in this checkpoint
        # AND 
        # if we need to make a new backup afterwards then we create a new checkpoint
        if ((len(self.checkpoints_history[current_checkpoint_id]["seeds_status"]) == 0 
            or all(value == int(SeedStatusEnum.ERROR) for value in self.checkpoints_history[current_checkpoint_id]["seeds_status"].values()))):   
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
        if ((len(self.checkpoints_history[current_checkpoint_id]["seeds_status"]) >= len(self.progresses))
            and
            (all([value == int(SeedStatusEnum.DONE) or value == int(SeedStatusEnum.ERROR) 
                for value in list(self.checkpoints_history[current_checkpoint_id]["seeds_status"].values())]) 
                == True)):
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
            self._on_experiment_update_callback(self.id, {"exp_status": int(ExperimentStatusEnum.CANCELLED)})
            self.clean_after_experiment()

    def _get_current_checkpoint_id(self, seed):
        current_checkpoint_id = next(
            checkpoint_id 
            for checkpoint_id, history in self.checkpoints_history.items()
            if seed not in history["seeds_status"] or history["seeds_status"][seed] == int(SeedStatusEnum.ERROR)
            )
        return current_checkpoint_id

    def clean_after_experiment(self):
        pass