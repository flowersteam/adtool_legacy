class BaseExperiment():
    def __init__(self, id, experiment_config, on_progress_callback, on_checkpoint_needed_callback, on_checkpoint_finished_callback):
        self.id = id
        self.experiment_config = experiment_config
        self._on_progress_callback = on_progress_callback
        self._on_checkpoint_needed_callback = on_checkpoint_needed_callback
        self._on_checkpoint_finished_callback = on_checkpoint_finished_callback
        self.progresses = [0 for _ in range(self.experiment_config['experiment']['config']["nb_seeds"])]
        self.current_progress = 0
        self.checkpoints_history = {}
        checkpoint_id = self._on_checkpoint_needed_callback(id, None)
        self.checkpoints_history[checkpoint_id] = {
            "seeds_done": [],
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
        # Create new checkpoint if needed
        if len(self.checkpoints_history[current_checkpoint_id]["seeds_done"]) == 0:
            checkpoint_id = self._on_checkpoint_needed_callback(self.id, current_checkpoint_id)
            self.checkpoints_history[checkpoint_id] = {
                "seeds_done": [],
                "parent_id": current_checkpoint_id
            }
        else:
            checkpoint_id = next([k for k,v in self.checkpoints_history if v["parent_id"] == current_checkpoint_id])
        
        # Update list of seeds for precedent checkpoint
        self.checkpoints_history[checkpoint_id]["seeds_done"].append(seed)

        # Put precedent checkpoint to done if all seeds finished it
        if len(self.checkpoints_history[current_checkpoint_id]) == self.experiment_config['experiment']['config']["nb_seeds"]:
            self._on_checkpoint_finished_callback(self.id, current_checkpoint_id)

        return checkpoint_id
    

