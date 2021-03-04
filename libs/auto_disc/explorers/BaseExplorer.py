from libs.utils.AttrDict import AttrDict
from libs.utils.auto_disc_parameters.AutoDiscParameter import get_default_values

class BaseExplorer ():
    '''
    Base class for exploration experiments.
    Allows to save and load exploration results
    '''

    CONFIG_DEFINITION = []

    def __init__(self, **kwargs):
        self.config = get_default_values(self, self.CONFIG_DEFINITION)
        self.config.update(kwargs)

    def initialize(self, input_wrapper, output_representation):
        self._input_wrapper = input_wrapper
        self._output_representation = output_representation

    def emit(self):
        raise NotImplementedError()

    def archive(self, parameters, observations):
        raise NotImplementedError()

    def optimize(self):
        raise NotImplementedError()


    # def save(self, filepath):
    #     """
    #     Saves the explorer object using torch.save function in pickle format
    #     /!\ We intentionally empty explorer.db from the pickle
    #     because the database is already automatically saved in external files each time the explorer call self.db.add_run_data
    #     """

    #     file = open(filepath, 'wb')

    #     # do not pickle the data as already saved in extra files
    #     tmp_data = self.db
    #     self.db.reset_empty_db()

    #     # pickle exploration object
    #     torch.save(self, file)

    #     # attach db again to the exploration object
    #     self.db = tmp_data


    # @staticmethod
    # def load(explorer_filepath, load_data=True, run_ids=None, map_location='cuda'):

    #     explorer = torch.load(explorer_filepath, map_location=map_location) #TODO: deal with gpu/cpu and relative/absolute path for explorer.db.config.db_directory

    #     if load_data:
    #         explorer.db = ExplorationDB(config=explorer.db.config)
    #         explorer.db.load(run_ids=run_ids, map_location=map_location)

    #     return explorer