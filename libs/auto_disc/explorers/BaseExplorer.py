from libs.utils.AttrDict import AttrDict

class BaseExplorer ():
    '''
    Base class for exploration experiments.
    Allows to save and load exploration results
    '''

    CONFIG_DEFINITION = []

    @classmethod
    def get_default_config(cls):
        default_config = AttrDict()
        for param in cls.CONFIG_DEFINITION:
            param_dict = param.to_dict()
            default_config[param_dict['name']] = param_dict['default']
        
        return default_config

    def __init__(self, **kwargs):
        self.config = self.get_default_config()
        self.config.update(kwargs)

    def emit(self):
        raise NotImplementedError()

    def archive(self, parameters, output_representation):
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