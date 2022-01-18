from auto_disc.output_representations import BaseOutputRepresentation

class DummyOutputRepresentation(BaseOutputRepresentation):
    '''
    Empty OutputRepresentation used when no representation of the system's output mut be used.
    '''
    def __init__(self, wrapped_input_space_key=None, **kwargs):
        super().__init__(wrapped_input_space_key=wrapped_input_space_key, **kwargs)
        
    def map(self, observations, is_output_new_discovery, **kwargs):
        return observations
