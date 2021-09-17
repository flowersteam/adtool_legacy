from auto_disc.output_representations import BaseOutputRepresentation

class DummyOutputRepresentation(BaseOutputRepresentation):
    '''
    Empty OutputRepresentation used when no representation of the system's output mut be used.
    '''
    def map(self, observations, is_output_new_discovery, **kwargs):
        return observations
