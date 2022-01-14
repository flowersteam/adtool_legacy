from auto_disc.input_wrappers import BaseInputWrapper

class DummyInputWrapper(BaseInputWrapper):
    '''
    Empty InputWrapper used when no wrapper should be used.
    '''

    def __init__(self, wrapped_output_space_key=None, **kwargs):
        super().__init__(wrapped_output_space_key=wrapped_output_space_key, **kwargs)
        
    def map(self, parameters, is_input_new_discovery, **kwargs):
        return parameters