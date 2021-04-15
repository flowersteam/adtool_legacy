from auto_disc.input_wrappers import BaseInputWrapper

class DummyInputWrapper(BaseInputWrapper):
    '''
    Empty InputWrapper used when no wrapper should be used.
    '''
    def map(self, parameters, **kwargs):
        return parameters