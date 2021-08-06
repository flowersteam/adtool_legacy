from addict import Dict

class BaseAutoDiscModule:
    '''
        Base class of all modules usable in auto_disc.
    '''
    CONFIG_DEFINITION = {}
    config = Dict()
    _access_history = None # Function to access (readonly) history of (input, output) pairs
    _call_output_history_update = None # Function to ask history of outputs to be updated (use this if some output_representations changed)
    _call_run_parameters_history_update = None # Function to ask history of run_parameters to be updated (use this if some input_wrappers changed)

    def set_history_access_fn(self, function):
        '''
            Set the function allowing module to access (readonly) its history of (input, output) pairs.
        '''
        self._access_history = function

    def set_call_output_history_update_fn(self, function):
        '''
            Set the function asking a refresh (raw_outputs will be processed again with output representations) of all outputs in history.
        '''
        self._call_output_history_update = function

    def set_call_run_parameters_history_update_fn(self, function):
        '''
            Set the function asking a refresh (raw_run_parameters will be processed again with output representations) of all run_parameters in history.
        '''
        self._call_run_parameters_history_update = function
