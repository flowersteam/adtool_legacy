from auto_disc.utils.callbacks import BaseCallback

class CustomPrintCallback(BaseCallback):
    def __init__(self, custom_message_to_print):
        """
        init the callback with a message to print
        custom_message_to_print: string
        """
        super().__init__()
        self._custom_message_to_print = custom_message_to_print

    def __call__(self, pipeline, run_idx, raw_run_parameters, run_parameters, raw_output, output, rendered_output, step_observations):
        """
        callback print a message
        """
        print(self._custom_message_to_print + " / Iteration: {}".format(run_idx))