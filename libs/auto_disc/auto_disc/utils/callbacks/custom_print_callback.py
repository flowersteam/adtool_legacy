from auto_disc.utils.callbacks import BaseCallback

class CustomPrintCallback(BaseCallback):
    def __init__(self, custom_message_to_print, **kwargs):
        """
        init the callback with a message to print
        custom_message_to_print: string
        """
        super().__init__(**kwargs)
        self._custom_message_to_print = custom_message_to_print

    def __call__(self, experiment_id, seed, **kwargs):
        """
        callback print a message
        """
        print(self._custom_message_to_print + " / Iteration: {}".format(kwargs["run_idx"]))