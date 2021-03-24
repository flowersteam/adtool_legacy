from libs.auto_disc.utils import BaseAutoDiscCallback

class CustomPrintCallback(BaseAutoDiscCallback):
    def __init__(self, custom_message_to_print):
        """
        init the callback with a message to print
        custom_message_to_print: string
        """
        super().__init__()
        self._custom_message_to_print = custom_message_to_print

    def __call__(self, **kwargs):
        """
        callback print a message
        """
        print(self._custom_message_to_print)