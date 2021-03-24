from libs.auto_disc.utils import BaseAutoDiscCallback

class CustomPrintCallback(BaseAutoDiscCallback):
    def __init__(self, custom_message_to_print):
        super().__init__()
        self._custom_message_to_print = custom_message_to_print

    def __call__(self, **kwargs):
        print(self._custom_message_to_print)