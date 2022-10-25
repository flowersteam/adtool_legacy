import threading
from typing import Any
from auto_disc.utils.callbacks import BaseCallback

Object = lambda **kwargs: type("Object", (), kwargs)()
global Interact
Interact = None#Object(callbacks={}, config={"idx":0})

class Interaction():
    def __init__(self) -> None:
        if Interact == None:
            self.interact = {}

    def init_seed(self, interact_callbacks, config,):
        self.interact[threading.current_thread().ident] = {}
        self.interact[threading.current_thread().ident]["callbacks"] = {}
        self.interact[threading.current_thread().ident]["config"] = {}
        self.interact[threading.current_thread().ident]["callbacks"].update(interact_callbacks)
        self.interact[threading.current_thread().ident]["config"].update(config)
    
    def __call__(self, callback_name, data, dict_info=None,**kwds: Any) -> Any:
        return self.interact[threading.current_thread().ident]["callbacks"][callback_name](data, self.interact[threading.current_thread().ident]["config"], dict_info)

Interact = Interaction()

class BaseInteractCallback(BaseCallback):
    '''
    Base class for cancelled callbacks used by the experiment pipelines when the experiment was cancelled.
    '''

    def __init__(self, **kwargs) -> None:
        """
            initialize attributes common to all cancelled callbacks

            Args:
                kwargs: some usefull args (e.g. experiment_id...)
        """
        super().__init__(**kwargs)
        self.interactMethod = kwargs["interactMethod"]

    def __call__(self, data, config, dict_info=None,**kwargs) -> None:
        """
            The function to call to effectively raise cancelled callback.
            Inform the user that the experience is in canceled status
            Args:
                experiment_id: current experiment id
                seed: current seed number
                kwargs: somme usefull parameters
        """
        self.interactMethod(data, config["seed"], dict_info)
        config["idx"] += 1