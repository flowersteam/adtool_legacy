# Add a new callback to the libs 
Six differents types of callbacks could be implemented in the AutomatedDiscoveryTool libs (on_cacelled, on_discovery, on_error, on_finished, on_save, on_save_finished).<br/>

on_cancelled_callbacks are call when the user decide to stop the callbacks and cancelled it.<br/>
on_discovery_callbacks are call each time the experiment make a new discovery.<br/>
on_error_callbacks are call when the experiment crash and raise an error.<br/>
on_finished_callbacks are call when the experiment finished.<br/>
on_save_callbacks are call when the experiment save data.<br/>
on_save_finished_callbacks are call when the experiment has finish saving.<br/>


1) Each callbakcs type have its own folder.<br/> 
To add a new callback create the file in the associate folder
    ```
    example: 
            libs/auto_disc/auto_disc/utils/callbacks/on_cancelled_callbacks/my_beautifull_new_on_cancelled_callback.py
            or
            libs/auto_disc/auto_disc/utils/callbacks/on_discovery_callbacks/my_beautifull_new_on_discovery_callback.py
    ```

2) The new callback must heritate base callback class of its own type.<br/>
   ```
   example:
        MyBeautifullNewOnCancelledCallback(BaseOnCancelledCallback):
        or
        MyBeautifullNewOnDiscoveryCallback(BaseOnDiscoveryCallback)
   ```
An example to implement a new calbback :

    
    from auto_disc.utils.callbacks.on_discovery_callbacks import BaseOnDiscoveryCallback
    
    class MyBeautifullNewOnDiscoveryCallback(BaseOnDiscoveryCallback):

        def __init__(self, folder_path, to_save_outputs, **kwargs) -> None:
            super().__init__(to_save_outputs, **kwargs)
            """do some brillant stuff"""

        def __call__(self, **kwargs) -> None:
            """do some brillant stuff"""
    
    Don't forget kwargs argument in the __init__ method.
    Each time our callback will be raise the __call__ method will be execute

3) add import in libs/auto_disc/auto_disc/utils/callbacks/callbacks_sub_folder/__init__.py
    ```
    example: 
        libs/auto_disc/auto_disc/utils/callbacks/on_discovery_callbacks/__init__.py
        libs/auto_disc/auto_disc/utils/callbacks/on_cancelled_callbacks/__init__.py
    ```
4) Add new callback in registration.py in REGISTRATION dict
   ```
   Modify REGISTRATION like this:

   REGISTRATION = {
        ...
        'callbacks': {
            'on_discovery':{
                'base': on_discovery_callbacks.BaseOnDiscoveryCallback,
                'disk': on_discovery_callbacks.OnDiscoverySaveCallbackOnDisk
            },
            'on_cancelled':{
                'base': on_cancelled_callbacks.BaseOnCancelledCallback
            },
            'on_error':{
                'base': on_error_callbacks.BaseOnErrorCallback
            },
            'on_finished':{
                'base': on_finished_callbacks.BaseOnFinishedCallback
            },
            'on_saved':{
                'base': on_save_callbacks.BaseOnSaveCallback,
                'disk': on_save_callbacks.OnSaveModulesOnDiskCallback
            },
            'on_save_finished':{
                'base': on_save_finished_callbacks.BaseOnSaveFinishedCallback
            },
        },...
    }

    ==>
    
    REGISTRATION = {
        ...
        'callbacks': {
            'on_discovery':{
                'base': on_discovery_callbacks.BaseOnDiscoveryCallback,
                'disk': on_discovery_callbacks.OnDiscoverySaveCallbackOnDisk
                'new': MyBeautifullNewOnDiscoveryCallback
            },
            'on_cancelled':{
                'base': on_cancelled_callbacks.BaseOnCancelledCallback
            },
            'on_error':{
                'base': on_error_callbacks.BaseOnErrorCallback
            },
            'on_finished':{
                'base': on_finished_callbacks.BaseOnFinishedCallback
            },
            'on_saved':{
                'base': on_save_callbacks.BaseOnSaveCallback,
                'disk': on_save_callbacks.OnSaveModulesOnDiskCallback
            },
            'on_save_finished':{
                'base': on_save_finished_callbacks.BaseOnSaveFinishedCallback
            },
        },...
    }
   ```

5) For now the software don't permit to add custom callback via GUI. You could use the libs in automous way like in libs/test/AutoDiscExperiment.py and add manually your personnal callbacks. The other way is to manually add you callback like in services/AutodsicServer/flask/experiments/remote_experiments.py on the __init__ method to use yours own callback with the soft.