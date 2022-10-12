# Add a new module to the libs 
Four different types of modules can be implemented in the AutomatedDiscoveryTool libs (Systems, Explorers, Input_wrapper, Output_representations).
1) Each module has its own folder.<br/> 
To add a new module create the file in the associated folder
    ```
    example: 
            libs/auto_disc/auto_disc/explorers/myBeautifullNewExplorer.py
            or
            libs/auto_disc/auto_disc/systems/python_systems/myBeautifullNewPythonSystems.py
    ```

2) The new module must inherit from its module base class (BaseSystem, BaseOutputRepresentation...) or from a class that inherits it itself. <br/>
Respect the parent class during the implementation 🤗 <br/>
You can add decorator to your module class, it's usefull to set module config parameters. You will directly set them in the GUI. <br/>
An example to implement a new explorer :

    ```
    from auto_disc.explorers import BaseExplorer
    from auto_disc.utils.config_parameters import StringConfigParameter

    @StringConfigParameter(name="a_string_parameter", possible_values=["first_possible_value", "second_possible_value"], default="first_possible_value")
    
    class NewExplorer(BaseExplorer):

        CONFIG_DEFINITION = {}

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

        def initialize(self, input_space, output_space, input_distance_fn):
            super().initialize(input_space, output_space, input_distance_fn)
            """do some brilliant stuff"""
        
        def sample(self):
            """do some brilliant stuff"""

        def observe(self, parameters, observations):
            """do some brilliant stuff"""

        def optimize(self):
            """do some brilliant stuff"""
    ```
    Don't forget kwargs argument in the __init__ method and CONFIG_DEFINITION.

3) Add import in libs/auto_disc/module_cat/subfolder_if_needed/__init__.py
    ```
    example: 
        libs/auto_disc/auto_disc/explorers/__init__.py
        libs/auto_disc/auto_disc/systems/python_systems/__init__.py
    ```
4) Add new module in registration.py in REGISTRATION dict
   ```
   Modify REGISTRATION like this:

   REGISTRATION = {
        'systems':{
            'PythonLenia': PythonLenia,
        },...
    }

    ==>
    
    REGISTRATION = {
        'systems':{
            'PythonLenia': PythonLenia,
            'MyBeautifullNewPythonSystems': MyBeautifullNewPythonSystems,
        },...
    }

    the dict key are used in GUI to choose your module when you setup an experiment.
   ```