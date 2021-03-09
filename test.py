from libs.systems.python_systems import PythonLenia
from libs.auto_disc.output_representations.specific import LeniaImageRepresentation
from libs.auto_disc.explorers import IMGEPExplorer

from libs.auto_disc import ExperimentPipeline
from libs.auto_disc.utils import BaseAutoDiscCallback

class CustomPrintCallback(BaseAutoDiscCallback):
    def __init__(self, custom_message_to_print):
        super().__init__()
        self._custom_message_to_print = custom_message_to_print

    def __call__(self, **kwargs):
        print(self._custom_message_to_print)


experiment = ExperimentPipeline(
    system=PythonLenia(
        config_kwargs={
            'SX':256, 
            'SY':256
        }),
    explorer=IMGEPExplorer(),
    input_wrappers=None,
    output_representations=[LeniaImageRepresentation()],
    on_exploration_callbacks=[CustomPrintCallback("Newly explored output !")]
)

experiment.run(5000)