from libs.systems.python_systems.Lenia import Lenia
from libs.auto_disc.output_representations.specific.LeniaOutputRepresentation import LeniaImageRepresentation
from libs.auto_disc.explorers.IMGEPExplorer import IMGEPExplorer

from libs.auto_disc.ExperimentPipeline import ExperimentPipeline
from libs.auto_disc.utils.BaseAutoDiscCallback import BaseAutoDiscCallback

class CustomPrintCallback(BaseAutoDiscCallback):
    def __init__(self, custom_message_to_print):
        super().__init__()
        self._custom_message_to_print = custom_message_to_print

    def __call__(self, **kwargs):
        print(self._custom_message_to_print)


experiment = ExperimentPipeline(
    system=Lenia(
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