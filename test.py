from libs.systems.python_systems import PythonLenia
from libs.auto_disc.output_representations.specific import LeniaImageRepresentation
from libs.auto_disc.explorers import IMGEPExplorer

from libs.auto_disc import ExperimentPipeline

from libs.auto_disc.utils.callbacks import CustomSaveCallback
from libs.auto_disc.utils.callbacks import CustomPrintCallback

import asyncio

if __name__ == "__main__":
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
        #on_exploration_callbacks=[CustomSaveCallback("/home/mperie/project/save_callback/")]
    )

    asyncio.run(experiment.run(100))