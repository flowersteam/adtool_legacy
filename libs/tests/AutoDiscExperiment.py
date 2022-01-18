import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../"))

from auto_disc.systems.python_systems import PythonLenia
from auto_disc.output_representations.specific import LeniaImageRepresentation, LeniaHandDefinedRepresentation
from auto_disc.input_wrappers.generic import TimesNInputWrapper, CppnInputWrapper
from auto_disc.explorers import IMGEPExplorer

from auto_disc import ExperimentPipeline

from auto_disc.utils.callbacks import CustomPrintCallback
from auto_disc.utils.callbacks.on_discovery_callbacks import OnDiscoverySaveCallbackOnDisk
from auto_disc.utils.callbacks.on_save_callbacks import OnSaveModulesOnDiskCallback
from auto_disc.utils.logger import AutoDiscLogger
from auto_disc.utils.logger.handlers import SetFileHandler 

if __name__ == "__main__":
    experiment_id = 1
    experiment = ExperimentPipeline(
        experiment_id=experiment_id,
        checkpoint_id=0,
        seed=42,
        save_frequency = 2,
        system=PythonLenia(final_step=200, scale_init_state=1.0),
        explorer=IMGEPExplorer(),
        input_wrappers=[CppnInputWrapper('init_state')], # Starting from the explorer !
        output_representations=[LeniaHandDefinedRepresentation()], # Starting from the system !
        on_discovery_callbacks=[CustomPrintCallback("Newly explored output !"), 
                                  OnDiscoverySaveCallbackOnDisk("./experiment_results/", 
                                                                to_save_outputs=[
                                                                    "Parameters sent by the explorer before input wrappers",
                                                                    "Parameters sent by the explorer after input wrappers",
                                                                    "Raw system output",
                                                                    "Representation of system output",
                                                                    "Rendered system output"
                                                                ])],
        on_save_callbacks=[OnSaveModulesOnDiskCallback("/home/mperie/project/test/test_runpy/mes_modules_perso/")],
        logger=AutoDiscLogger(42, 0, SetFileHandler("/home/mperie/project/test/test_runpy/", experiment_id))
    )

    experiment.run(10)
