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
    logger=AutoDiscLogger(42, 0, [SetFileHandler("./", experiment_id)])
    experiment = ExperimentPipeline(
        experiment_id=experiment_id,
        seed=42,
        save_frequency = 2,
        system=PythonLenia(final_step=200, scale_init_state=1.0, logger=logger),
        explorer=IMGEPExplorer(logger=logger),
        input_wrappers=[
            CppnInputWrapper('init_state', logger=logger), 
            TimesNInputWrapper('R', n=10, logger=logger), 
            TimesNInputWrapperChild('b', l=20, logger=logger)], # Starting from the explorer !
        output_representations=[LeniaHandDefinedRepresentation(logger=logger)], # Starting from the system !
        on_discovery_callbacks=[CustomPrintCallback("Newly explored output !", logger=logger), 
                                  OnDiscoverySaveCallbackOnDisk("./experiment_results/", logger=logger, 
                                                                to_save_outputs=[
                                                                    "Parameters sent by the explorer before input wrappers",
                                                                    "Parameters sent by the explorer after input wrappers",
                                                                    "Raw system output",
                                                                    "Representation of system output",
                                                                    "Rendered system output"
                                                                ])],
        on_save_callbacks=[OnSaveModulesOnDiskCallback("./", logger=logger)],
    )

    experiment.run(10)
