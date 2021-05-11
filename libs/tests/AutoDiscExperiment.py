import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../"))

from auto_disc.systems.python_systems import PythonLenia
from auto_disc.output_representations.specific import LeniaImageRepresentation, LeniaHandDefinedRepresentation
from auto_disc.input_wrappers.generic import TimesNInputWrapper, CppnInputWrapper
from auto_disc.explorers import IMGEPExplorer

from auto_disc import ExperimentPipeline

from auto_disc.utils.callbacks import CustomSaveCallback, CustomExpeDBSaveCallback, CustomPrintCallback

if __name__ == "__main__":
    experiment = ExperimentPipeline(
        system=PythonLenia(scale_init_state=1.0),
        explorer=IMGEPExplorer(),
        input_wrappers=[CppnInputWrapper('init_state')], # Starting from the explorer !
        output_representations=[LeniaHandDefinedRepresentation()], # Starting from the system !
        on_exploration_callbacks=[CustomPrintCallback("Newly explored output !"), CustomExpeDBSaveCallback('http://127.0.0.1:5001', 0)]#CustomSaveCallback("./experiment_results/")]
    )

    experiment.run(100)