import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../"))

from auto_disc.systems.python_systems import PythonLenia, CellularForm
from auto_disc.output_representations.generic import DummyOutputRepresentation
from auto_disc.output_representations.specific import LeniaImageRepresentation, LeniaHandDefinedRepresentation
from auto_disc.input_wrappers.generic import TimesNInputWrapper, CppnInputWrapper, DummyInputWrapper
from auto_disc.explorers import IMGEPExplorer

from auto_disc import ExperimentPipeline

from auto_disc.utils.callbacks import CustomPrintCallback
from auto_disc.utils.callbacks.on_discovery_callbacks import OnDiscoverySaveCallbackOnDisk

if __name__ == "__main__":
    # experiment = ExperimentPipeline(
    #     experiment_id=1,
    #     checkpoint_id=0,
    #     seed=42,
    #     system=PythonLenia(final_step=200, scale_init_state=1.0),
    #     explorer=IMGEPExplorer(),
    #     input_wrappers=[CppnInputWrapper('init_state')], # Starting from the explorer !
    #     output_representations=[LeniaHandDefinedRepresentation()], # Starting from the system !
    #     on_discovery_callbacks=[CustomPrintCallback("Newly explored output !"),
    #                               OnDiscoverySaveCallbackOnDisk("./experiment_results/",
    #                                                             to_save_outputs=[
    #                               "14_flowers_1.0"                                  "Parameters sent by the explorer before input wrappers",
    #                                                                 "Parameters sent by the explorer after input wrappers",
    #                                                                 "Raw system output",
    #                                                                 "Representation of system output",
    #                                                                 "Rendered system output"
    #                                                             ])]
    # )
    #
    # experiment.run(5)
    experiment = ExperimentPipeline(
        experiment_id=1,
        checkpoint_id=0,
        seed=42,
        system=CellularForm(
            final_step=200,
        ),
        explorer=IMGEPExplorer(),
        input_wrappers=[DummyInputWrapper()], # Starting from the explorer !
        output_representations=[DummyOutputRepresentation()], # Starting from the system !
        on_discovery_callbacks=[CustomPrintCallback("Newly explored output !"),
                                  OnDiscoverySaveCallbackOnDisk("./experiment_results/",
                                                                to_save_outputs=[
                                                                    "Parameters sent by the explorer before input wrappers",
                                                                    "Parameters sent by the explorer after input wrappers",
                                                                    "Raw system output",
                                                                    "Representation of system output",
                                                                    "Rendered system output"
                                                                ])]
    )

    experiment.run(5)
