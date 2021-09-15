import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../"))

from auto_disc.systems.python_systems import PythonLenia
from auto_disc.output_representations.specific import LeniaImageRepresentation, LeniaHandDefinedRepresentation, LeniaImageSelector
from auto_disc.output_representations.generic import PCA, UMAP,VAE
from auto_disc.input_wrappers.generic import TimesNInputWrapper, CppnInputWrapper
from auto_disc.explorers import IMGEPExplorer

from auto_disc import ExperimentPipeline

from auto_disc.utils.callbacks import CustomPrintCallback
from auto_disc.utils.callbacks.on_discovery_callbacks import OnDiscoverySaveCallbackOnDisk

from auto_disc.run import _set_seed

if __name__ == "__main__":
    seed = 42
    _set_seed(seed)
    experiment = ExperimentPipeline(
        experiment_id=0,
        checkpoint_id=0,
        seed=seed,
        save_frequency=200,
        system=PythonLenia(final_step=200, scale_init_state=1.0),
        explorer=IMGEPExplorer(num_of_random_initialization=20),
        input_wrappers=[CppnInputWrapper('init_state')], # Starting from the explorer !
        output_representations=[LeniaImageSelector(), VAE(wrapped_input_space_key='image', encoder_n_latents=3, train_period=10, input_tensors_device="cuda")], # Starting from the system !
        on_discovery_callbacks=[CustomPrintCallback("Newly explored output !"), 
                                  OnDiscoverySaveCallbackOnDisk("./experiment_results/", 
                                                                to_save_outputs=[
                                                                    #"Parameters sent by the explorer before input wrappers",
                                                                    #"Parameters sent by the explorer after input wrappers",
                                                                    #"Raw system output",
                                                                    #"Representation of system output",
                                                                    "Rendered system output"
                                                                ])]
    )

    experiment.run(100)
