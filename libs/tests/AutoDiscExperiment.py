import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../"))

from auto_disc.systems.python_systems import PythonLenia, LeniaExpandedDiff
from auto_disc.systems.executable_systems import SimCells
from auto_disc.output_representations.specific import LeniaImageRepresentation, LeniaHandDefinedRepresentation, SimCellsMatRenderToRGB
from auto_disc.output_representations.generic import PCA, UMAP, VAE, HOLMES_VAE, SliceSelector
from auto_disc.input_wrappers.generic import TimesNInputWrapper, CppnInputWrapper
from auto_disc.input_wrappers.specific import SimcellsMatnucleusInputWrapper
from auto_disc.explorers import IMGEPExplorer

from auto_disc import ExperimentPipeline

from auto_disc.utils.callbacks import CustomPrintCallback
from auto_disc.utils.callbacks.on_discovery_callbacks import OnDiscoverySaveCallbackOnDisk
from auto_disc.utils.callbacks.on_save_callbacks import OnSaveModulesOnDiskCallback

from auto_disc.run import _set_seed

if __name__ == "__main__":
    seed = 42
    _set_seed(seed)
    representation_type = "VAE"
    load_checkpoint = False
    if representation_type == "VAE":
        representation = VAE(wrapped_input_space_key="slice_states",
                                               input_tensors_device="cuda",

                                               encoder_name="Burgess",
                                               encoder_n_latents=16,
                                               encoder_n_conv_layers=6,
                                               encoder_feature_layer=2,
                                               encoder_hidden_channels=16,
                                               encoder_hidden_dims=64,
                                               encoder_conditional_type="gaussian",

                                               weights_init_name="pytorch",
                                               #weights_init_checkpoint_filepath="./checkpoints/output_representations/exp_0_idx_39.pickle",
                                               #weights_init_checkpoint_keys="network_state_dict",

                                               loss_name="VAE",
                                               optimizer_name="Adam",

                                               tb_record_loss_frequency=1,
                                               tb_record_images_frequency=10,
                                               tb_record_embeddings_frequency=10,
                                               tb_record_memory_max=100,

                                                train_period=5,
                                                n_epochs_per_train_period=5,

                                                dataloader_batch_size=20,
                                                dataloader_num_workers=5,
                                                dataloader_drop_last=True,

                                                expand_output_space=True)


    elif representation_type=="HOLMES_VAE":
        representation = HOLMES_VAE(wrapped_input_space_key="slice_states",
                                               input_tensors_device="cuda",

                                               encoder_name="Burgess",
                                               encoder_n_latents=16,
                                               encoder_n_conv_layers=6,
                                               encoder_feature_layer=2,
                                               encoder_hidden_channels=16,
                                               encoder_hidden_dims=64,
                                               encoder_conditional_type="gaussian",

                                               weights_init_name="pytorch",

                                               loss_name="VAE",
                                               optimizer_name="Adam",

                                               tb_record_loss_frequency=1,
                                               tb_record_images_frequency=10,
                                               tb_record_embeddings_frequency=10,
                                               tb_record_memory_max=100,

                                               create_connections_lf=True,
                                               create_connections_gf=False,
                                               create_connections_gfi=True,
                                               create_connections_lfi=True,
                                               create_connections_recon=True,

                                               split_active=True,
                                               split_loss_key="recon",
                                               split_type="plateau",
                                               split_parameters_epsilon=1000,
                                               split_parameters_n_steps_average=50,
                                               n_min_epochs_before_split=5,
                                               n_min_epochs_between_splits=5,
                                               n_min_points_for_split=5,
                                               n_max_splits=10,

                                               boundary_name="cluster.KMeans",

                                               train_period=20,
                                               n_epochs_per_train_period=20,
                                               alternated_backward_active=True,
                                               alternated_backward_period=10,
                                               alternated_backward_connections=2,

                                               dataloader_batch_size=20,
                                               dataloader_num_workers=5,
                                               dataloader_drop_last=True,

                                               expand_output_space=True,
                                               )

    experiment = ExperimentPipeline(
        experiment_id=0,
        checkpoint_id=0,
        seed=seed,
        save_frequency=20,
        #system=PythonLenia(final_step=200, scale_init_state=1.0),
        #system=SimCells(final_step=20),
        system=LeniaExpandedDiff(final_step=200, scale_init_state=3),
        explorer=IMGEPExplorer(num_of_random_initialization=20),
        input_wrappers=[CppnInputWrapper("init_wall")], #SimcellsMatnucleusInputWrapper()],
        output_representations=[SliceSelector(wrapped_input_space_key="states"),
                                #SimCellsMatRenderToRGB(),
                                #SliceSelector(wrapped_input_space_key="matrender_rgb"),
                                representation],
        on_discovery_callbacks=[CustomPrintCallback("Newly explored output !"), 
                                  OnDiscoverySaveCallbackOnDisk("./experiment_results/", 
                                                                to_save_outputs=[
                                                                    #"Parameters sent by the explorer before input wrappers",
                                                                    #"Parameters sent by the explorer after input wrappers",
                                                                    #"Raw system output",
                                                                    #"Representation of system output",
                                                                    "Rendered system output"
                                                                ])],
        on_save_callbacks=[OnSaveModulesOnDiskCallback("./checkpoints/")]
    )

    if load_checkpoint:
        import pickle
        with open("./checkpoints/output_representations/exp_0_idx_39.pickle", "rb") as f:
            representation_dict = pickle.load(f)
        experiment._output_representations[-1].load(representation_dict[-1])

    experiment.run(100)
