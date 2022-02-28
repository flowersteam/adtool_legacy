import h5py
import os
import matplotlib.pyplot as plt
import torch
import yaml

from auto_disc.systems import BaseSystem
from auto_disc.utils.config_parameters import IntegerConfigParameter, StringConfigParameter
from auto_disc.utils.spaces import BoxSpace, DictSpace, MultiDiscreteSpace, MultiBinarySpace
from auto_disc.utils.spaces.utils import ConfigParameterBinding
from auto_disc.utils.misc.torch_utils import to_sparse_tensor
from auto_disc.utils.mutators import GaussianMutator

from matplotlib.animation import FuncAnimation
import io
import imageio

""" =============================================================================================
SimCells Main
============================================================================================= """

@IntegerConfigParameter(name="SX", default=256, min=1)
@IntegerConfigParameter(name="SY", default=256, min=1)
@IntegerConfigParameter(name="final_step", default=200, min=1, max=1000)
@IntegerConfigParameter(name="n_cells_type", default=2, min=1)
@IntegerConfigParameter(name="observations_export_interval", default=10, min=1)
@StringConfigParameter(name="observations_export_matrices", default="MatRender")
@StringConfigParameter(name="executable_folder", default="/home/mayalen/code/07-SimCells/resources/512/SimCells/bin/")
@StringConfigParameter(name="scs_template_filepath", default="/home/mayalen/code/09-AutoDisc/AutomatedDiscoveryTool/libs/auto_disc/input_wrappers/specific/simcells_config_template.scs")

class SimCells(BaseSystem):
    CONFIG_DEFINITION = {}

    input_space = DictSpace(
        update_rule=DictSpace(
            Cell1=DictSpace(
                durotaxis=DictSpace(
                    EltActAddECMDurotaxisForce=DictSpace(
                        Fmax=BoxSpace(low=0.0, high=1.0, shape=(), mutator=GaussianMutator(mean=0.0, std=0.05), indpb=1.0)
                    ))),
            Cell2=DictSpace(
                contraction=DictSpace(
                    EltActAddECMContractionForce=DictSpace(
                        intensity=BoxSpace(low=0.0, high=1.0, shape=(), mutator=GaussianMutator(mean=0.0, std=0.05), indpb=1.0)
                    ))),
        ),
        matnucleus_phenotype = BoxSpace(low=0, high=0, shape=(ConfigParameterBinding("SX"), ConfigParameterBinding("SY")), dtype=torch.int8)
    )

    output_space = DictSpace(
        MatRender=BoxSpace(low=0, high=255, shape=(ConfigParameterBinding("final_step") // ConfigParameterBinding("observations_export_interval"),
                                                     ConfigParameterBinding("SX"), ConfigParameterBinding("SY")))
        # TODO: "MatRender,MatNucleus,MatNucleusX,MatNucleusY,MatMembraneField"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #quick fix
        self.input_space["matnucleus_phenotype"].hight = self.config.n_cells_type
        self.config.observations_export_matrices = self.config.observations_export_matrices.split(",")
        for k in self.output_space.spaces.keys():
            if k not in self.config.observations_export_matrices:
                del self.output_space.spaces[k]

        self.observations_export_folder = f"simcells_h5_observations/experiment_{self.logger._AutoDiscLogger__experiment_id:06d}/seed_{self.logger._seed:06d}/" #last "/" important for simcells
        self.scs_export_folder = f"simcells_scs_parameters/experiment_{self.logger._AutoDiscLogger__experiment_id:06d}/seed_{self.logger._seed:06d}"
        self.scs_output_filepath = os.path.join(self.scs_export_folder, "run_<run_id>_simcellsconfig.scs")
        self.run_idx = 0




    def reset(self, run_parameters):

        # /!\ to read yaml the scs must have 4 spaces instead of tabs
        # /!\ characters like %, * must be between '' for the yaml to be read
        with open(self.config.scs_template_filepath, "r") as f:
            scs_template = yaml.safe_load(f)

        # insert system hyper-parameters
        scs_template['SX'] = self.config.SX
        scs_template['SY'] = self.config.SY
        scs_template['SZ'] = 1

        # insert update rule parameters in the template
        for protocell in scs_template['Model']['ProtoAgents']:
            if 'ProtoCell' in protocell.keys():
                protocell = protocell['ProtoCell']
                if protocell['name'] in run_parameters['update_rule'].keys():
                    for protocell_behavior in protocell['Behaviors']:
                        protocell_behavior = protocell_behavior['Behavior']
                        if protocell_behavior['name'] in run_parameters['update_rule'][protocell['name']].keys():
                            for protocell_behavior_elm in protocell_behavior['CdtsActs']:
                                elm_name = list(protocell_behavior_elm.keys())[0]
                                if elm_name in run_parameters['update_rule'][protocell['name']][
                                    protocell_behavior['name']].keys():
                                    for param_key, param_val in \
                                    run_parameters['update_rule'][protocell['name']][protocell_behavior['name']][
                                        elm_name].items():
                                        protocell_behavior_elm[elm_name][param_key] = float(
                                            "{:.3f}".format(param_val.item()))

        # insert initialization parameters in the template
        matnucleus_phenotype = to_sparse_tensor(run_parameters['matnucleus_phenotype'])
        nucleus_positions = matnucleus_phenotype.indices().t().int().contiguous()
        nb_nucleus = len(nucleus_positions)
        scs_template['NbNucleus'] = nb_nucleus
        nucleus_types = matnucleus_phenotype.values().unsqueeze(1)
        if len(nucleus_positions) > 0:
            scs_template['NucleusList'] = []
            for nucleus_idx, nucleus_pos in enumerate(nucleus_positions):
                cur_nucleus = {'x': int(nucleus_pos[0]), 'y': int(nucleus_pos[1]), 'z': 0,
                               'v': int(nucleus_types[nucleus_idx])}
                scs_template['NucleusList'].append({'Nucleus': cur_nucleus})

        scs_output_filepath = self.scs_output_filepath.replace("<run_id>", f"{self.run_idx:07d}")
        scs_output_folder = '/'.join(['.'] + scs_output_filepath.split('/')[:-1])
        if not os.path.exists(scs_output_folder):
            os.makedirs(scs_output_folder)
        with open(scs_output_filepath, "w") as f:
            yaml.dump(scs_template, f, sort_keys=False)

        # simcells not reading yaml as it, mush change spaces to tabs and remove quotes around characters
        with open(scs_output_filepath, "r") as f:
            yaml_lines = f.readlines()
        for line_idx, line in enumerate(yaml_lines):
            line = line.replace('  ', '\t')
            line = line.replace('- ', '\t- ')
            line = line.replace(' null', '')
            line = line.replace('\'%\'', '%')
            line = line.replace('\'*\'', '*')
            line = line.replace('\'>\'', '>')
            line = line.replace('Background2D:', 'Background2D: ')
            yaml_lines[line_idx] = line
        with open(scs_output_filepath, "w") as f:
            f.writelines(yaml_lines)

        self.step_idx = 0


    def step(self, intervention_parameters=None):

        if self.step_idx == 0: # run exectuable
            scs_output_filepath = self.scs_output_filepath.replace("<run_id>", f"{self.run_idx:07d}")
            assert (os.path.exists(scs_output_filepath))
            assert (os.path.exists(self.config.executable_folder))
            if not (os.path.exists(self.observations_export_folder)):
                os.makedirs(self.observations_export_folder)

            os.system(
                f"{os.path.join(self.config.executable_folder, 'core')} -noGUI -file {scs_output_filepath} "  # space important
                f"-exportInterval {self.config.observations_export_interval} -finalStep {self.config.final_step} "
                f"-exportMatrix {','.join(self.config.observations_export_matrices)} -exportPrefix {self.observations_export_folder} ")

            self._observations = {}
            # self._observations["timepoints"] = torch.zeros(self.config.final_step+1).bool()
            # for t in list(range(0, self.config.final_step + 1, self.config.observations_export_interval)):
            #     self._observations["timepoints"][t] = True
            timepoints = range(0, self.config.final_step + 1, self.config.observations_export_interval)

            for mat_name in self.config.observations_export_matrices:
                self._observations[mat_name] = torch.empty(len(timepoints), 1, self.config.SX, self.config.SY)
                for obs_idx, obs_timepoint in enumerate(timepoints):
                        obs_filename = os.path.join(self.observations_export_folder, f"{mat_name}-step-{obs_timepoint}.h5")
                        obs_file = h5py.File(obs_filename, "r")
                        self._observations[mat_name][obs_idx] = torch.tensor(obs_file['dset'])

        current_observation = {}
        for k, v in self._observations.items():
            current_observation[k] = v[self.step_idx]

        self.step_idx += 1

        return current_observation, 0, self.step_idx >= self.config.final_step // self.config.observations_export_interval, None

    def observe(self):
        return self._observations

    def render(self, mode="PIL_image"):

        im_array = []
        for matrender in self._observations["MatRender"]:
            matrender = matrender.int()
            bbb = ((matrender & 0x00FF0000) >> 16).float() / 255.0
            ggg = ((matrender & 0x0000FF00) >> 8).float() / 255.0
            rrr = ((matrender & 0x000000FF)).float() / 255.0
            color_array = torch.stack([bbb, ggg, rrr], dim=-1)
            color_array[(color_array == torch.tensor([0.0, 0.0, 0.0])).all(-1)] = 1.0  # N(=1)WHC
            #color_array = torch.cat([bbb, ggg, rrr], dim=0)
            #color_array[(color_array == 0.0).all(0).unsqueeze(0).repeat(3, 1, 1)] = 1.0
            im_array.append(color_array.squeeze())

        if mode == "human":
            fig = plt.figure(figsize=(4, 4))
            animation = FuncAnimation(fig, lambda frame: plt.imshow(frame), frames=im_array)
            plt.axis('off')
            plt.tight_layout()
            return plt.show()
        elif mode == "PIL_image":
            byte_img = io.BytesIO()
            imageio.mimwrite(byte_img, im_array, 'mp4', fps=30, output_params=["-f", "mp4"])
            return (byte_img, "mp4")
        else:
            raise NotImplementedError

    def close(self):
        pass
