from auto_disc.input_wrappers import BaseInputWrapper
from auto_disc.input_wrappers.generic.cppn import pytorchneat
from auto_disc.input_wrappers.generic.cppn.utils import CPPNGenomeSpace
from auto_disc.utils.config_parameters import DecimalConfigParameter, IntegerConfigParameter, StringConfigParameter, DictConfigParameter
from auto_disc.utils.spaces import DictSpace, BoxSpace, MultiBinarySpace
from auto_disc.utils.misc.torch_utils import PI
from auto_disc.utils.mutators import GaussianMutator

import math
import numbers
from os import path
import random
import torch

class BiasedMultiBinarySpace(MultiBinarySpace):

    def __init__(self, n, indpb_sample=1.0, indpb=1.0):

        if type(n) in [tuple, list, torch.tensor]:
            input_n = n
        else:
            input_n = (n,)
        if isinstance(indpb_sample, numbers.Number):
            indpb_sample = torch.full(input_n, indpb_sample, dtype=torch.float64)
        self.indpb_sample = torch.as_tensor(indpb_sample, dtype=torch.float64)

        MultiBinarySpace.__init__(self, n=n, indpb=indpb)

    def sample(self):
        return torch.bernoulli(self.indpb_sample).to(self.dtype)



@IntegerConfigParameter(name="n_cells_type", default=2, min=1)
@StringConfigParameter(name="neat_config_filepath", default=path.join(path.dirname(path.realpath(__file__)), "simcells_neat_config.cfg"))
@IntegerConfigParameter(name="cppn_n_passes", default=2, min=1)
@IntegerConfigParameter(name="drop_spacing", default=70)
@IntegerConfigParameter(name="drop_radius", default=20)
@IntegerConfigParameter(name="drop_ncells_min", default=0)
@IntegerConfigParameter(name="drop_ncells_max", default=20)
@DictConfigParameter(name="p_per_type", default={0: 0.8, 1:0.2})
@DecimalConfigParameter(name="p_collagen", default=0.005)
@IntegerConfigParameter(name="collagen_spacing", default=40)
@IntegerConfigParameter(name="collagen_radius", default=15)
class SimcellsMatnucleusInputWrapper(BaseInputWrapper):

    input_space = DictSpace(
        cell_pattern_genome = CPPNGenomeSpace(neat_config_filepath=path.join(path.dirname(path.realpath(__file__)), "simcells_neat_config.cfg")),
        collagen_concentration = BoxSpace(low=0.1, high=0.8, shape=(), mutator=GaussianMutator(mean=0.0, std=0.05), indpb=1.0),
        fibro_majority = BiasedMultiBinarySpace(n=1, indpb_sample=0.8, indpb=0.05),  # 1=majority fibro, 0=majority myofibro
    )


    def __init__(self, wrapped_output_space_key=None):
        super().__init__("matnucleus_phenotype")
        assert list(self.config.p_per_type.keys()) == list(range(self.config.n_cell_types))
        assert sum(self.config.p_per_type.values()) == 1.0

    def initialize(self, output_space):
        super().initialize(output_space)
        # quick fix
        import neat
        self.input_space["cell_pattern_genome"].neat_config = neat.Config(pytorchneat.selfconnectiongenome.SelfConnectionGenome,
                                       neat.DefaultReproduction,
                                       neat.DefaultSpeciesSet,
                                       neat.DefaultStagnation,
                                       self.config.neat_config_filepath
                                       )


    def map(self, parameters, is_input_new_discovery, **kwargs):
        initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(parameters["cell_pattern_genome"],
                                                                      self.input_space["cell_pattern_genome"].neat_config)

        output_shape = self.output_space[self.wrapped_output_space_key].shape
        SY = output_shape[0]
        SX = output_shape[1]
        output_shape = [o // self.config.drop_spacing for o in output_shape]
        cppn_input = pytorchneat.utils.create_image_cppn_input(output_shape, is_distance_to_center=True, is_bias=False)
        cppn_output = initialization_cppn.activate(cppn_input, self.config.cppn_n_passes)
        drop_presence = cppn_output[..., 0] > 0
        drop_density = (1.0 - cppn_output[..., 1].abs()) * (self.config.drop_ncells_max - self.config.drop_ncells_min) + self.config.drop_ncells_min

        nucleus_coords = torch.empty((0, 2))
        offset = self.config.drop_spacing // 2
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                if drop_presence[i, j]:
                    drop_n_cells = int(drop_density[i, j])
                    for cell_idx in range(drop_n_cells):
                        # uniform sampling in drop radius for now # see: https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
                        r = self.config.drop_radius * math.sqrt(random.random())
                        theta = random.random() * 2.0 * PI
                        cell_y = int(max(min(offset + i * self.config.drop_spacing + r * math.cos(theta), SY - 1), 0))
                        cell_x = int(max(min(offset + j * self.config.drop_spacing + r * math.sin(theta), SX- 1), 0))
                        v = torch.tensor([cell_y, cell_x])
                        if not (v == nucleus_coords).all(-1).any():
                            nucleus_coords = torch.cat([nucleus_coords, v.unsqueeze(0)])


        # collagen
        matnucleus_coords = torch.empty((0,2))
        matnucleus_feats = []
        offset = self.config.collagen_spacing // 2
        for i in range(SY // self.config.collagen_spacing):
            for j in range(SX // self.config.collagen_spacing):
                if torch.rand(()) < parameters.collagen_concentration: # spray of "collagen grains"
                    drop_n_grains = int(self.config.p_collagen * PI * (self.config.collagen_spacing/2.0)**2)
                    for cell_idx in range(drop_n_grains):
                        r = self.config.collagen_radius * math.sqrt(random.random())
                        theta = random.random() * 2.0 * PI
                        grain_y = int(max(min(offset + i * self.config.collagen_spacing + r * math.cos(theta), SY - 1), 0))
                        grain_x = int(max(min(offset + j * self.config.collagen_spacing + r * math.sin(theta), SX- 1), 0))
                        v = torch.tensor([grain_y, grain_x])
                        if not (v == nucleus_coords).all(-1).any() and not (v == matnucleus_coords).all(-1).any():
                            matnucleus_coords = torch.cat([matnucleus_coords, v.unsqueeze(0)])
                            matnucleus_feats.append(self.config.n_cell_types+1)

        matnucleus_feats = torch.tensor(matnucleus_feats).unsqueeze(1)

        if len(nucleus_coords) > 0:
            if bool(parameters.fibro_majority):
                sampling_weights = torch.tensor(self.config.p_per_type, dtype=torch.float)
            else:
                sampling_weights = torch.tensor(self.config.p_per_type[::-1], dtype=torch.float)
            matnucleus_coords = torch.cat([matnucleus_coords, nucleus_coords])
            matnucleus_feats = torch.cat([matnucleus_feats, (torch.multinomial(sampling_weights, len(nucleus_coords), replacement=True) + 1).char().unsqueeze(1)])

        phenotype_matnucleus = torch.sparse.IntTensor(matnucleus_coords.t().long(), matnucleus_feats.squeeze(), (SX,SY)).coalesce()
        phenotype_matnucleus = phenotype_matnucleus.to_dense()

        parameters[self.wrapped_output_space_key] = phenotype_matnucleus
        del parameters['cell_pattern_genome']
        del parameters['collagen_concentration']
        del parameters['fibro_majority']
        return parameters
