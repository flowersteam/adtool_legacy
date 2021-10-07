from addict import Dict
from auto_disc.utils.spaces import DictSpace, BoxSpace
from auto_disc.input_wrappers.generic.cppn.utils import CPPNGenomeSpace
from auto_disc.input_wrappers.generic.cppn import pytorchneat
from auto_disc.input_wrappers import BaseInputWrapper
from auto_disc.utils.config_parameters import IntegerConfigParameter, StringConfigParameter
from auto_disc.utils.spaces.utils import ConfigParameterBinding


#TODO: @StringConfigParameter(name="neat_config_filepath", default="config.cfg", possible_values="all")
@IntegerConfigParameter(name="n_passes", default=2, min=1)

class CppnInputWrapper(BaseInputWrapper):
    input_space = DictSpace(
        genome=CPPNGenomeSpace(
            neat_config_filepath="/home/mayalen/code/09-AutoDisc/AutomatedDiscoveryTool/libs/auto_disc/input_wrappers/generic/cppn/config.cfg"
        )
    )

    def initialize(self, output_space):
        super().initialize(output_space)
        self.input_space[f"genome_{self.wrapped_output_space_key}"] = self.input_space.spaces.pop("genome")

    def map(self, parameters, is_input_new_discovery, **kwargs):
        cppn_genome = parameters[f"genome_{self.wrapped_output_space_key}"]
        initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(cppn_genome, self.input_space[f"genome_{self.wrapped_output_space_key}"].neat_config)

        output_shape = self.output_space[self.wrapped_output_space_key].shape
        cppn_input = pytorchneat.utils.create_image_cppn_input(output_shape, is_distance_to_center=True, is_bias=True)
        cppn_output = initialization_cppn.activate(cppn_input, self.config.n_passes)
        cppn_net_output = (1.0 - cppn_output.abs()).squeeze(-1)  #TODO: allow other mapping of cppn output in config

        parameters[self.wrapped_output_space_key] = cppn_net_output
        del parameters[f"genome_{self.wrapped_output_space_key}"]
        return parameters

        