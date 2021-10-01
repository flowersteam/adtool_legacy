from addict import Dict
from auto_disc.utils.spaces import DictSpace
from auto_disc.input_wrappers.generic.cppn.utils import CPPNGenomeSpace
from auto_disc.input_wrappers.generic.cppn import pytorchneat
from auto_disc.input_wrappers import BaseInputWrapper
from auto_disc.utils.config_parameters import IntegerConfigParameter, StringConfigParameter
from auto_disc.utils.spaces.utils import ConfigParameterBinding


@StringConfigParameter(name="neat_config_filepath", default="./config.cfg", possible_values="all")
@IntegerConfigParameter(name="n_passes", default=2, min=1)

class CppnInputWrapper(BaseInputWrapper):
    """ Base class to map the parameters sent by the explorer to the system's input space
    """
    CONFIG_DEFINITION = {}
    config = Dict()

    input_space = DictSpace(
        genome=CPPNGenomeSpace(ConfigParameterBinding("neat_config_filepath"))
    )

    def __init__(self, wrapped_output_space_key):
        super().__init__(wrapped_output_space_key)

    def map(self, parameters, is_input_new_discovery, **kwargs):
        cppn_genome = parameters['genome']
        initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(cppn_genome, self.input_space['genome'].neat_config)

        output_shape = self.output_space[self.wrapped_output_space_key].shape
        cppn_input = pytorchneat.utils.create_image_cppn_input(output_shape, is_distance_to_center=True, is_bias=True)
        cppn_output = initialization_cppn.activate(cppn_input, self.config.n_passes)
        cppn_net_output = (1.0 - cppn_output.abs()).squeeze()  #TODO: allow other mapping of cppn output in config

        # TODO: why that? what if we need several CPPN wrappers in parallel (different keys than "genome")
        parameters[self.wrapped_output_space_key] = cppn_net_output
        del parameters['genome']
        return parameters

        