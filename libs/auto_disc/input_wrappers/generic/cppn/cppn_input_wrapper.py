from addict import Dict
from auto_disc.utils.spaces import DictSpace
from copy import deepcopy
from auto_disc.input_wrappers.generic.cppn.utils import CPPNGenomeSpace
from auto_disc.input_wrappers.generic.cppn import pytorchneat
from auto_disc.input_wrappers import BaseInputWrapper
from auto_disc.utils.config_parameters import IntegerConfigParameter

@IntegerConfigParameter(name="n_passes", default=2, min=1)
class CppnInputWrapper(BaseInputWrapper):
    """ Base class to map the parameters sent by the explorer to the system's input space
    """

    input_space = DictSpace(
        genome = CPPNGenomeSpace()
    )

    def __init__(self, wrapped_output_space_key):
        super().__init__(wrapped_output_space_key)

    def map(self, parameters, **kwargs):
        cppn_genome = parameters['genome']
        initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(cppn_genome, self.input_space['genome'].neat_config)

        cppn_output_height = int(self.output_space[self.wrapped_output_space_key].shape[1])
        cppn_output_width = int(self.output_space[self.wrapped_output_space_key].shape[0])

        cppn_input = pytorchneat.utils.create_image_cppn_input((cppn_output_height, cppn_output_width), is_distance_to_center=True, is_bias=True)
        cppn_output = initialization_cppn.activate(cppn_input, self.config.n_passes)
        cppn_net_output = (1.0 - cppn_output.abs()).squeeze()

        parameters[self.wrapped_output_space_key] = cppn_net_output
        del parameters['genome']
        return parameters

        