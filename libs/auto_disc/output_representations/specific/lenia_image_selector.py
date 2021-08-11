from auto_disc.output_representations import BaseOutputRepresentation
from auto_disc.utils.config_parameters import IntegerConfigParameter
from auto_disc.utils.spaces.utils import ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace


@IntegerConfigParameter(name="timestep", default=-1)
@IntegerConfigParameter(name="SX", default=256, min=1)
@IntegerConfigParameter(name="SY", default=256, min=1)

class LeniaImageSelector(BaseOutputRepresentation):
    '''
        Selects a single image from Lenia states output.
    '''

    output_space = DictSpace(
        image=BoxSpace(low=0, high=10, shape=(ConfigParameterBinding("SX"), ConfigParameterBinding("SY"),))
    )

    def __init__(self, wrapped_input_space_key=None):
        super().__init__('states')

    def map(self, observations, is_output_new_discovery, **kwargs):
        return observations.states[self.config.timestep]
