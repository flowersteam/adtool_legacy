from addict import Dict
from auto_disc.utils.spaces import DictSpace

class BaseSystem():
    """The main BaseSystem class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        reset
        step
        observe
        render
        close
    The config, input space as well as output space of the system are defined in CONFIG_DEFINITION, INPUT_SPACE_DEFINITION and OUTPUT_SPACE_DEFINITION using:
        `AutoDiscParameter(
                    name="dummy", 
                    type=int, 
                    values_range=[-1, 1], 
                    default=0)`
    For space parameters, use
        `AutoDiscParameter(
                    name="dummu", 
                    type=ParameterTypesEnum.get('SPACE'),
                    default=AutoDiscSpaceDefinition(
                        dims=[10, 10, 3],
                        bounds=[0, 1],
                        type=ParameterTypesEnum.get('FLOAT')
                    ))`
    Non-modifiable parameters should set
        `modifiable=False`
    For scalar spaces, set
        `dims=[]`
    """

    config = Dict()
    input_space = DictSpace()
    output_space = DictSpace()
    step_output_space = DictSpace()

    def __init__(self):
        self.input_space.initialize(self)
        self.output_space.initialize(self)
        self.step_output_space.initialize(self)

    def reset(self, run_parameters):
        """Resets the environment to an initial state and returns an initial
        observation.
        Args:
            run_parameters (AttrDict): the input parameters of the system provided by the agent
        Returns:
            observation (AttrDict): the initial observation.
        """
        raise NotImplementedError

    def step(self, action=None):
        """Run one timestep of the system's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (AttrDict): an action provided by the agent
        Returns:
            observation (AttrDict): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (AttrDict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def observe(self):
        """
        Returns the overall output of the system according to the last `reset()`.
        Use this function once the step function has returned `done=True` to give the system's output to the Output Representation (and then the explorer).
        """
        raise NotImplementedError

    def render(self, **kwargs):
        """Renders the environment.
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass
