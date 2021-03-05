from libs.utils.AttrDict import AttrDict
from libs.utils.auto_disc_parameters.AutoDiscParameter import get_default_values

class BaseSystem():
    """The main AbstractSystem class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        reset
        step
        render
        close
        #TODO: seed
        #TODO: compute_reward
    The config and input space of the system are defined in CONFIG_DEFINITION, INPUT_SPACE_DEFINITION and OUTPUT_SPACE_DEFINITION using:
        AutoDiscParameter(
                    name="dummy", 
                    type=int, 
                    values_range=[-1, 1], 
                    default=0)
    """

    CONFIG_DEFINITION = []
    INPUT_SPACE_DEFINITION = []
    OUTPUT_SPACE_DEFINITION = []
    STEP_OUTPUT_SPACE_DEFINITION = []

    def __init__(self, config_kwargs={}, input_space_kwargs={}, output_space_kwargs={}, step_output_space_kwargs={}):
        self.config = get_default_values(self, self.CONFIG_DEFINITION)
        self.config.update(config_kwargs)

        self.input_space = get_default_values(self, self.INPUT_SPACE_DEFINITION)
        self.input_space.update(input_space_kwargs)

        self.output_space = get_default_values(self, self.OUTPUT_SPACE_DEFINITION)
        self.output_space.update(output_space_kwargs)

        self.step_output_space = get_default_values(self, self.STEP_OUTPUT_SPACE_DEFINITION)
        self.step_output_space.update(step_output_space_kwargs)

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
