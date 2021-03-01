from libs.utils.AttrDict import AttrDict

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
    The config and input space of the system are defined in CONFIG_DEFINITION and INPUT_SPACE_DEFINITION using:
        AutoDiscParameter(
                    name="dummy", 
                    type=int, 
                    values_range=[-1, 1], 
                    default=0)
    """

    CONFIG_DEFINITION = []
    INPUT_SPACE_DEFINITION = []

    @classmethod
    def get_default_config(cls):
        default_config = AttrDict()
        for param in cls.CONFIG_DEFINITION:
            param_dict = param.to_dict()
            default_config[param_dict['name']] = param_dict['default']
        
        return default_config

    def __init__(self, **kwargs):
        self.config = self.get_default_config()
        self.config.update(kwargs)

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
