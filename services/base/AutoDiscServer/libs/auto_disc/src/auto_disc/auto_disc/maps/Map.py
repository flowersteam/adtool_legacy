from auto_disc.utils.leaf.Leaf import Leaf
from auto_disc.utils.leaf.locators.locators import BlobLocator
from abc import ABCMeta, abstractmethod
from typing import Dict, Any


class Map(Leaf, metaclass=ABCMeta):
    """
    This is an abstract class that defines the interface for a `Map`, and should
    be inherited from by concrete implementations of `Map`s.

    A `Map` is a model for a function (in the mathematical sense, i.e., an 
    input-output relation) that is also stateful. It takes a payload of data as 
    input and returns an new payload of data as output, without mutating 
    the input.
    """

    @abstractmethod
    def __init__(self,
                 premap_key: str = "input",
                 postmap_key: str = "output"
                 ) -> None:
        """
        A `Map` minimally sets the premap and postmap keys. 

        #### Args
        - premap_key (str): key in the input dict for the input (i.e., "params")
        - postmap_key (str): key in the output dict for the output (i.e., "output")
        """
        super().__init__()
        # TODO: make this default, maybe use @property
        self.locator = BlobLocator()

    @abstractmethod
    def map(self, input: Dict) -> Dict:
        """
        A `Map` operates on regular Python dicts, but it views them as 
        structured. The `premap_key` and `postmap_key` are used to define
        the structured elements that the `Map` operates on. Often, the 
        `postmap_key` does not exist in the input dict, and is added by the
        `Map` as output.

        Whether or not the `premap_key` exists in the output dict is up to the
        implementation of the specific `Map`. We recommend preserving it.

        #### Args
        - input (dict): generic dict containing input data to the map at 
            `premap_key`
        #### Returns
        - output (dict): generic dict containing output data from the map at
            `postmap_key`
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> Any:
        """
        Curiosity exploration requires that the `Map` objects are agents in 
        some sense which have a memory of what data has passed through. The 
        way in which this is implemented in practice is that `Map` objects
        may generally be stateful, often by tracking the history of data that
        passes through. Then, they implement a `sample` method which returns
        a random sample of the latent representation they have encoded.

        #### Returns
        - sample (Any): a random sample of the latent representation, in 
            whatever format depending on the implementation of the `Map`
        """
        raise NotImplementedError
