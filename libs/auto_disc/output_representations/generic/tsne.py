from auto_disc.output_representations import BaseOutputRepresentation
import torch
from auto_disc.utils.config_parameters import StringConfigParameter, DecimalConfigParameter, IntegerConfigParameter
from auto_disc.utils.spaces.utils import distance, ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError
import numpy as np

@IntegerConfigParameter(name="n_components", default=3, min=1)
@DecimalConfigParameter(name="perplexity", default=30.0, min=5.0, max=50.0)
@StringConfigParameter(name="init", default="random", possible_values=["random", "pca"])
@IntegerConfigParameter(name="fit_period", default=10, min=1)

class TSNE(BaseOutputRepresentation):
    '''
    TSNE OutputRepresentation.
    '''
    output_space = DictSpace(
        embedding=BoxSpace(low=0, high=0, shape=(ConfigParameterBinding("n_components"),))
    )

    def __init__(self, wrapped_input_space_key=None):
        super().__init__(wrapped_input_space_key=wrapped_input_space_key)

        self.algorithm = make_pipeline(StandardScaler(),
                                       manifold.TSNE(n_components=self.config.n_components,
                                                     perplexity=self.config.perplexity,
                                                     init=self.config.init))

    def map(self, observations, is_output_new_discovery, **kwargs):
        input = observations.reshape(1,-1)
        try:
            output = self.algorithm.transform(input)
            output = torch.from_numpy(output).flatten()
        except NotFittedError as nfe:
            print("The TSNE instance is not fitted yet, returning null embedding")
            output = torch.zeros(self.output_space["embedding"].shape, dtype=self.output_space["embedding"].dtype)

        if (self.CURRENT_RUN_INDEX % self.config.fit_period == 0) and (
                self.CURRENT_RUN_INDEX > 0) and not is_output_new_discovery:
            self.fit_update()

        return output

    def fit_update(self):
        history = self._access_history()
        input_library = np.stack([history['input'][i].flatten() for i in range(len(history['input']))])
        self.algorithm.fit(input_library)
        self._call_output_history_update()

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        # L2 + add regularizer to avoid dead outcomes
        dist = distance.calc_l2(embedding_a, embedding_b)
        return dist
