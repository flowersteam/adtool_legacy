from auto_disc.output_representations import BaseOutputRepresentation
import torch
from auto_disc.utils.config_parameters import StringConfigParameter, DecimalConfigParameter, IntegerConfigParameter
from auto_disc.utils.spaces.utils import distance, ConfigParameterBinding
from auto_disc.utils.spaces import DictSpace, BoxSpace
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError
import numpy as np

@IntegerConfigParameter(name="n_components", default=3, min=1)
@IntegerConfigParameter(name="n_neighbors", default=15, min=2, max=200)
@DecimalConfigParameter(name="min_dist", default=0.1, min=0.0, max=0.99)
@StringConfigParameter(name="init", default="spectral", possible_values=["spectral", ])
@StringConfigParameter(name="metric", default="euclidean", possible_values=["euclidean", ])
@IntegerConfigParameter(name="fit_period", default=10, min=1)

class UMAP(BaseOutputRepresentation):
    '''
    UMAP OutputRepresentation.
    '''
    output_space = DictSpace(
        embedding=BoxSpace(low=0, high=0, shape=(ConfigParameterBinding("n_components"),))
    )

    def __init__(self, wrapped_input_space_key=None):
        super().__init__(wrapped_input_space_key=wrapped_input_space_key)

        self.algorithm = make_pipeline(StandardScaler(),
                                       umap.UMAP(n_components=self.config.n_components,
                                                 n_neighbors=self.config.n_neighbors,
                                                 min_dist=self.config.min_dist,
                                                 init=self.config.init,
                                                 metric=self.config.metric,
                                                 ))

    def map(self, observations, is_output_new_discovery, **kwargs):
        input = observations.reshape(1,-1)
        try:
            output = self.algorithm.transform(input)
            output = torch.from_numpy(output).flatten()
        except NotFittedError as nfe:
            print("The UMAP instance is not fitted yet, returning null embedding")
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

