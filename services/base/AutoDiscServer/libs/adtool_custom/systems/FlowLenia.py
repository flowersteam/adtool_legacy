from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from auto_disc.auto_disc.systems.System import System
from auto_disc.auto_disc.utils.expose_config.defaults import Defaults, defaults
from flow_lenia.main import Config, FlowLenia, State, conn_from_matrix


@dataclass
class FlowLeniaParams(Defaults):
    SX: int = defaults(256, min=1, max=2048)
    SY: int = defaults(256, min=1, max=2048)
    nb_k: int = defaults(1, min=1, max=100)
    C: int = defaults(1, min=1, max=100)
    dd: int = defaults(5, min=1, max=100)


@FlowLeniaParams.expose_config()
class FlowLenia(System):
    def __init__(self, SX: int, SY: int, nb_k: int, C: int, dd: int):
        super().__init__()

        self.SX = SX
        self.SY = SY
        self.C = C
        self.nb_k = nb_k
        self.dd = dd

    def map(self, input: Dict) -> Dict:
        intermed_dict = deepcopy(input)
        param_tensor = intermed_dict["params"]

        nb_k = 10
        M = np.ones((C, C), dtype=int) * nb_k
        nb_k = int(M.sum())
        c0, c1 = conn_from_matrix(M)

        dt = param_tensor[0].item()
        theta_A = param_tensor[1].item()
        sigma = param_tensor[2].item()

        config = Config(
            SX=SX,
            SY=SY,
            nb_k=1,
            C=C,
            c0=c0,
            c1=c1,
            dt=dt,
            theta_A=theta_A,
            dd=dd,
            sigma=sigma,
        )

        fl = FlowLenia(config)
        roll_fn = jax.jit(fl.rollout_fn, static_argnums=(2,))

        # Initialize state and parameters
        seed = 9  # @param {type : "integer"}
        key = jax.random.PRNGKey(seed)
        params_seed, state_seed = jax.random.split(key)
        params = fl.rule_space.sample(params_seed)
        c_params = fl.kernel_computer(params)  # Process params

        mx, my = SX // 2, SY // 2  # center coordinated
        A0 = (
            jnp.zeros((SX, SY, C))
            .at[mx - 20 : mx + 20, my - 20 : my + 20, :]
            .set(jax.random.uniform(state_seed, (40, 40, C)))
        )
        state = State(A=A0)

        T = 500
        (_, orbit) = roll_fn(c_params, state, T)
        input["output"] = orbit

        return input

    def render(self, data_dict: dict) -> bytes:
        return b""
