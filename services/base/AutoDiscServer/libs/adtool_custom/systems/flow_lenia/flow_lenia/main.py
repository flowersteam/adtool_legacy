import os
import typing as t
from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import moviepy.editor as mvp
import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


os.environ['FFMPEG_BINARY'] = 'ffmpeg'


class VideoWriter:
    def __init__(self, filename, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()

    def show(self, **kw):
        self.close()
        fn = self.params['filename']
        display(mvp.ipython_display(fn, **kw))


def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)


def ker_f(x, a, w, b): return (
    b * jnp.exp(- (x[..., None] - a)**2 / w)).sum(-1)


def bell(x, m, s): return jnp.exp(-((x - m) / s)**2 / 2)


def growth(U, m, s):
    return bell(U, m, s) * 2 - 1


kx = jnp.array([
    [1., 0., -1.],
    [2., 0., -2.],
    [1., 0., -1.]
])
ky = jnp.transpose(kx)


def sobel_x(A):
    """
    A : (x, y, c)
    ret : (x, y, c)
    """
    return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], kx, mode='same')
                       for c in range(A.shape[-1])])


def sobel_y(A):
    return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], ky, mode='same')
                       for c in range(A.shape[-1])])


@jax.jit
def sobel(A):
    return jnp.concatenate(
        (sobel_y(A)[
            :, :, None, :], sobel_x(A)[
            :, :, None, :]), axis=2)


def get_kernels(SX: int, SY: int, nb_k: int, params):
    mid = SX // 2
    Ds = [np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) /
          ((params['R'] + 15) * params['r'][k]) for k in range(nb_k)]  # (x,y,k)
    K = jnp.dstack([sigmoid(-(D - 1) * 10) * ker_f(D, params["a"][k],
                   params["w"][k], params["b"][k]) for k, D in zip(range(nb_k), Ds)])
    nK = K / jnp.sum(K, axis=(0, 1), keepdims=True)
    return nK


def conn_from_matrix(mat):
    C = mat.shape[0]
    c0 = []
    c1 = [[] for _ in range(C)]
    i = 0
    for s in range(C):
        for t in range(C):
            n = mat[s, t]
            if n:
                c0 = c0 + [s] * n
                c1[t] = c1[t] + list(range(i, i + n))
            i += n
    return c0, c1


def conn_from_lists(c0, c1, C):
    return c0, [[i == c1[i] for i in range(len(c0))] for c in range(C)]


class ReintegrationTracking:

    def __init__(
            self,
            SX=256,
            SY=256,
            dt=.2,
            dd=5,
            sigma=.65,
            border="wall",
            has_hidden=False,
            hidden_dims=None,
            mix="softmax"):
        self.SX = SX
        self.SY = SY
        self.dt = dt
        self.dd = dd
        self.sigma = sigma
        self.has_hidden = has_hidden
        self.hidden_dims = hidden_dims
        self.border = border if border in ['wall', 'torus'] else 'wall'
        self.mix = mix

        self.apply = self._build_apply()

    def __call__(self, *args):
        return self.apply(*args)

    def _build_apply(self):

        x, y = jnp.arange(self.SX), jnp.arange(self.SY)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + .5  # (SX, SY, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd + 1):
            for dy in range(-dd, dd + 1):
                dxs.append(dx)
                dys.append(dy)
        dxs = jnp.array(dxs)
        dys = jnp.array(dys)
        # -----------------------------------------------------------------------------------------------
        if not self.has_hidden:

            @partial(jax.vmap, in_axes=(None, None, 0, 0))
            def step(X, mu, dx, dy):
                Xr = jnp.roll(X, (dx, dy), axis=(0, 1))
                mur = jnp.roll(mu, (dx, dy), axis=(0, 1))
                if self.border == 'torus':
                    dpmu = jnp.min(jnp.stack(
                        [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None]))
                         for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                    ), axis=0)
                else:
                    dpmu = jnp.absolute(pos[..., None] - mur)
                sz = .5 - dpmu + self.sigma
                area = jnp.prod(jnp.clip(sz, 0, min(
                    1, 2 * self.sigma)), axis=2) / (4 * self.sigma**2)
                nX = Xr * area
                return nX

            def apply(X, F):

                ma = self.dd - self.sigma  # upper bound of the flow maggnitude
                # (x, y, 2, c) : target positions (distribution centers)
                mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma)
                if self.border == "wall":
                    mu = jnp.clip(mu, self.sigma, self.SX - self.sigma)
                nX = step(X, mu, dxs, dys).sum(axis=0)

                return nX
        # -----------------------------------------------------------------------------------------------
        else:

            @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
            def step_flow(X, H, mu, dx, dy):
                """Summary
                """
                Xr = jnp.roll(X, (dx, dy), axis=(0, 1))
                Hr = jnp.roll(H, (dx, dy), axis=(0, 1))  # (x, y, k)
                mur = jnp.roll(mu, (dx, dy), axis=(0, 1))

                if self.border == 'torus':
                    dpmu = jnp.min(jnp.stack(
                        [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None]))
                         for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                    ), axis=0)
                else:
                    dpmu = jnp.absolute(pos[..., None] - mur)

                sz = .5 - dpmu + self.sigma
                area = jnp.prod(jnp.clip(sz, 0, min(
                    1, 2 * self.sigma)), axis=2) / (4 * self.sigma**2)
                nX = Xr * area
                return nX, Hr

            def apply(X, H, F):

                ma = self.dd - self.sigma  # upper bound of the flow maggnitude
                # (x, y, 2, c) : target positions (distribution centers)
                mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma)
                if self.border == "wall":
                    mu = jnp.clip(mu, self.sigma, self.SX - self.sigma)
                nX, nH = step_flow(X, H, mu, dxs, dys)

                if self.mix == 'avg':
                    nH = jnp.sum(nH * nX.sum(axis=-1, keepdims=True), axis=0)
                    nX = jnp.sum(nH, axis=0)
                    nH = nH / (nX.sum(axis=-1, keepdims=True) + 1e-10)

                elif self.mix == "softmax":
                    expnX = jnp.exp(nX.sum(axis=-1, keepdims=True)) - 1
                    nX = jnp.sum(nX, axis=0)
                    nH = jnp.sum(nH * expnX, axis=0) / \
                        (expnX.sum(axis=0) + 1e-10)  # avg rule

                elif self.mix == "stoch":
                    categorical = jax.random.categorical(
                        jax.random.PRNGKey(42),
                        jnp.log(nX.sum(axis=-1, keepdims=True)),
                        axis=0)
                    mask = jax.nn.one_hot(
                        categorical, num_classes=(2 * self.dd + 1)**2, axis=-1)
                    mask = jnp.transpose(mask, (3, 0, 1, 2))
                    nH = jnp.sum(nH * mask, axis=0)
                    nX = jnp.sum(nX, axis=0)

                elif self.mix == "stoch_gene_wise":
                    mask = jnp.concatenate(
                        [jax.nn.one_hot(jax.random.categorical(
                            jax.random.PRNGKey(42),
                            jnp.log(nX.sum(axis=-1, keepdims=True)),
                            axis=0),
                            num_classes=(2 * dd + 1)**2, axis=-1)
                         for _ in range(self.hidden_dims)],
                        axis=2)
                    # (2dd+1**2, x, y, nb_k)
                    mask = jnp.transpose(mask, (3, 0, 1, 2))
                    nH = jnp.sum(nH * mask, axis=0)
                    nX = jnp.sum(nX, axis=0)

                return nX, nH

        return apply


@chex.dataclass
class Params:
    """Flow Lenia update rule parameters
    """
    r: jnp.ndarray
    b: jnp.ndarray
    w: jnp.ndarray
    a: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray
    R: float


@chex.dataclass
class CompiledParams:
    """Flow Lenia compiled parameters
    """
    fK: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray


class RuleSpace:

    """Rule space for Flow Lenia system

    Attributes:
        kernel_keys (TYPE): Description
        nb_k (int): number of kernels of the system
        spaces (TYPE): Description
    """

    # -----------------------------------------------------------------------------
    def __init__(self, nb_k: int):
        """
        Args:
            nb_k (int): number of kernels in the update rule
        """
        self.nb_k = nb_k
        self.kernel_keys = 'r b w a m s h'.split()
        self.spaces = {
            "r": {'low': .2, 'high': 1., 'mut_std': .2, 'shape': None},
            "b": {'low': .001, 'high': 1., 'mut_std': .2, 'shape': (3,)},
            "w": {'low': .01, 'high': .5, 'mut_std': .2, 'shape': (3,)},
            "a": {'low': .0, 'high': 1., 'mut_std': .2, 'shape': (3,)},
            "m": {'low': .05, 'high': .5, 'mut_std': .2, 'shape': None},
            "s": {'low': .001, 'high': .18, 'mut_std': .01, 'shape': None},
            "h": {'low': .01, 'high': 1., 'mut_std': .2, 'shape': None},
            'R': {'low': 2., 'high': 25., 'mut_std': .2, 'shape': None},
        }
    # -----------------------------------------------------------------------------

    def sample(self, key: jnp.ndarray) -> Params:
        """sample a random set of parameters

        Returns:
            Params: sampled parameters

        Args:
            key (jnp.ndarray): random generation key
        """
        kernels = {}
        for k in 'rmsh':
            key, subkey = jax.random.split(key)
            kernels[k] = jax.random.uniform(
                key=subkey,
                minval=self.spaces[k]['low'],
                maxval=self.spaces[k]['high'],
                shape=(
                    self.nb_k,
                ))
        for k in "awb":
            key, subkey = jax.random.split(key)
            kernels[k] = jax.random.uniform(
                key=subkey,
                minval=self.spaces[k]['low'],
                maxval=self.spaces[k]['high'],
                shape=(
                    self.nb_k,
                    3))
        R = jax.random.uniform(
            key=key,
            minval=self.spaces['R']['low'],
            maxval=self.spaces['R']['high'])
        return Params(R=R, **kernels)


class KernelComputer:

    """Summary

    Attributes:
        apply (Callable): main function transforming raw params (Params) in copmiled ones (CompiledParams)
        SX (int): X size
        SY (int): Y size
    """

    def __init__(self, SX: int, SY: int, nb_k: int):
        """Summary

        Args:
            SX (int): Description
            SY (int): Description
            nb_k (int): Description
        """
        self.SX = SX
        self.SY = SY

        mid = SX // 2

        def compute_kernels(params: Params) -> CompiledParams:
            """Compute kernels and return a dic containing kernels fft

            Args:
                params (Params): raw params of the system

            Returns:
                CompiledParams: compiled params which can be used as update rule
            """

            Ds = [np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) /
                  ((params.R + 15) * params.r[k]) for k in range(nb_k)]  # (x,y,k)
            K = jnp.dstack([sigmoid(-(D - 1) * 10) * ker_f(D, params.a[k],
                           params.w[k], params.b[k]) for k, D in zip(range(nb_k), Ds)])
            # Normalize kernels
            nK = K / jnp.sum(K, axis=(0, 1), keepdims=True)
            fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0, 1)),
                              axes=(0, 1))  # Get kernels fft

            return CompiledParams(fK=fK, m=params.m, s=params.s, h=params.h)

        self.apply = jax.jit(compute_kernels)

    def __call__(self, params: Params):
        """callback to apply
        """
        return self.apply(params)


# ==================================================================================================================
# ==================================================FLOW LENIA============
# ==================================================================================================================

@chex.dataclass
class Config:

    """Configuration of Flow Lenia system
    """
    SX: int
    SY: int
    nb_k: int
    C: int
    c0: t.Iterable
    c1: t.Iterable
    dt: float
    dd: int = 5
    sigma: float = .65
    n: int = 2
    theta_A: float = 1.
    border: str = 'wall'


@chex.dataclass
class State:

    """State of the system
    """
    A: jnp.ndarray


class FlowLenia:

    """class building the main functions of Flow Lenia

    Attributes:
        config (FL_Config): config of the system
        kernel_computer (KernelComputer): kernel computer
        rollout_fn (Callable): rollout function
        RT (ReintegrationTracking): Description
        rule_space (RuleSpace): Rule space of the system
        step_fn (Callable): system step function
    """

    # ------------------------------------------------------------------------------

    def __init__(self, config: Config):
        """
        Args:
            config (Config): config of the system
        """
        self.config = config

        self.rule_space = RuleSpace(config.nb_k)

        self.kernel_computer = KernelComputer(
            self.config.SX, self.config.SY, self.config.nb_k)

        self.RT = ReintegrationTracking(
            self.config.SX,
            self.config.SY,
            self.config.dt,
            self.config.dd,
            self.config.sigma,
            self.config.border)

        self.step_fn = self._build_step_fn()

        self.rollout_fn = self._build_rollout()

    # ------------------------------------------------------------------------------

    def __call__(self, state: State, params: CompiledParams) -> State:
        """callback to step function

        Args:
            state (State): Description
            params (CompiledParams): Description

        Returns:
            State: Description
        """
        return self.step_fn(state, params)

    # ------------------------------------------------------------------------------

    def _build_step_fn(self) -> t.Callable[[State, CompiledParams], State]:
        """Build step function of the system according to config

        Returns:
            t.Callable[[State, CompiledParams], State]: step function which outputs next state
            given a state and params
        """

        def step(state: State, params: CompiledParams) -> State:
            """
            Main step

            Args:
                state (State): state of the system
                params (CompiledParams): params

            Returns:
                State: new state of the system

            """
            # ---------------------------Original Lenia------------------------
            A = state.A

            fA = jnp.fft.fft2(A, axes=(0, 1))  # (x,y,c)

            fAk = fA[:, :, self.config.c0]  # (x,y,k)

            U = jnp.real(jnp.fft.ifft2(
                params.fK * fAk, axes=(0, 1)))  # (x,y,k)

            U = growth(U, params.m, params.s) * params.h  # (x,y,k)

            U = jnp.dstack([U[:, :, self.config.c1[c]].sum(axis=-1)
                           for c in range(self.config.C)])  # (x,y,c)

            # -------------------------------FLOW------------------------------------------

            nabla_U = sobel(U)  # (x, y, 2, c)

            nabla_A = sobel(A.sum(axis=-1, keepdims=True))  # (x, y, 2, 1)

            alpha = jnp.clip((A[:, :, None, :] /
                              self.config.theta_A)**self.config.n, .0, 1.)

            F = nabla_U * (1 - alpha) - nabla_A * alpha

            nA = self.RT.apply(A, F)

            return State(A=nA)

        return step

    # ------------------------------------------------------------------------------

    def _build_rollout(
            self) -> t.Callable[[CompiledParams, State, int], t.Tuple[State, State]]:
        """build rollout function

        Returns:
            t.Callable[[CompiledParams, State, int], t.Tuple[State, State]]: Description
        """
        def scan_step(carry: t.Tuple[State, CompiledParams],
                      x) -> t.Tuple[t.Tuple[State, CompiledParams], State]:
            """Summary

            Args:
                carry (t.Tuple[State, CompiledParams]): Description
                x (TYPE): Description

            Returns:
                t.Tuple[t.Tuple[State, CompiledParams], State]: Description
            """
            state, params = carry
            nstate = jax.jit(self.step_fn)(state, params)
            return (nstate, params), nstate

        def rollout(params: CompiledParams, init_state: State,
                    steps: int) -> t.Tuple[State, State]:
            """Summary

            Args:
                params (CompiledParams): Description
                init_state (State): Description
                steps (int): Description

            Returns:
                t.Tuple[State, State]: Description
            """
            return jax.lax.scan(
                scan_step, (init_state, params), None, length=steps)

        return rollout


class Lenia:

    """class building the main functions of Flow Lenia

    Attributes:
        config (FL_Config): config of the system
        kernel_computer (KernelComputer): kernel computer
        rollout_fn (Callable): rollout function
        RT (ReintegrationTracking): Description
        rule_space (RuleSpace): Rule space of the system
        step_fn (Callable): system step function
    """

    # ------------------------------------------------------------------------------

    def __init__(self, config: Config):
        """
        Args:
            config (Config): config of the system
        """
        self.config = config

        self.rule_space = RuleSpace(config.nb_k)

        self.kernel_computer = KernelComputer(
            self.config.SX, self.config.SY, self.config.nb_k)

        self.RT = ReintegrationTracking(
            self.config.SX,
            self.config.SY,
            self.config.dt,
            self.config.dd,
            self.config.sigma,
            self.config.border)

        self.step_fn = self._build_step_fn()

        self.rollout_fn = self._build_rollout()

    # ------------------------------------------------------------------------------

    def __call__(self, state: State, params: CompiledParams) -> State:
        """callback to step function

        Args:
            state (State): Description
            params (CompiledParams): Description

        Returns:
            State: Description
        """
        return self.step_fn(state, params)

    # ------------------------------------------------------------------------------

    def _build_step_fn(self) -> t.Callable[[State, CompiledParams], State]:
        """Build step function of the system according to config

        Returns:
            t.Callable[[State, CompiledParams], State]: step function which outputs next state
            given a state and params
        """

        def step(state: State, params: CompiledParams) -> State:
            """
            Main step

            Args:
                state (State): state of the system
                params (CompiledParams): params

            Returns:
                State: new state of the system

            """
            # ---------------------------Original Lenia------------------------
            A = state.A

            fA = jnp.fft.fft2(A, axes=(0, 1))  # (x,y,c)

            fAk = fA[:, :, self.config.c0]  # (x,y,k)

            U = jnp.real(jnp.fft.ifft2(
                params.fK * fAk, axes=(0, 1)))  # (x,y,k)

            U = growth(U, params.m, params.s) * params.h  # (x,y,k)

            U = jnp.dstack([U[:, :, self.config.c1[c]].sum(axis=-1)
                           for c in range(self.config.C)])  # (x,y,c)

            # -------------------------------FLOW------------------------------------------

            nA = jnp.clip(A + self.config.dt * U, 0., 1.)

            return State(A=nA)

        return step

    # ------------------------------------------------------------------------------

    def _build_rollout(
            self) -> t.Callable[[CompiledParams, State, int], t.Tuple[State, State]]:
        """build rollout function

        Returns:
            t.Callable[[CompiledParams, State, int], t.Tuple[State, State]]: Description
        """
        def scan_step(carry: t.Tuple[State, CompiledParams],
                      x) -> t.Tuple[t.Tuple[State, CompiledParams], State]:
            """Summary

            Args:
                carry (t.Tuple[State, CompiledParams]): Description
                x (TYPE): Description

            Returns:
                t.Tuple[t.Tuple[State, CompiledParams], State]: Description
            """
            state, params = carry
            nstate = jax.jit(self.step_fn)(state, params)
            return (nstate, params), nstate

        def rollout(params: CompiledParams, init_state: State,
                    steps: int) -> t.Tuple[State, State]:
            """Summary

            Args:
                params (CompiledParams): Description
                init_state (State): Description
                steps (int): Description

            Returns:
                t.Tuple[State, State]: Description
            """
            return jax.lax.scan(
                scan_step, (init_state, params), None, length=steps)

        return rollout
# @title Utils


def state2img(A):
    C = A.shape[-1]
    if C == 1:
        return A[..., 0]
    if C == 2:
        return np.dstack([A[..., 0], A[..., 0], A[..., 1]])
    return A[..., :3]


def main():
    # Configuration
    number_of_kernels = 10  # @param {type:"raw"}
    nb_k = number_of_kernels
    world_size = 128  # @param {type : "integer"}
    SX = SY = world_size
    C = 1  # @param {type : "integer"}
    dt = 0.2  # @param
    theta_A = 2.0  # @param
    sigma = 0.65  # @param
    M = np.ones((C, C), dtype=int) * nb_k
    nb_k = int(M.sum())
    c0, c1 = conn_from_matrix(M)
    config = Config(SX=SX, SY=SY, nb_k=nb_k, C=C, c0=c0, c1=c1,
                    dt=dt, theta_A=theta_A, dd=5, sigma=sigma)
    fl = FlowLenia(config)
    roll_fn = jax.jit(fl.rollout_fn, static_argnums=(2,))

    # Initialize state and parameters
    seed = 9  # @param {type : "integer"}
    key = jax.random.PRNGKey(seed)
    params_seed, state_seed = jax.random.split(key)
    params = fl.rule_space.sample(params_seed)
    c_params = fl.kernel_computer(params)  # Process params

    mx, my = SX // 2, SY // 2  # center coordinated
    A0 = jnp.zeros((SX, SY, C)).at[mx - 20:mx + 20, my - 20:my + 20, :].set(
        jax.random.uniform(state_seed, (40, 40, C))
    )
    state = State(A=A0)

    # Collect rollout and visualize
    T = 500
    (final_state, _), states = roll_fn(c_params, state, T)

    with VideoWriter("example.mp4", 60) as vid:
        for i in range(T):
            vid.add(state2img(states.A[i]))
    assert 1


if __name__ == "__main__":
    main()
