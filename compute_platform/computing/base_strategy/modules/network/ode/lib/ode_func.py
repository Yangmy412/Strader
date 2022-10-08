import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Laplace, kl_divergence
from typing import Union, Optional
from torchsde import SDEIto


from .func import divergence_bf, divergence_approx, sample_rademacher_like, sample_gaussian_like, \
    divergence_bf_aug, divergence_approx_aug

# from computing.base_strategy.modules.network.ode.lib.ode_net import AutoencoderDiffEqNet

from computing.base_strategy.modules.network.ode.lib.ode_net import ODEnet

__all__ = ["ODEfunc", "CNFODEfunc", "AugCNFODEfunc", "LatentSDEFunc"]


class ODEfunc(nn.Module):
    """
    包装出一个ode function，用于作为odeint的function输入。主要是完成偏导工作。
    """

    def __init__(self, diffeq: Optional[ODEnet]):
        super(ODEfunc, self).__init__()

        self.diffeq = diffeq

        self.register_buffer("_num_evals", torch.tensor(0.))

    def num_evals(self):
        """

        Returns: 调用的次数，用于计算nfe的值

        """
        return self._num_evals.item()

    def forward(self, t, states):
        """

        Args:
            t:
            states: y

        Returns: dy_dt

        """

        return self.diffeq(states)


class CNFODEfunc(ODEfunc):
    """
    包装出一个用于cnf的ode function，用于作为odeint的function输入。主要是完成偏导和trace的工作。
    Link: https://github.com/rtqichen/ffjord
    """

    def __init__(self, diffeq:ODEnet, divergence_fn="approximate", residual=False, rademacher=False):
        """
        @param diffeq: 一个用于拟合常微分方程的网络
        @param divergence_fn: 选择trace的计算方法 [brute_force, approximate]
        @param residual:
        @param rademacher: 专门用于approximate的trace计算方法
                           if true，从rademacher分布中sample noise (self._e)
                           if false，从gaussian分布中sample noise
        """
        super(CNFODEfunc, self).__init__(diffeq)
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = basic_net.wrappers.diffeq_wrapper(diffeq)
        self.residual = residual
        self.rademacher = rademacher
        self._e = None

        # select function for trace calculate
        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def sample_for_hutchinson_estimator(self, y):
        """
        Args:
            y:

        Returns:

        """
        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
            else:
                self._e = sample_gaussian_like(y)

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]
        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            self._e = torch.zeros_like(y)
            if isinstance(self.effective_shape, int):
                sample_like = y[:, : self.effective_shape]
            else:
                sample_like = y
                for dim, size in enumerate(self.effective_shape):
                    sample_like = sample_like.narrow(dim + 1, 0, size)

            if self.rademacher:
                sample = sample_rademacher_like(sample_like)
            else:
                sample = sample_gaussian_like(sample_like)
            if isinstance(self.effective_shape, int):
                self._e[:, : self.effective_shape] = sample
            else:
                pad_size = []
                for idx in self.effective_shape:
                    pad_size.append(0)
                    pad_size.append(y.shape[-idx - 1] - self.effective_shape[-idx - 1])
                pad_size = tuple(pad_size)
                self._e = torch.functional.padding(sample, pad_size, mode="constant")
            ## pad zeros

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf_aug(dy, y, self.effective_shape).view(
                    batchsize, 1
                )
            else:
                divergence = self.divergence_fn(
                    dy, y, self.effective_shape, e=self._e
                ).view(batchsize, 1)
        if self.residual:
            dy = dy - y
            if isinstance(self.effective_dim, int):
                divergence -= (
                        torch.ones_like(divergence)
                        * torch.tensor(
                    np.prod(y.shape[1:]) * self.effective_shape / y.shape[1],
                    dtype=torch.float32,
                ).to(divergence)
                )
            else:
                divergence -= (
                        torch.ones_like(divergence)
                        * torch.tensor(
                    np.prod(self.effective_shape),
                    dtype=torch.float32,
                ).to(divergence)
                )
        return tuple(
            [dy, -divergence]
            + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]]
        )


class AugCNFODEfunc(CNFODEfunc):
    """
    Wrapper to make neural nets for use in augmented continuous normalizing flows
    """

    def __init__(self, diffeq, divergence_fn="approximate", residual=False, rademacher=False, effective_dim=None):
        super(AugCNFODEfunc, self).__init__(diffeq=diffeq, divergence_fn=divergence_fn, residual=residual,
                                            rademacher=rademacher)
        assert effective_dim is not None
        self.effective_shape = effective_dim

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf_aug
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx_aug

    def sample_for_hutchinson_estimator(self, y):
        """
        Args:
            y:

        Returns:

        """
        if self._e is None:
            self._e = torch.zeros_like(y)
            if self.rademacher:
                self._e[:, : self.effective_dim] = sample_rademacher_like(y[:, : self.effective_dim])
            else:
                self._e[:, : self.effective_dim] = sample_gaussian_like(y[:, : self.effective_dim])

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]
        # increment num evals
        self._num_evals += 1

        # convert to tensor
        if isinstance(t, torch.Tensor):
            t = t.type_as(y)
        else:
            t = torch.tensor(t).type_as(y)

        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            self._e = torch.zeros_like(y)
            if isinstance(self.effective_shape, int):
                sample_like = y[:, : self.effective_shape]
            else:
                sample_like = y
                for dim, size in enumerate(self.effective_shape):
                    sample_like = sample_like.narrow(dim + 1, 0, size)

            if self.rademacher:
                sample = sample_rademacher_like(sample_like)
            else:
                sample = sample_gaussian_like(sample_like)
            if isinstance(self.effective_shape, int):
                self._e[:, : self.effective_shape] = sample
            else:
                pad_size = []
                for idx in self.effective_shape:
                    pad_size.append(0)
                    pad_size.append(y.shape[-idx - 1] - self.effective_shape[-idx - 1])
                pad_size = tuple(pad_size)
                self._e = torch.functional.padding(sample, pad_size, mode="constant")
            ## pad zeros

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf_aug(dy, y, self.effective_shape).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, self.effective_shape, e=self._e).view(batchsize, 1)
        if self.residual:
            dy = dy - y
            if isinstance(self.effective_dim, int):
                divergence -= (
                        torch.ones_like(divergence)
                        * torch.tensor(
                    np.prod(y.shape[1:]) * self.effective_shape / y.shape[1],
                    dtype=torch.float32,
                ).to(divergence)
                )
            else:
                divergence -= (
                        torch.ones_like(divergence)
                        * torch.tensor(
                    np.prod(self.effective_shape),
                    dtype=torch.float32,
                ).to(divergence)
                )
        return tuple(
            [dy, -divergence]
            + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]]
        )




class LatentSDEFunc(SDEIto):
    def __init__(self, sde_net, noise_type="diagonal"):
        """
        基于SDEIto基类的func，即默认sde_type="ito"
        Args:
            theta:
            noise_type: Optional [diagonal, additive, scalar, general], diagonal as default.
        """
        super(LatentSDEFunc, self).__init__(noise_type=noise_type)
        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        self.f_net = sde_net.f_net
        # Shared diffusion
        self.g_net = sde_net.g_net
        # Prior drift.
        self.h_net = sde_net.h_net

        self._ctx = None

    def get_prior_params(self):
        return self.theta, self.mu, self.sigma

    def set_context(self, ctx):
        """
        encoder的结果，传入用于后验计算
        Args:
            ctx:

        Returns:

        """
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def h(self, t, y):  # Prior drift.
        """
        Latent SDE特殊的一个函数，用于存放prior drift
        Args:
            t:
            y:

        Returns:

        """
        return self.h_net(y)

    def f(self, t, y):  # Approximate posterior drift.
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def g(self, t, y):  # Shared diffusion. / Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)
