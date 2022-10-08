from typing import Union

from computing.base_strategy.modules.network.base_model.batch_norm import MovingBatchNorm1d
from computing.base_strategy.modules.network.ode.lib.ode_net import ODEnet
from computing.base_strategy.modules.network.ode.lib.ode_func import CNFODEfunc
from computing.base_strategy.modules.network.flows.cnf.cnf import CNF, SequentialCNF
from computing.base_strategy.modules.network.ode.lib.ode_net import AugODEnet
from computing.base_strategy.modules.network.ode.lib.ode_func import AugCNFODEfunc

__all__ = ["build_cnf", "build_aug_cnf", "build_sequential_cnf", "build_ctfp"]

def build_cnf(input_dim: int, hidden_dims: Union[str, list, tuple], layer_type:str, nonlinearity:str,
              strides: Union[str, list, tuple, bool]=None,
              time_length=1.0, regularization_fns=None):
    """
        Args:
            input_dim: ode_net parameter
            hidden_dims: ode_net parameter
            layer_type: ode_net parameter
            nonlinearity: ode_net parameter
            strides: ode_net parameter
            time_length: cnf parameter
            regularization_fns: cnf parameter

        Returns:
            conditional continuous normalizing flows with ode
        """
    hidden_dims = tuple(map(int, hidden_dims.split(","))) if type(hidden_dims)==str else hidden_dims
    strides = tuple(map(int, strides.split(",")))  if type(strides)==str else strides

    diffeq = ODEnet(in_dim=input_dim,  hidden_dims=hidden_dims, out_dim =input_dim, strides=strides,
                    layer_type=layer_type, nonlinearity=nonlinearity)

    odefunc = CNFODEfunc(diffeq=diffeq, divergence_fn="approximate", residual=False, rademacher=True)

    cnf = CNF(odefunc=odefunc, T=time_length, train_T=True, regularization_fns=regularization_fns,
              solver='dopri5', rtol=1e-5, atol=1e-5)
    return cnf


def build_sequential_cnf(num_blocks, batch_norm: bool,
                         input_dim: int, hidden_dims: Union[str, list, tuple], layer_type:str, nonlinearity:str,
                         strides: Union[str, list, tuple, bool]=None,
                         time_length=1.0, regularization_fns=None):
    """
    Args:
        num_blocks: the block num of the sequential model
        batch_norm: if ture, use 1-D batch_norm
        input_dim: ode_net parameter
        hidden_dims: ode_net parameter
        layer_type: ode_net parameter
        nonlinearity: ode_net parameter
        strides: ode_net parameter
        time_length: cnf parameter
        regularization_fns: cnf parameter

    Returns:
        sequential conditional continuous normalizing flows with ode
    """
    chain = [build_cnf(input_dim, hidden_dims, layer_type, nonlinearity, strides,
                       time_length, regularization_fns, ) for _ in range(num_blocks)]
    if batch_norm:
        bn_layers = [
            MovingBatchNorm1d(input_dim, bn_lag=0)
            for _ in range(num_blocks)
        ]
        bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=0)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = SequentialCNF(chain)
    return model


def build_aug_cnf(input_dim: int, effective_dim: int, hidden_dims: Union[str, list, tuple],
                  layer_type:str, nonlinearity:str, strides: Union[str, list, tuple, bool]=None,
                  time_length=1.0, regularization_fns={}):
    """
        Args:
            input_dim: ode_net parameter
            effective_dim: ode_net parameter
            hidden_dims: ode_net parameter
            layer_type: ode_net parameter
            nonlinearity: ode_net parameter
            strides: ode_net parameter

            time_length: cnf parameter
            regularization_fns: cnf parameter

        Returns:
            a CTFP model, i.e., conditional continuous normalizing flows with aug_ode
        """
    hidden_dims = tuple(map(int, hidden_dims.split(","))) if type(hidden_dims)==str else hidden_dims
    strides = tuple(map(int, strides.split(",")))  if type(strides)==str else strides

    diffeq = AugODEnet(in_dim=input_dim,  effective_dim=effective_dim,
                       hidden_dims=hidden_dims, out_dim =input_dim, strides=strides,
                       layer_type=layer_type, nonlinearity=nonlinearity)

    odefunc = AugCNFODEfunc(diffeq=diffeq, divergence_fn="approximate", residual=False, rademacher=True,
                            effective_dim=effective_dim)

    cnf = CNF(odefunc=odefunc, T=time_length, train_T=True, regularization_fns=regularization_fns,
              solver='dopri5', rtol=1e-5, atol=1e-5)
    return cnf


def build_ctfp(num_blocks, batch_norm: bool,
               input_dim: int, effective_dim: int, hidden_dims: Union[str, list, tuple],
               layer_type: str, nonlinearity: str, strides: Union[str, list, tuple, bool]=None,
               time_length=1.0, regularization_fns={}):
    """
    Args:
        num_blocks: the block num of the sequential model
        batch_norm: if ture, use 1-D batch_norm
        input_dim: ode_net parameter
        hidden_dims: ode_net parameter
        layer_type: ode_net parameter
        nonlinearity: ode_net parameter
        strides: ode_net parameter
        time_length: cnf parameter
        regularization_fns: cnf parameter

    Returns:
        sequential conditional continuous normalizing flows with ode
    """
    chain = [build_aug_cnf(input_dim, effective_dim, hidden_dims, layer_type, nonlinearity, strides,
                           time_length, regularization_fns) for _ in range(num_blocks)]
    if batch_norm:
        bn_layers = [
            MovingBatchNorm1d(input_dim, bn_lag=0)
            for _ in range(num_blocks)
        ]
        bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=0)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = SequentialCNF(chain)
    return model



