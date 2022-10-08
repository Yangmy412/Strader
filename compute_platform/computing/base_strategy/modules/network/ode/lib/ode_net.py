import copy
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from computing.base_strategy.modules.network.ode.lib import basic_net
from computing.base_strategy.modules.network.ode.lib.func import NONLINEARITIES, squeeze, unsqueeze

__all__ = ["ODEnet", "AugODEnet", "SDEnet"]


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in ode, 用于封装ode_func，有特定的IO
    """

    def __init__(self, in_dim: int, hidden_dims: Union[int, list, tuple], out_dim: int,
                 strides=None, layer_type="concat", nonlinearity="softplus"):
        """
        多层的 diffeq net，base_layer网络分为「线性网络」和「卷积网络」2类
        Args:
            in_dim: 如果是线性网络，表示输入数据的feature维度；
                    如果是卷积网络，表示输入数据的in_channels大小。
            hidden_dims: 如果是int类型，表示是单层网络;
                         如果list/tuple类型，表示是多层网络，其中每一个元素表示每一层的输出维度。
            out_dim: 如果是线性网络，表示输出数据的feature维度；
                     如果是卷积网络，表示输出数据的in_channels大小。
            strides: 如果是线性网络，填写此参数无效；
                     如果是卷积网络，if strides is None，所有卷积层使用basic_net中的默认参数；
                                   if strides is not None，the type of strides must be list，such as [1,2,None],
                                                           其中的每一个元素表示每一层卷积的参数，该元素仅支持这值[1,2,-2,None]。
                                                           当stride==None时，表示使用basic_net中的默认参数；
                                                           当stride==1/2/-2时，具体的卷积的参数见_get_conv_layer_kwarg函数
            layer_type: 单层网络的类型，目前支持的单层网络有 ["ignore", "hyper", "squash", "concat", "concat_v2",
                                                          "concatsquash", "blend", "concatcoord", "conv_ignore",
                                                          "conv_hyper", "conv_squash", "conv_concat", "conv_concat_v2",
                                                          "conv_concatsquash", "conv_blend", "conv_concatcoord"]，
                         其中，如果前缀为"conv"表示卷积网络，反之为线性网络
            nonlinearity: 激活函数的类型。除了最后一层，每一层网络后面会跟着激活函数。
                          目前支持的激活函数有 ["tanh", "softplus", "elu", "swish", "square", "identity"]，
                          如果想要了解具体的激活函数，可以查看lib.func.NONLINEARITIES。
        """
        super(ODEnet, self).__init__()
        self.base_layer = {
            "ignore": basic_net.IgnoreLinear,
            "hyper": basic_net.HyperLinear,
            "squash": basic_net.SquashLinear,
            "concat": basic_net.ConcatLinear,
            "concat_v2": basic_net.ConcatLinear_v2,
            "concatsquash": basic_net.ConcatSquashLinear,
            "blend": basic_net.BlendLinear,
            "concatcoord": basic_net.ConcatLinear,
            "conv_ignore": basic_net.IgnoreConv2d,
            "conv_hyper": basic_net.HyperConv2d,
            "conv_squash": basic_net.SquashConv2d,
            "conv_concat": basic_net.ConcatConv2d,
            "conv_concat_v2": basic_net.ConcatConv2d_v2,
            "conv_concatsquash": basic_net.ConcatSquashConv2d,
            "conv_blend": basic_net.BlendConv2d,
            "conv_concatcoord": basic_net.ConcatCoordConv2d,
        }[layer_type]
        self.layer_type = [None] * (len(hidden_dims) + 1) if strides is None else strides
        self.strides = strides
        self.nonlinearity = nonlinearity

        # dim process
        hidden_dims = [hidden_dims] if type(hidden_dims) is int else hidden_dims
        dims = [in_dim] + hidden_dims + [out_dim]

        # build layers and add them
        self.layers, self.activation_fns = self._bulid_layers(dims)

    def _get_conv_layer_kwargs(self, stride):
        """
        这一丢参数适合cv数据，不适合时序
        """
        if stride is None:
            layer_kwargs = {}
        elif stride == 1:
            layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
        elif stride == 2:
            layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
        elif stride == -2:
            layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
        else:
            raise ValueError('Unsupported stride: {}'.format(stride))
        return layer_kwargs

    def _bulid_layers(self, dims):
        """
        Args:
            dims: 如果是int类型，表示是单层网络;
                  如果list/tuple类型，表示是多层网络，其中每一个元素表示每一层的输入维度。
        Returns:
            layers: 2个nn.ModuleList，分别是layers和activation_fns,
                    它们交替构成：layer1->activation1->layer2->....->activation_dim-2->layer_dim-1
        """
        layers = []
        activation_fns = []

        for i in range(len(dims) - 1):
            if "conv" in self.layer_type:    # conv网络
                layer_kwargs = self._get_conv_layer_kwargs(self.strides[i])
                layers.append(self.base_layer(dims[i], dims[i + 1], **layer_kwargs))
            else:   # linear网络
                layers.append(self.base_layer(dims[i], dims[i + 1]))

            # if i < len(dims) - 2:
            activation_fns.append(NONLINEARITIES[self.nonlinearity])

        return nn.ModuleList(layers), nn.ModuleList(activation_fns)

    def forward(self, t, x):
        for l, layer in enumerate(self.layers):
            x = layer(t, x)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                x = self.activation_fns[l](x)
        return x


class AugODEnet(ODEnet):
    """
    Class to make neural nets for use in augmented continuous normalizing flows
    Only consider one-dimensional data yet
    """

    def __init__(self, in_dim: int, effective_dim: int, hidden_dims: Union[int, list, tuple], out_dim: int,
                 aug_hidden_dims=None, strides=None, layer_type="concat", nonlinearity="softplus"):
        """
        生成两个多层的net，base_layer网络分为「线性网络」和「卷积网络」2类
        Args:
            in_dim: 如果是线性网络，表示输入数据的feature维度 + augment维度；
                    如果是卷积网络，表示输入数据的in_channels大小 + augment的in_channels大小。
            effective_dim： 如果是线性网络，表示输入数据的feature维度；
                            如果是卷积网络，表示输入数据的in_channels大小
            hidden_dims: 如果是int类型，表示是单层网络;
                         如果list/tuple类型，表示是多层网络，其中每一个元素表示每一层的输出维度。
            out_dim: 如果是线性网络，表示输出数据的feature维度；
                     如果是卷积网络，表示输出数据的in_channels大小。
            aug_hidden_dims: augment部分数据参与的网络的hidden_dims；
                             if is None，则拷贝hidden_dims
            strides: 如果是线性网络，填写此参数无效；
                     如果是卷积网络，if strides is None，所有卷积层使用basic_net中的默认参数；
                                    if strides is not None，the type of strides must be list，such as [1,2,None],
                                                            其中的每一个元素表示每一层卷积的参数，该元素仅支持这值[1,2,-2,None]。
                                                            当stride==None时，表示使用basic_net中的默认参数；
                                                            当stride==1/2/-2时，具体的卷积的参数见_get_conv_layer_kwarg函数
            layer_type: 单层网络的类型，目前支持的单层网络有 ["ignore", "hyper", "squash", "concat", "concat_v2",
                                                          "concatsquash", "blend", "concatcoord", "conv_ignore",
                                                          "conv_hyper", "conv_squash", "conv_concat", "conv_concat_v2",
                                                          "conv_concatsquash", "conv_blend", "conv_concatcoord"]，
                         其中，如果前缀为"conv"表示卷积网络，反之为线性网络
            nonlinearity: 激活函数的类型。除了最后一层，每一层网络后面会跟着激活函数。
                          目前支持的激活函数有 ["tanh", "softplus", "elu", "swish", "square", "identity"]，
                          如果想要了解具体的激活函数，可以查看lib.func.NONLINEARITIES。
        """
        super(AugODEnet, self).__init__(in_dim=in_dim, hidden_dims=hidden_dims, out_dim=effective_dim,
                                        strides=strides, layer_type=layer_type, nonlinearity=nonlinearity)
        self.effective_dim = effective_dim

        # build self aug_layers
        if aug_hidden_dims is None:
            aug_hidden_dims = copy.copy(hidden_dims)

        aug_dims = [in_dim - effective_dim] + aug_hidden_dims + [in_dim - effective_dim]
        self.aug_layers, self.aug_activation_fns = self._bulid_layers(aug_dims)

    def forward(self, t, y):
        dx = y
        aug = y[:, self.effective_dim:]

        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if l != len(self.layers) - 1:
            dx = self.activation_fns[l](dx)

        for l, layer in enumerate(self.aug_layers):
            aug = layer(t, aug)
            # if l < len(self.aug_layers) - 1:
            aug = self.aug_activation_fns[l](aug)

        dx = torch.cat([dx, aug], dim=1)
        return dx


class SDEnet(nn.Module):
    """
        Helper class to make neural nets for use in sde, 用于封装ode_func，有特定的IO
    """

    def __init__(self, latent_size: int, context_size: int, hidden_dims: Union[int, list, tuple],
                 layer_type="ignore", nonlinearity="softplus", ):
        """
        复合了sde需要的三个net，base_layer网络都是「线性网络」
        @param latent_size: 输入和输出的size
        @param context_size: 参与后验的输入size
        @param hidden_dims: 如果是int类型，表示是单层网络;
                            如果list/tuple类型，表示是多层网络，其中每一个元素表示每一层的输出维度。
        @param layer_type: 单层网络的类型，目前支持的单层网络有 ["ignore", "hyper", "squash", "concat", "concat_v2",
                                                         "concatsquash", "blend", "concatcoord", "conv_ignore",
                                                         "conv_hyper", "conv_squash", "conv_concat", "conv_concat_v2",
                                                         "conv_concatsquash", "conv_blend", "conv_concatcoord"]，
        @param nonlinearity: 激活函数的类型。除了最后一层，每一层网络后面会跟着激活函数。
                             目前支持的激活函数有 ["tanh", "softplus", "elu", "swish", "square", "identity"]，
                             如果想要了解具体的激活函数，可以查看lib.func.NONLINEARITIES。
        """
        super(SDEnet, self).__init__()

        base_layer = {
            "ignore": basic_net.IgnoreLinear,
            "hyper": basic_net.HyperLinear,
            "squash": basic_net.SquashLinear,
            "concat": basic_net.ConcatLinear,
            "concat_v2": basic_net.ConcatLinear_v2,
            "concatsquash": basic_net.ConcatSquashLinear,
            "blend": basic_net.BlendLinear,
            "concatcoord": basic_net.ConcatLinear,
            "conv_ignore": basic_net.IgnoreConv2d,
            "conv_hyper": basic_net.HyperConv2d,
            "conv_squash": basic_net.SquashConv2d,
            "conv_concat": basic_net.ConcatConv2d,
            "conv_concat_v2": basic_net.ConcatConv2d_v2,
            "conv_concatsquash": basic_net.ConcatSquashConv2d,
            "conv_blend": basic_net.BlendConv2d,
            "conv_concatcoord": basic_net.ConcatCoordConv2d,
        }[layer_type]

        # dim process
        if type(hidden_dims) is int:
            hidden_dims = [hidden_dims]
        # prior drift net
        f_net_dims = [latent_size + context_size] + hidden_dims + [latent_size]
        # posterior drift net
        h_net_dims = [latent_size] + hidden_dims + [latent_size]

        # build layers and add them
        self.f_net = nn.Sequential()
        self.h_net = nn.Sequential()

        for i in range(len(f_net_dims) - 1):
            self.f_net.add_module("net" + str(i), base_layer(f_net_dims[i], f_net_dims[i + 1]))
            self.f_net.add_module("activation" + str(i), NONLINEARITIES[nonlinearity])
            self.h_net.add_module("net" + str(i), base_layer(h_net_dims[i], h_net_dims[i + 1]))
            self.h_net.add_module("activation" + str(i), NONLINEARITIES[nonlinearity])

        # diffusion net
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_dims[0]),
                    nn.Softplus(),
                    nn.Linear(hidden_dims[0], 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )



