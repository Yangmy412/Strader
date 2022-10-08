import time
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from computing.base_strategy.modules.network.flows.cnf.jacobian_func import log_jaco, inversoft_jaco
from computing.core.module import Module
from computing.core.dtype import LearningVariable, HyperParameter
from computing.base_strategy.modules.network.base_model.attention import GAT
from computing.base_strategy.modules.network.base_model.linear import MLP
from computing.base_strategy.modules.network.flows.cnf.cnf_regularization import create_regularization_fns
from computing.base_strategy.modules.network.flows.utils.build_model import build_ctfp


class MicroTimeSeriesModel(nn.Module):
    """
    The micro_ts model aims to generate one stock's micro time series, i.e., the minute-level prices of the stock.
    """
    def __init__(self, iwae_num, sample_iwae_num, num_blocks, batch_norm: bool,
                 input_dim: int, flow_hidden_dims: Union[str, list, tuple], layer_type: str, nonlinearity: str,
                 activation: str, strides: Union[str, list, tuple, bool] = None, time_length=1.0, regularization_fns={}):
        """
        Args:
            iwae_num: number of samples
            num_blocks:  block number，default 1
            batch_norm: whether bn
            input_dim: default 4, means ohlc
            flow_hidden_dims: odenet
            layer_type: odenet
            nonlinearity: odenet
            strides: odenet
            time_length: maximum length of time
            regularization_fns: regular term
        """
        super(MicroTimeSeriesModel, self).__init__()
        self.iwae_num = iwae_num
        self.sample_iwae_num = sample_iwae_num
        self.input_dim = input_dim
        self.activation = activation

        self.mu = nn.Parameter(torch.ones(1, self.input_dim) * 0)
        self.pre_softplus_sigma = torch.ones(1, self.input_dim) * 0.02

        regularization_fns, self.regularization_coeffs = create_regularization_fns(regularization_fns)
        self.spf = build_ctfp(num_blocks, batch_norm, input_dim + 1, input_dim, flow_hidden_dims,
                              layer_type, nonlinearity, strides, time_length, regularization_fns)

    def forward(self, x, t, mask, final_index, target_time):
        """
        Args:
            x: observed data, a 3-D tensor of shape batch_size x max_length x feature_num
            t: observed times, a 3-D tensor of shape batch_size x max_length x 1
            mask: a 2-D binary tensor of shape batch_size x max_length
                  showing whether the position is observation or padded dummy variables
            final_index: a 1-D tensor of shape batch_size
            target_time: a list of target time

        Returns:

        """
        self.pre_softplus_sigma = self.pre_softplus_sigma.to(x)
        x = x.repeat_interleave(self.iwae_num, 0)  # [b*i,m,f]
        t = t.repeat_interleave(self.iwae_num, 0)  # [b*i,m,1]
        mask = mask.repeat_interleave(self.iwae_num, 0)  # [b*i,m]

        # micro-encoder
        z, likelihood = self.observe_to_base(x, t, mask)  # [b*i,m,f], [1]

        # extrapolate
        z_extra = self.sde_extrapolation(z, t, mask, final_index, target_time)  # [b*i,t,f]

        # micro-decoder
        t_extra = target_time.view(1, -1, 1).repeat_interleave(z_extra.shape[0], 0)  # [b*i,t,1]
        x_extra = self.base_to_observe(z, t, z_extra, t_extra)  # [b,m,f], [b,t,f]
        return x_extra, likelihood

    ###########################################################################################
    # ===================================== base_to_observe ===================================
    def base_to_observe(self, z, t, z_extra, t_extra):
        batch_size, max_length, effective_num = z.shape
        z_extra_length = z_extra.shape[1]

        if self.training:
            z_zeros = torch.zeros(batch_size, max_length - z_extra_length, effective_num).to(z)
            z_cat = torch.cat([z_zeros, z_extra], dim=1).view(-1, effective_num)  # [b*i*(m+t),f]
            t_zeros = torch.zeros(batch_size, max_length - z_extra_length, 1).to(z)
            t_cat = torch.cat([t_zeros, t_extra.to(t)], dim=1).view(-1, 1)  # [b*i*(m+t),1]
            input = torch.cat([z_cat, t_cat], dim=1)  # [b*i*(m+t),f+1]
        else:
            z_cat = torch.cat([z, z_extra], dim=1).view(-1, effective_num)  # [b*i*(m+t),f]
            t_cat = torch.cat([t, t_extra.to(t)], dim=1).view(-1, 1)  # [b*i*(m+t),1]
            input = torch.cat([z_cat, t_cat], dim=1)  # [b*i*(m+t),f+1]

        x, _ = self.spf(input, torch.zeros(input.shape[0], 1).to(input), reverse=True)
        x = x[:, :-1]  # [b*i*(m+t),f]
        x = x.view(batch_size // self.iwae_num, self.iwae_num, -1, effective_num)  # [b,i,m+t,f]
        x = torch.mean(x, dim=1)  # [b,m+t,f]
        x_extra = x[:, - z_extra_length:]  # [b,t,f]

        return x_extra

    ###########################################################################################
    # ===================================== sde_extrapolation =================================
    def sde_extrapolation(self, z, t, mask, final_index, target_times):
        batch_size, _, effective_num = z.shape
        # final_index = mask.shape[1] - mask.flip(-1).max(1).indices - 1  # 最后一条存在的数据的index，[b*i]
        final_index = final_index.to(torch.int64)
        z_pre = torch.gather(z, 1, final_index.view(-1, 1, 1).expand(batch_size, 1, effective_num)).squeeze(
            1)  # [b*i,f]
        t_pre = torch.gather(t, 1, final_index.view(-1, 1, 1).expand(batch_size, 1, 1)).squeeze(1)  # [b*i,1]

        target_times = target_times.unsqueeze(0).repeat_interleave(batch_size, 0).to(z)
        z_extra = self.sample_from_base_process(target_times, t_pre, z_pre)
        return z_extra

    def sample_from_base_process(self, t_target, t_pre, z_pre):
        """
        do extrapolation

        Args:
            t_target: target time point tensor of batch_size x target_len
            t_pre: a single time point of the previous observation of batch_size x 1
            z_pre: observations at t_prev of shape batch_size x feature_num
        Returns: samples
        """
        delta_time = t_target - t_pre  # [b,t]
        delta_time = delta_time.repeat_interleave(self.sample_iwae_num, 0).unsqueeze(2)  # [b*i,t,1]
        z_pre = z_pre.repeat_interleave(self.sample_iwae_num, 0).unsqueeze(1)  # [b*i,1,f]

        pre_softplus_sigma = self.last_pre_softplus_sigma if self.pre_softplus_sigma.isnan().sum() != 0 else self.pre_softplus_sigma
        sigma = torch.max(F.relu(pre_softplus_sigma), torch.tensor(1e-5))
        mean = torch.log(z_pre) + self.mu * delta_time  # [b*i,t,f]
        std = torch.ones_like(z_pre) * (sigma * self.helper_std_function(delta_time))  # [b*i,t,f]

        log_base_distributions = torch.distributions.Normal(mean, std)
        log_samples = log_base_distributions.sample()  # [b*i,t,f]

        samples = torch.exp(log_samples)  # [b*i,t,f]
        samples = samples.reshape(-1, self.sample_iwae_num, delta_time.shape[1], z_pre.shape[2]).mean(1)
        return samples  # [b,t,f]

    ###########################################################################################
    # ===================================== observe_to_base ===================================
    def observe_to_base(self, x, t, mask):
        batch_size, max_length, feature_num = x.shape

        aux = torch.cat([torch.zeros_like(x), t], dim=2)  # [b*i,m,f+1]

        aux = aux.view(-1, aux.shape[2])  # [b*i*m,f+1]
        aux, _ = self.spf(aux, torch.zeros(aux.shape[0], 1).to(aux), reverse=True)
        aux = aux[:, self.input_dim:]  # [b*i*m,1] (1 is time feature)

        if self.activation == "exp":
            transform_values, transform_logdet = log_jaco(x)  # v：[b,m,i][100,89,1] det: [b,m][100,89]
        elif self.activation == "softplus":
            transform_values, transform_logdet = inversoft_jaco(x)
        elif self.activation == "identity":
            transform_values = x
            transform_logdet = torch.sum(torch.zeros_like(x), dim=2)
        else:
            raise NotImplementedError
        input = torch.cat([transform_values.view(-1, feature_num), aux], dim=-1)  # [b*i*m,f+1]
        # print("input", input.isnan().sum(), input.isinf().sum(), (input < 0).sum())
        z, logdet = self.spf(input, torch.zeros(input.shape[0], 1).to(input))  # [b*i*m,f+1], [b*i*m,1]
        z = z[:, :-1].reshape(batch_size, max_length, feature_num)  # [b*i,m,f]
        logdet = logdet.reshape(batch_size, max_length)  # [b*i,m]

        likelihood = self.calculate_likelihood(z, logdet, t, mask, transform_logdet)  # [b]
        return z, likelihood

    def calculate_likelihood(self, z, logdet, t, mask, transform_logdet):  # [b,m,f], [b], [b,m,1], [b,m], [b,m]
        z_prev = z[:, :-1]
        z_current = z[:, 1:]
        t_prev = t[:, :-1]
        t_current = t[:, 1:]

        ll = self.density_calculation(z_current, t_current, z_prev, t_prev)
        ll = (ll - logdet[:, 1:] - transform_logdet[:, 1:]) * mask[:, 1:]

        ll = ll.mean(1).reshape(-1, self.iwae_num)
        ll = (torch.softmax(ll.detach(), 1) * ll).sum(1)
        return ll

    def density_calculation(self, z_current, t_current, z_prev=None, t_prev=None):
        """
        Args:
            z_current: current step base variable, a 2-D tensor of shape batch_size x effective_num
            t_current: current step stamps, a 2-D tensor of shape batch_size x 1
            z_prev: previous step base variable
            t_prev: previous step stamps

        Returns:
            the likelihood
        """
        pre_softplus_sigma = self.last_pre_softplus_sigma if self.pre_softplus_sigma.isnan().sum() != 0 \
            else self.pre_softplus_sigma
        sigma = torch.max(F.relu(pre_softplus_sigma), torch.tensor(1e-5).to(pre_softplus_sigma))
        self.last_pre_softplus_sigma = pre_softplus_sigma.clone().detach()
        mask = (torch.ones_like(z_current) * t_current != 0)
        if z_prev is None:  # the first prediction
            return 0
        else:
            delta_time = t_current - t_prev
            mean = torch.log(z_prev) + (self.mu - sigma ** 2 / 2) * delta_time
            std = torch.ones_like(z_prev) * (sigma * self.helper_std_function(delta_time))

        normal_dist = torch.distributions.Normal(mean, std)
        likelihood = normal_dist.log_prob(torch.log(z_current))
        likelihood = likelihood * (mask * 1.)
        likelihood = likelihood.sum(-1)
        return likelihood

    @staticmethod
    def helper_std_function(param):
        # Helper function for computing torch.sqrt(param)
        result = torch.sqrt(param)
        result = torch.maximum(result, torch.ones_like(result) * 0.00002)
        return result


class STraderStateModel(Module):
    """
    The stocks' stochastic time series model to generate STrader's state.
    """
    def __init__(self, stock_num, input_dim=4, iwae_num=1, sample_iwae_num=128, num_blocks=1, batch_norm=False,
                 flow_hidden_dims=[16, 32, 32, 16], layer_type="ignore", nonlinearity="softplus", strides=None,
                 activation="softplus", time_length=1.0, regularization_fns={},
                 save_path=None, model_path=None):
        """
        Args:
            stock_num: the number of stocks
            input_dim: default 4, means ohlc
            iwae_num: number of samples
            num_blocks:  the number of block in cnf，default 1
            batch_norm: whether bn
            flow_hidden_dims, layer_type, nonlinearity, strides: the params of odenet in cnf
            activation: the last transformation after cnf
            time_length: maximum length of time
            regularization_fns: regular term
        """
        super(STraderStateModel, self).__init__(module_id=-1)
        self.save_path = HyperParameter(save_path)
        if model_path:
            self.micro_ts = torch.load(model_path)
        else:
            self.micro_ts = nn.ModuleList(
                [MicroTimeSeriesModel(iwae_num, sample_iwae_num, num_blocks, batch_norm, input_dim, flow_hidden_dims,
                     layer_type, nonlinearity, activation, strides, time_length, regularization_fns)
                 for _ in range(stock_num)])

        self.target_micro_price = LearningVariable(None)
        self.likelihood = LearningVariable(None)

        self.register_decide_hooks(['target_micro_price', 'likelihood'])

    def forward(self, state, target_time):
        """
        Args:
            state: a dict contains follow elements:
                   x: observed data, a 4-D tensor of shape batch_size x stock_num x max_length x feature_num
                   t: observed times, a 4-D tensor of shape batch_size x stock_num x max_length x 1
                   mask: a 3-D binary tensor of shape batch_size x stock_num x max_length
                         showing whether the position is observation or padded dummy variables
                   pre_w: the normalize weight at last weight, a 2-D tensor of shape batch_size x stock_num
            target_time: a list of target time
        Returns:

        """
        x, t, mask, final_index = state["x"], state["t"], state["mask"], state["final_index"]
        batch_size, stock_num, max_length, feature_num = x.shape

        x_extra = torch.zeros(batch_size, stock_num, len(target_time), feature_num).to(x)
        self.likelihood = 0
        for s in range(stock_num):
            x_extra[:, s], likelihood = self.micro_ts[s](x[:, s], t[:, s], mask[:, s], final_index[:, s], target_time)
            self.likelihood += likelihood

        self.target_micro_price = x_extra

    def update(self):
        if self.save_path:
            timestamp_hash = time.time()
            file_name = self.save_path + "_" + str(timestamp_hash) + ".pth"
            torch.save(self.micro_ts, file_name)
            print("The model have saved in {}".format(file_name))
        else:
            print("The model don't save, because there is no save path")



class DecisionMaking(Module):
    """
    The STrader's policy network to compute the portfolio's result.
    """
    def __init__(self, k, feature_num, trading_points_num, hidden_size, mlp_layer_num, head_num, dropout=0.1,
                 alpha=0.2):
        super(DecisionMaking, self).__init__(module_id=-1)
        self.k = HyperParameter(k)
        self.gat = GAT(trading_points_num * feature_num, hidden_size, hidden_size, head_num, dropout, alpha)
        self.w_mlp = MLP(hidden_size + 1, [hidden_size] * mlp_layer_num + [1])
        self.score_mlp = MLP(feature_num, [hidden_size] * mlp_layer_num + [1])

        self.w = LearningVariable(None)
        self.trading_points = LearningVariable(None)
        self.score = LearningVariable(None)
        self.register_decide_hooks(['w', 'trading_points', 'score'])

    def torch_cov(self, x):
        d = x.shape[-1]
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return 1 / (d - 1) * x @ x.transpose(-1, -2)

    def forward(self, micro_price, pre_w):
        """
        @param micro_price: batch_size x stock_num x trading_points_num x feature_num
        @param pre_w: batch_size x (1+stock_num)
        """
        batch_size, stock_num, trading_points_num, feature_num = micro_price.shape
        # 1. calculate portfolio
        micro_price_cash = torch.ones([batch_size, 1, trading_points_num, feature_num], device=micro_price.device)
        micro_price_feature = torch.cat([micro_price_cash, micro_price], dim=1)  # [b,1+s,t,4]
        micro_price_feature = micro_price_feature.view(batch_size, stock_num + 1, -1)  # [b,1+s,t*4]
        hidden = self.gat(micro_price_feature, self.torch_cov(micro_price_feature))  # [b,1+s,h]

        feature = torch.cat([hidden, pre_w.unsqueeze(-1)], dim=-1)  # [b,1+s,h+1]
        out = self.w_mlp(feature).squeeze(-1)  # [b,1+s]
        w = torch.softmax(out, dim=-1)

        # 2. generate trading points candidate
        score = torch.sigmoid(self.score_mlp(micro_price)).squeeze(-1)
        sell_points = torch.topk(score, self.k, dim=-1, largest=True, sorted=False).indices  # [b,s,k]
        buy_points = torch.topk(score, self.k, dim=-1, largest=False, sorted=False).indices  # [b,s,k]

        buy_or_sell = (w[:, 1:] > pre_w[:, 1:])
        buy_or_sell = buy_or_sell.unsqueeze(-1).expand(sell_points.shape)
        trading_points = torch.where(buy_or_sell, buy_points, sell_points)  # [b,s,k]

        self.w, self.trading_points, self.score = w, trading_points, score