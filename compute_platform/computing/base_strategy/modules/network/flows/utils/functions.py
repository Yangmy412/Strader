import math

import hamiltorch
import torch


def reparameterize(mu, logvar, eps):
    std = torch.exp(0.5 * logvar)
    z = mu + eps * std
    return z


def sample_from_base_dist(mu, logvar, sample_num=1):
    if sample_num > 1:
        mu = mu.unsqueeze(0)
        logvar = logvar.unsqueeze(0)
        eps = torch.rand(sample_num, *mu.shape).to(mu.device)
    else:
        eps = torch.randn_like(mu)
    return reparameterize(mu, logvar, eps)


def log_prob_func(params):
    dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
    return dist.log_prob(params).sum()


def sample_from_base_dist_by_hmc(mu, logvar, sample_num, step_size, num_steps_per_sample, sample_dim=1):
    """
    Args:
        mu: torch.tensor, [...,d]
        logvar: torch.tensor, [...,d]
        sample_num: int
    """
    assert sample_num > 1
    shape_list = list(mu.shape)

    eps = []
    for _ in range(math.prod(shape_list[:-1])):
        params_init = torch.zeros(shape_list[-1:]).to(mu.device)
        eps_ = hamiltorch.sample(log_prob_func=log_prob_func, params_init=params_init,
                                 num_samples=sample_num, step_size=step_size,
                                 num_steps_per_sample=num_steps_per_sample, debug=0)
        eps_ = torch.stack(eps_, dim=1)  # [feature_dim, sample_num]
        eps.append(eps_)

    eps = torch.stack(eps)  # [batch_size, feature_dim, sample_num]
    eps = eps.reshape(*shape_list, sample_num)
    eps = eps.permute(list(range(sample_dim)) + [len(shape_list)] + list(range(sample_dim, len(shape_list))))

    mu, logvar = mu.unsqueeze(sample_dim), logvar.unsqueeze(sample_dim)  # [b,1,d]
    return reparameterize(mu, logvar, eps)


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)
