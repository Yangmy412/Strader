import torch.nn as nn
import torch

__all__ = ["NONLINEARITIES", "unsqueeze", "squeeze", "log_normal_standard", "log_normal_diag", "init_network_weights",
           "split_last_dim", "_flip", "_get_minibatch_jacobian",
           "divergence_bf",
           "divergence_approx",
           "sample_rademacher_like",
           "sample_gaussian_like"]


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x ** 2),
    "identity": Lambda(lambda x: x),
}


def unsqueeze(input, upscale_factor=2):
    """
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    """
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor ** 2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(
        batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width
    )

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input, downscale_factor=2):
    """
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    """
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor ** 2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size,
        in_channels,
        out_height,
        downscale_factor,
        out_width,
        downscale_factor,
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim // 2

    if len(data.size()) == 3:
        res = data[:, :, :last_dim], data[:, :, last_dim:]

    if len(data.size()) == 2:
        res = data[:, :last_dim], data[:, last_dim:]
    return res


def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.

    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(y[:, j], x, torch.ones_like(y[:, j]), retain_graph=True,
                                      create_graph=True)[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


def divergence_bf(dx, y, effective_dim=None, **unused_kwargs):
    """
    Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    Args:
        dx: Output of the neural ODE function
        y: input to the neural ODE function
        effective_dim: 除去aug dim之后的维度
        **unused_kwargs:

    Returns:
        sum_diag: 求一个trace的可导的解(determin)
    """
    effective_dim = y.shape[1] if effective_dim == None else effective_dim
    assert effective_dim <= y.shape[1]

    sum_diag = 0.
    for i in range(effective_dim):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def divergence_approx(f, y, e=None):
    """
        Calculates the approx trace of the Jacobian df/dz.
        link: https://github.com/rtqichen/ffjord/
        Args:
            f: Output of the neural ODE function
            y: input to the neural ode function
            e:
        Returns:
            sum_diag: estimate log determinant of the df/dy 求一个trace的可导的无偏估计解
    """
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def divergence_bf_aug(dx, y, effective_dim, **unused_kwargs):
    """
    The function for computing the exact log determinant of jacobian for augmented ode

    Parameters
        dx: Output of the neural ODE function
        y: input to the neural ode function
        effective_dim (int): the first n dimension of the input being transformed
                             by normalizing flows to compute log determinant
    Returns:
        sum_diag: determin
    """
    sum_diag = 0.0
    assert effective_dim <= y.shape[1]
    for i in range(effective_dim):
        sum_diag += (
            torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0]
                .contiguous()[:, i]
                .contiguous()
        )
    return sum_diag.contiguous()


def divergence_approx_aug(f, y, effective_dim, e=None):
    """
    The function for estimating log determinant of jacobian
    for augmented ode using Hutchinson's Estimator

    Parameters
        f: Output of the neural ODE function
        y: input to the neural ode function
        effective_dim (int): the first n dimensions of the input being transformed
                             by normalizing flows to compute log determinant

    Returns:
        sum_diag: estimate log determinant of the df/dy
    """
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)
