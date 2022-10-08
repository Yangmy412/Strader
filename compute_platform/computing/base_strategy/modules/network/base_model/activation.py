import torch.nn as nn

ACTIVATION = {
    "elu": nn.ELU(inplace=True),
    "relu": nn.ReLU(inplace=True),
    "leaky_relu": nn.LeakyReLU(negative_slope=0.2, inplace=True),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus()
}
