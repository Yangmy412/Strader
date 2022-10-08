import torch.nn as nn

ACTIVATION = {
    "elu": nn.ELU(inplace=True),
    "relu": nn.ReLU(inplace=True),
    "leaky_relu": nn.LeakyReLU(negative_slope=0.2, inplace=True),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "soft_plus": nn.Softplus()
}


class MLP(nn.Module):
    def __init__(self, input_size, output_size_list, activation=None, last_activation=None):
        """
        @param input_size: number of dimensions for each point in each sequence.
        @param output_size_list: list of ints，number of output dimensions for each layer.
        """
        super().__init__()
        self.input_size = input_size  # e.g. 2
        self.output_size_list = output_size_list  # e.g. [128, 128, 128, 128]
        network_size_list = [input_size] + self.output_size_list  # e.g. [2, 128, 128, 128, 128]
        network_list = []

        # iteratively build neural network.
        for i in range(1, len(network_size_list) - 1):
            network_list.append(nn.Linear(network_size_list[i - 1], network_size_list[i], bias=True))
            if activation is not None:
                if activation in ACTIVATION.keys():
                    network_list.append(ACTIVATION[activation])
                else:
                    raise IOError("The activation is not supported now!")

        # add final layer, create sequential container.
        network_list.append(nn.Linear(network_size_list[-2], network_size_list[-1]))
        if last_activation is not None:
            if last_activation in ACTIVATION.keys():
                network_list.append(ACTIVATION[last_activation])
            else:
                raise IOError("The activation is not supported now!")

        # construct the mlp
        self.mlp = nn.Sequential(*network_list)

    def forward(self, x):
        return self.mlp(x)
