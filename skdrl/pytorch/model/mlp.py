import torch
import torch.nn as nn


class NaiveMultiLayerPerceptron(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: list = [64, 32],
                 hidden_act_func: str = 'ReLU',
                 out_act_func: str = 'Identity'):
        super(NaiveMultiLayerPerceptron, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act_func = getattr(nn, hidden_act_func)()
        self.out_act_func = getattr(nn, out_act_func)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self.layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self.layers.append(self.out_act_func)
            else:
                self.layers.append(self.hidden_act_func)

    def forward(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs


if __name__ == '__main__':
    net = NaiveMultiLayerPerceptron(10, 1, [20, 12], 'ReLU', 'Identity')
    print(net)

    xs = torch.randn(size=(12, 10))
    ys = net(xs)
    print(ys)
