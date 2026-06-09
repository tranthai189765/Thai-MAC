"""Neural network building blocks: noisy layers, recurrent encoders and attention."""

import os
import json
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable


class NoisyLinear(nn.Linear):
    """Linear layer with factorized Gaussian noise (NoisyNet) for exploration."""

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = Parameter(torch.Tensor(out_features))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):
            init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.constant_(self.sigma_weight, self.sigma_init)
            init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight),
                        self.bias + self.sigma_bias * Variable(self.epsilon_bias))

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class BiRNN(torch.nn.Module):
    """Bidirectional LSTM/GRU sequence encoder."""

    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = 'lstm' in head_name
        if self.lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True).to(device)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True).to(device)
        self.feature_dim = hidden_size * 2
        self.device = device

    def forward(self, x, state=None):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        if self.lstm:
            out, (_, hn) = self.rnn(x, (h0, c0))
        else:
            out, hn = self.rnn(x, h0)
        return out, hn


class RNN(torch.nn.Module):
    """Unidirectional LSTM/GRU sequence encoder."""

    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = 'lstm' in head_name
        if self.lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.feature_dim = hidden_size
        self.device = device

    def forward(self, x, state=None):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        if self.lstm:
            out, (_, hn) = self.rnn(x, (h0, c0))
        else:
            out, hn = self.rnn(x, h0)
        return out, hn


def xavier_init(layer):
    """Xavier-uniform initialize a linear layer in place and return it."""
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.constant_(layer.bias, 0)
    return layer


class AttentionLayer(torch.nn.Module):
    """Self-attention over a set of entity embeddings.

    Given an input of shape ``[batch, num_entities, feature_dim]`` it returns
    the per-entity attended features and their sum (a permutation-invariant
    global feature), which lets the policy reason over a variable number of
    targets/obstacles.
    """

    def __init__(self, feature_dim, weight_dim, device):
        super(AttentionLayer, self).__init__()
        self.in_dim = feature_dim
        self.device = device

        self.Q = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.K = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.V = xavier_init(nn.Linear(self.in_dim, weight_dim))

        self.feature_dim = weight_dim

    def forward(self, x):
        """Compute ``softmax(Q K^T) V`` and a summed global feature.

        :param x: tensor of shape ``[batch, num_entities, feature_dim]``
        :return: ``(z, global_feature)`` with ``z`` of shape
            ``[batch, num_entities, weight_dim]``
        """
        q = torch.tanh(self.Q(x))
        k = torch.tanh(self.K(x))
        v = torch.tanh(self.V(x))

        z = torch.bmm(F.softmax(torch.bmm(q, k.permute(0, 2, 1)), dim=2), v)
        global_feature = z.sum(dim=1)
        return z, global_feature

    def save_parameters(self, file_path):
        """Save layer parameters to a file."""
        torch.save(self.state_dict(), file_path)

    def load_parameters(self, file_path):
        """Load layer parameters from a file."""
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
        else:
            raise FileNotFoundError(f"Model parameters file not found: {file_path}")

    def save_parameters_JSON(self, file_path):
        """Save layer parameters to a JSON file."""
        params = {k: v.tolist() for k, v in self.state_dict().items()}
        with open(file_path, 'w') as f:
            json.dump(params, f)
