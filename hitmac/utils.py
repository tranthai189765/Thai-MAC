"""General-purpose helpers: logging, weight initialization and gradient sharing."""

from __future__ import division

import math
import json
import logging

import torch
import numpy as np
from torch.autograd import Variable


def setup_logger(logger_name, log_file, level=logging.INFO):
    """Attach a file and stream handler to a named logger."""
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def read_config(file_path):
    """Read a JSON configuration file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def norm_col_init(weights, std=1.0):
    """Column-normalized weight initialization used for actor/critic heads."""
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, device, device_share):
    """Copy local gradients into the shared model (A3C parameter server)."""
    diff_device = device != device_share
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not diff_device:
            return
        elif not diff_device:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.to(device_share)


def ensure_shared_grads_param(params, shared_params, gpu=False):
    """Copy gradients between two iterables of parameters."""
    for param, shared_param in zip(params, shared_params):
        if shared_param.grad is not None and not gpu:
            return
        if not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.clone().cpu()


def weights_init(m):
    """Fan-in/fan-out uniform initialization for Conv and Linear layers."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def weights_init_mlp(m):
    """Row-normalized normal initialization for MLP layers."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal(x, mu, sigma, device):
    """Gaussian probability density of ``x`` under ``N(mu, sigma)``."""
    pi = torch.from_numpy(np.array([math.pi])).float()
    pi = Variable(pi).to(device)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b


def check_path(path):
    """Create ``path`` if it does not already exist."""
    import os
    if not os.path.exists(path):
        os.mkdir(path)


def goal_id_filter(goals):
    """Return the indices of goals that are switched on (value > 0.5)."""
    return np.where(goals > 0.5)[0]


def norm(x, scale):
    """Standardize ``x`` along the batch dimension and rescale."""
    assert len(x.shape) <= 2
    # Normalize with batch mean and std; the epsilon prevents division by zero.
    x = scale * (x - x.mean(0)) / (x.std(0) + 1e-6)
    return x


class ToTensor(object):
    """Convert an ``(N, H, W, C)`` image batch to a ``(N, C, H, W)`` float tensor."""

    def __call__(self, sample):
        sample = sample.transpose(0, 3, 1, 2)
        return torch.from_numpy(sample.astype(np.float32))
