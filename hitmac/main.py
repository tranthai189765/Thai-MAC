"""Training entry point.

Launch with, e.g.::

    python -m hitmac.main --env Pose-v1 --model multi-att-shap --workers 6

A single evaluator process scores and checkpoints the shared model while
``--workers`` worker processes collect experience and update it asynchronously
(A3C).
"""

from __future__ import print_function, division

import os
import time
from datetime import datetime

import torch
import torch.multiprocessing as mp

from .arguments import parse_args
from .train import train
from .evaluate import evaluate
from .model import build_model
from .environment import make_env
from .shared_optim import SharedRMSprop, SharedAdam

os.environ["OMP_NUM_THREADS"] = "1"


def start():
    args = parse_args()
    args.shared_optimizer = True

    if args.gpu_ids == -1:
        torch.manual_seed(args.seed)
        args.gpu_ids = [-1]
        device_share = torch.device('cpu')
        mp.set_start_method('spawn')
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn', force=True)
        if len(args.gpu_ids) > 1:
            device_share = torch.device('cpu')
        else:
            device_share = torch.device('cuda:' + str(args.gpu_ids[-1]))

    # Build the shared model on the parameter-server device and share its memory.
    env = make_env(args.env, args)
    shared_model = build_model(env, args, device_share).to(device_share)
    shared_model.share_memory()
    env.close()
    del env

    if args.load_coordinator_dir is not None:
        saved_state = torch.load(args.load_coordinator_dir, map_location=lambda storage, loc: storage)
        if args.load_coordinator_dir.endswith('pth'):
            shared_model.load_state_dict(saved_state['model'], strict=False)
        else:
            shared_model.load_state_dict(saved_state)

    params = shared_model.parameters()
    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(params, lr=args.lr)
        elif args.optimizer == 'Adam':
            optimizer = SharedAdam(params, lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    current_time = datetime.now().strftime('%b%d_%H-%M')
    args.log_dir = os.path.join(args.log_dir, args.env, current_time)

    manager = mp.Manager()
    train_modes = manager.list()
    n_iters = manager.list()
    processes = []

    # Evaluator process (also responsible for checkpointing).
    p = mp.Process(target=evaluate, args=(args, shared_model, optimizer, train_modes, n_iters))
    p.start()
    processes.append(p)
    time.sleep(args.sleep_time)

    # Asynchronous training workers.
    for rank in range(args.workers):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer, train_modes, n_iters))
        p.start()
        processes.append(p)
        time.sleep(args.sleep_time)

    for p in processes:
        time.sleep(args.sleep_time)
        p.join()


if __name__ == '__main__':
    start()
