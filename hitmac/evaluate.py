"""Evaluation process: periodically scores the shared model and checkpoints it."""

from __future__ import division

import os
import time
import logging

import torch
import numpy as np
from tensorboardX import SummaryWriter
from setproctitle import setproctitle as ptitle

from .model import build_model
from .agent import Agent
from .environment import make_env
from .utils import setup_logger


def evaluate(args, shared_model, optimizer, train_modes, n_iters):
    """Continuously evaluate the shared model and save the best/latest checkpoint."""
    ptitle('Test Agent')
    n_iter = 0
    writer = SummaryWriter(os.path.join(args.log_dir, 'Test'))
    gpu_id = args.gpu_ids[-1]

    log_name = '{}_log'.format(args.env)
    setup_logger(log_name, r'{0}/logger'.format(args.log_dir))
    log = {log_name: logging.getLogger(log_name)}
    for k, v in vars(args).items():
        log[log_name].info('{0}: {1}'.format(k, v))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    env = make_env(args.env, args)
    env.seed(args.seed)
    start_time = time.time()
    count_eps = 0

    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.model = build_model(env, args, device).to(device)
    player.model.eval()
    max_score = -100

    while True:
        AG = 0
        reward_sum = np.zeros(player.num_agents)
        reward_sum_list = []
        len_sum = 0
        for _ in range(args.test_eps):
            player.model.load_state_dict(shared_model.state_dict())
            player.reset()
            reward_sum_ep = np.zeros(player.num_agents)
            rotation_sum_ep = 0
            fps_counter = 0
            t0 = time.time()
            count_eps += 1
            fps_all = []
            while True:
                player.action_test()
                fps_counter += 1
                reward_sum_ep += player.reward
                rotation_sum_ep += player.rotation
                if player.done:
                    AG += reward_sum_ep[0] / rotation_sum_ep * player.num_agents
                    reward_sum += reward_sum_ep
                    reward_sum_list.append(reward_sum_ep[0])
                    len_sum += player.eps_len
                    fps = fps_counter / (time.time() - t0)
                    n_iter = sum(n_iters)
                    for i, r_i in enumerate(reward_sum_ep):
                        writer.add_scalar('test/reward' + str(i), r_i, n_iter)
                    fps_all.append(fps)
                    writer.add_scalar('test/fps', fps, n_iter)
                    writer.add_scalar('test/eps_len', player.eps_len, n_iter)
                    break

        ave_AG = AG / args.test_eps
        ave_reward_sum = reward_sum / args.test_eps
        len_mean = len_sum / args.test_eps
        reward_step = reward_sum / len_sum
        mean_reward = np.mean(reward_sum_list)
        std_reward = np.std(reward_sum_list)

        log[log_name].info(
            "Time {0}, ave eps reward {1}, ave eps length {2}, reward step {3}, FPS {4}, "
            "mean reward {5}, std reward {6}, AG {7}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                np.around(ave_reward_sum, decimals=2), np.around(len_mean, decimals=2),
                np.around(reward_step, decimals=2), np.around(np.mean(fps_all), decimals=2),
                mean_reward, std_reward, np.around(ave_AG, decimals=2)))

        # Keep the best model so far; otherwise overwrite the latest snapshot.
        if ave_reward_sum[0] >= max_score:
            print('save best!')
            max_score = ave_reward_sum[0]
            model_dir = os.path.join(args.log_dir, 'best.pth')
        else:
            model_dir = os.path.join(args.log_dir, 'new.pth')
        state_to_save = {"model": player.model.state_dict(),
                         "optimizer": optimizer.state_dict()}
        torch.save(state_to_save, model_dir)

        time.sleep(args.sleep_time)
        if n_iter > args.max_step:
            env.close()
            for idx in range(args.workers):
                train_modes[idx] = -100
            break
