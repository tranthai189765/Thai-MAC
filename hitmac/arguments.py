"""Command-line arguments for training and evaluation."""

import argparse


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser shared by training and evaluation."""
    parser = argparse.ArgumentParser(
        description="HiT-MAC: hierarchical multi-agent coordination on MATE, trained with A3C."
    )

    # --- Optimization -----------------------------------------------------
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='reward discount factor')
    parser.add_argument('--tau', type=float, default=1.0, help='lambda parameter for GAE')
    parser.add_argument('--entropy', type=float, default=0.01, help='entropy regularization coefficient')
    parser.add_argument('--grad-entropy', type=float, default=1.0, help='entropy gradient coefficient')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'RMSprop'],
                        help='shared optimizer')
    parser.add_argument('--amsgrad', default=True, help='use the AMSGrad variant of Adam')
    parser.add_argument('--shared-optimizer', dest='shared_optimizer', action='store_true',
                        help='share optimizer statistics across workers')

    # --- A3C rollout ------------------------------------------------------
    parser.add_argument('--workers', type=int, default=1,
                        help='number of asynchronous training workers')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='number of forward steps per A3C update')
    parser.add_argument('--max-step', type=int, default=5_000_000,
                        help='maximum number of training steps')
    parser.add_argument('--test-eps', type=int, default=1,
                        help='number of episodes per evaluation round')

    # --- Environment / model ---------------------------------------------
    parser.add_argument('--env', default='Pose-v1',
                        help='environment id: Pose-v0 (executor) or Pose-v1 (coordinator)')
    parser.add_argument('--model', default='single',
                        help='policy variant, e.g. "single-att" (executor) or "multi-att-shap" (coordinator)')
    parser.add_argument('--input-size', type=int, default=80, help='observation input size')
    parser.add_argument('--lstm-out', type=int, default=256, help='hidden feature size')

    # --- Checkpoints / logging -------------------------------------------
    parser.add_argument('--load-coordinator-dir', default=None,
                        help='checkpoint to load the coordinator policy from')
    parser.add_argument('--load-executor-dir', default=None,
                        help='checkpoint to load the executor policy from')
    parser.add_argument('--log-dir', default='logs/',
                        help='directory for TensorBoard logs and checkpoints')

    # --- Hardware / reproducibility --------------------------------------
    parser.add_argument('--gpu-ids', type=int, default=-1, nargs='+',
                        help='GPU ids to use, -1 for CPU only')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--fix', dest='fix', action='store_true',
                        help='use a fixed random seed across all workers')

    # --- Miscellaneous ----------------------------------------------------
    parser.add_argument('--norm-reward', dest='norm_reward', action='store_true',
                        help='normalize rewards online')
    parser.add_argument('--render', dest='render', action='store_true',
                        help='render the environment during evaluation')
    parser.add_argument('--render-save', dest='render_save', action='store_true',
                        help='save rendered frames')
    parser.add_argument('--train-mode', type=int, default=-1, help='training mode flag')
    parser.add_argument('--sleep-time', type=int, default=0,
                        help='delay (seconds) between launching workers')

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return get_parser().parse_args(argv)
