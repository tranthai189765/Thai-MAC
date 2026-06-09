"""Environment factory.

Maps an environment id to the matching single-team MATE wrapper:

* ``Pose-v0`` -> :class:`~hitmac.env_executor.ExecutorEnv`
  (low-level per-camera control)
* ``Pose-v1`` -> :class:`~hitmac.env_coordinator.CoordinatorEnv`
  (high-level target assignment)
"""

from __future__ import division


def make_env(env_id, args, rank=-1):
    """Create the environment selected by ``env_id``."""
    if 'v0' in env_id:
        from .env_executor import ExecutorEnv
        return ExecutorEnv(args)

    from .env_coordinator import CoordinatorEnv
    return CoordinatorEnv(args)
