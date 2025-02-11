from __future__ import division


def create_env(env_id, args, rank=-1):
    if 'v0' in env_id:
        from changed_environment_executors import Env_Executors as poseEnv
        env = poseEnv()
    else:  # 'v1' in env_id:
        from changed_environment_coordinator import Env_Coordinator as poseEnv
        env = poseEnv()

    return env