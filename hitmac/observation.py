"""Observation parsing for the MATE multi-camera tracking environment.

The raw per-camera observation is a flat vector that packs the camera's own
state followed by the per-target and per-obstacle entries. These helpers slice
that vector into structured ``[num_cameras, num_entities, num_features]``
tensors and, optionally, express target/obstacle positions relative to the
observing camera.

Per-target feature layout (5 values): ``[x, y, ..., ..., visible_flag]``.
Per-obstacle feature layout (4 values): ``[x, y, ..., visible_flag]``.
"""

import numpy as np
import torch as T

# Indices of the camera's own (x, y) position inside the raw observation vector.
CAMERA_X_INDEX = 13
CAMERA_Y_INDEX = 14
# Offset at which the per-target block starts in the raw observation vector.
TARGET_BLOCK_OFFSET = 22
TARGET_FEATURES = 5
OBSTACLE_FEATURES = 4


def camera_observation_overs_targets(env, input):
    """Parse one camera's observation into a ``[num_targets, 5]`` tensor.

    Visible targets have their ``(x, y)`` shifted into the camera's frame.
    """
    num_targets = env.num_targets
    x_axis_of_camera = input[CAMERA_X_INDEX]
    y_axis_of_camera = input[CAMERA_Y_INDEX]
    output = np.zeros((num_targets, TARGET_FEATURES), dtype=np.float64)

    for i in range(num_targets):
        start_idx = TARGET_BLOCK_OFFSET + TARGET_FEATURES * i
        end_idx = start_idx + TARGET_FEATURES
        output[i, :] = input[start_idx:end_idx]
        if output[i, 4] == 1:  # target is visible -> use camera-relative coords
            output[i, 0] -= x_axis_of_camera
            output[i, 1] -= y_axis_of_camera

    return T.tensor(output, dtype=T.float64, requires_grad=True)


def joint_camera_observation_over_targets(env, input):
    """Stack target observations of all cameras into ``[num_cameras, num_targets, 5]``."""
    num_cameras = env.num_cameras
    num_targets = env.num_targets

    output = T.zeros((num_cameras, num_targets, TARGET_FEATURES), dtype=T.float64)
    for i in range(num_cameras):
        output[i, :, :] = camera_observation_overs_targets(env, input[i])
    return output


def camera_observation_overs_targets_no_normalize(env, input):
    """Like :func:`camera_observation_overs_targets` but in absolute coordinates."""
    num_targets = env.num_targets
    output = np.zeros((num_targets, TARGET_FEATURES), dtype=np.float64)

    for i in range(num_targets):
        start_idx = TARGET_BLOCK_OFFSET + TARGET_FEATURES * i
        end_idx = start_idx + TARGET_FEATURES
        output[i, :] = input[start_idx:end_idx]

    return T.tensor(output, dtype=T.float64, requires_grad=True)


def joint_camera_observation_over_targets_no_normalize(env, input):
    """Absolute-coordinate variant returned as a NumPy array."""
    num_cameras = env.num_cameras
    num_targets = env.num_targets

    output = T.zeros((num_cameras, num_targets, TARGET_FEATURES), dtype=T.float64)
    for i in range(num_cameras):
        output[i, :, :] = camera_observation_overs_targets_no_normalize(env, input[i])
    return output.detach().numpy()


def camera_observation_overs_obstacles(env, input):
    """Parse one camera's observation into a ``[num_obstacles, 4]`` tensor.

    Visible obstacles have their ``(x, y)`` shifted into the camera's frame.
    """
    num_obstacles = env.num_obstacles
    num_targets = env.num_targets
    x_axis_of_camera = input[CAMERA_X_INDEX]
    y_axis_of_camera = input[CAMERA_Y_INDEX]
    output = np.zeros((num_obstacles, OBSTACLE_FEATURES), dtype=np.float64)

    obstacle_block_offset = TARGET_BLOCK_OFFSET + TARGET_FEATURES * num_targets
    for i in range(num_obstacles):
        start_idx = obstacle_block_offset + OBSTACLE_FEATURES * i
        end_idx = start_idx + OBSTACLE_FEATURES
        output[i, :] = input[start_idx:end_idx]
        if output[i, 3] == 1:  # obstacle is visible -> use camera-relative coords
            output[i, 0] -= x_axis_of_camera
            output[i, 1] -= y_axis_of_camera

    return T.tensor(output, dtype=T.float64, requires_grad=True)


def joint_camera_observation_over_obstacles(env, input):
    """Stack obstacle observations of all cameras into ``[num_cameras, num_obstacles, 4]``."""
    num_cameras = env.num_cameras
    num_obstacles = env.num_obstacles

    output = T.zeros((num_cameras, num_obstacles, OBSTACLE_FEATURES), dtype=T.float64)
    for i in range(num_cameras):
        output[i, :, :] = camera_observation_overs_obstacles(env, input[i])
    return output


def target_pose_array(env, input):
    """Return one absolute ``(x, y)`` per target, or ``(-inf, -inf)`` if unseen.

    A target's pose is taken from the first camera that currently sees it.
    """
    obs = joint_camera_observation_over_targets_no_normalize(env, input)
    num_cameras, num_targets, num_feature = obs.shape

    output = np.zeros((num_targets, 2), dtype=np.float64)
    filled = np.zeros(num_targets, dtype=int)
    for i in range(num_cameras):
        for j in range(num_targets):
            output[j, 0] = -np.inf
            output[j, 1] = -np.inf
            if obs[i, j, num_feature - 1] > 0 and filled[j] == 0:
                output[j, 0] = obs[i, j, 0]
                output[j, 1] = obs[i, j, 1]
                filled[j] = 1

    return output
