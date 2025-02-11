import numpy as np
import torch as T

def camera_observation_overs_targets(env, input):
    num_targets = env.num_targets
    rows = num_targets
    columns = 5  # Mỗi mục tiêu có 5 thông số
    x_axis_of_camera = input[13]
    y_axis_of_camera = input[14]
    output = np.zeros((rows, columns), dtype=np.float64)

    for i in range(rows):
        start_idx = 22 + 5 * i
        end_idx = start_idx + 5
        output[i, :] = input[start_idx:end_idx]
        if output[i, 4] == 1:
            output[i, 0] -= x_axis_of_camera
            output[i, 1] -= y_axis_of_camera

    # Chuyển numpy array thành tensor
    output = T.tensor(output, dtype=T.float64, requires_grad=True)

    return output

def joint_camera_observation_over_targets(env, input):
    num_cameras = env.num_cameras
    num_targets = env.num_targets
    columns = 5  # Mỗi mục tiêu có 5 thông số
    rows = num_targets
    depths = num_cameras

    # Tạo tensor 3D ban đầu
    output = T.zeros((depths, rows, columns), dtype=T.float64)

    for i in range(depths):
        output[i, :, :] = camera_observation_overs_targets(env, input[i])

    return output

def camera_observation_overs_targets_no_normalize(env, input):
    num_targets = env.num_targets
    rows = num_targets
    columns = 5  # Mỗi mục tiêu có 5 thông số
    x_axis_of_camera = input[13]
    y_axis_of_camera = input[14]
    output = np.zeros((rows, columns), dtype=np.float64)

    for i in range(rows):
        start_idx = 22 + 5 * i
        end_idx = start_idx + 5
        output[i, :] = input[start_idx:end_idx]

    # Chuyển numpy array thành tensor
    output = T.tensor(output, dtype=T.float64, requires_grad=True)

    return output

def joint_camera_observation_over_targets_no_normalize(env, input):
    num_cameras = env.num_cameras
    num_targets = env.num_targets
    columns = 5  # Mỗi mục tiêu có 5 thông số
    rows = num_targets
    depths = num_cameras

    # Tạo tensor 3D ban đầu
    output = T.zeros((depths, rows, columns), dtype=T.float64)

    for i in range(depths):
        output[i, :, :] = camera_observation_overs_targets_no_normalize(env, input[i])
    
    output = output.detach().numpy()

    return output

def camera_observation_overs_obstacles(env, input):
    num_obstacles = env.num_obstacles
    num_targets = env.num_targets
    rows = num_obstacles
    columns = 4  # Mỗi chướng ngại vật có 4 thông số
    x_axis_of_camera = input[13]
    y_axis_of_camera = input[14]
    output = np.zeros((rows, columns), dtype=np.float64)

    for i in range(rows):
        start_idx = 22 + 5 * num_targets + 4 * i
        end_idx = start_idx + 4
        # Sử dụng detach để chuyển đổi sang numpy
        output[i, :] = input[start_idx:end_idx]
        if output[i, 3] == 1:
            output[i, 0] -= x_axis_of_camera
            output[i, 1] -= y_axis_of_camera

    # Chuyển numpy array thành tensor
    output = T.tensor(output, dtype=T.float64, requires_grad=True)

    return output

def joint_camera_observation_over_obstacles(env, input):
    num_cameras = env.num_cameras
    num_obstacles = env.num_obstacles
    columns = 4  # Mỗi chướng ngại vật có 4 thông số
    rows = num_obstacles
    depths = num_cameras

    # Tạo tensor 3D ban đầu
    output = T.zeros((depths, rows, columns), dtype=T.float64)

    for i in range(depths):
        output[i, :, :] = camera_observation_overs_obstacles(env, input[i])

    return output