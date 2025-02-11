import changed_observation as changed_observation
import numpy as np

def target_pose_array(env, input):
    input = changed_observation.joint_camera_observation_over_targets_no_normalize(env, input)
    # print("next_input = ", input)
    num_cameras = input.shape[0]
    num_targets = input.shape[1]
    num_feature = input.shape[2]
    output = np.zeros((num_targets, 2), dtype=np.float64)
    fill = np.zeros(num_targets, dtype = int)
    for i in range(num_cameras):
        for j in range(num_targets):
            output[j, 0] = -np.inf
            output[j, 1] = -np.inf
            if(input[i, j, num_feature-1] > 0 and fill[j] == 0):
                output[j, 0] = input[i, j, 0]
                output[j, 1] = input[i, j, 1]
                fill[j] = 1
    
    return output