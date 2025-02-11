import mate
import numpy as np
import torch
import math
import changed_observation as changed_observation

K = 10
MAX_STEPS = 100

class Env_Executors:
    def __init__(self):
        env = mate.make('MultiAgentTracking-v0')
        env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
        self.env = env    
        self.n = self.env.num_teammates
        self.num_target = self.env.num_opponents
        self.goals4cam = np.ones([self.n, self.num_target])
        self.count_steps = 0

        # linh tinh
        self.num_cameras = self.n
        self.num_targets = self.num_target
        self.num_obstacles = self.env.num_obstacles
        self.action_space = self.env.action_space
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

    def reset(self):
        self.count_steps = 0
        self.goals4cam = np.ones([self.n, self.num_target])

        return self.env.reset()
    
    def seed(self, para):
        self.env.seed(para)
    
    def step(self, actions):
        state_multi, reward_multi, done, cameras_info = self.env.step(actions)

        # take observation over targets
        if isinstance(state_multi, torch.Tensor):
             new_state_multi = state_multi.detach().cpu().numpy()  # Đưa về CPU trước khi chuyển sang NumPy
        else:
             new_state_multi = state_multi  # Giữ nguyên nếu không phải tensor

        obs_over_targets = changed_observation.joint_camera_observation_over_targets(self.env, new_state_multi) 
        obs_over_targets = obs_over_targets.detach().numpy()

        self.count_steps += 1

        new_reward_multi = self.multi_reward(state_multi, obs_over_targets, self.goals4cam)

        if self.count_steps > MAX_STEPS:
            done = True

        if self.count_steps % K == 0:
            self.reset_goalmap(obs_over_targets)

        return state_multi, new_reward_multi, done, cameras_info
    
    def reset_goalmap(self, obs_over_targets):
        num_features = obs_over_targets.shape[2]
        for i in range(self.n):
            for j in range(self.num_target):
                self.goals4cam[i][j] = obs_over_targets[i,j, num_features-1]
    

    def get_hori_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]

        angle_now = np.arctan2(y_delt, x_delt)/ np.pi * 180 - current_pose[2]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360

        return angle_now
    
    
    def angle_reward(self, angle_h, current_view, real_visible):
        hori_reward = 1 - abs(angle_h) / current_view
        if real_visible:
            reward = np.clip(hori_reward, -1, 1)
        else:
            reward = -1
        
        return reward
    
    def multi_reward(self, obs, obs_over_targets, goals4cam):
        # generate reward
        camera_local_rewards = []
        num_features = obs_over_targets.shape[2]

        for i in range(self.n):
            local_rewards = []
            for j in range(self.num_target):
                # camera location
                cam_pose = [0, 0, 0]
                cam_pose[0] = obs[i, 13]
                cam_pose[1] = obs[i, 14]
                
                # cam direction
                x = obs[i, 16]
                y = obs[i, 17]
                phi = math.atan2(x, y)
                cam_pose[2] = math.degrees(phi)

                # camera view (degree)
                cam_view = math.degrees(obs[i, 18])

                # real_visible 
                if obs_over_targets[i, j, num_features-1] > 0:
                    real_visible = True
                else:
                    real_visible = False

                # target location
                target_pose = [0, 0]
                target_pose[0] = obs_over_targets[i, j, 0] + cam_pose[0]
                target_pose[1] = obs_over_targets[i, j, 1] + cam_pose[1]

                angle_h = self.get_hori_direction(cam_pose, target_pose)

                reward = self.angle_reward(angle_h, cam_view, real_visible)

                if goals4cam is None and real_visible or goals4cam is not None and goals4cam[i][j] > 0:
                    local_rewards.append(reward)

            camera_local_rewards.append(np.mean(local_rewards) if len(local_rewards) > 0 else 0)
            camera_local = camera_local_rewards

        return camera_local





            



    
