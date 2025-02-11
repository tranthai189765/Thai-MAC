from changed_main import parser
from changed_model import A3C_Single
import changed_observation as changed_observation
import numpy as np
import mate
import torch
from utils import goal_id_filter
import math
from changed_take_target_pose import target_pose_array

args = parser.parse_args()

VIEW_MEAN_DEGREES = 45
MAX_STEPS = 100

class Env_Coordinator:
    def __init__(self, slave_rule = args.load_executor_dir is None):
        env = mate.make('MultiAgentTracking-v0')
        env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
        self.env = env
        self.n = self.env.num_teammates
        self.num_target = self.env.num_opponents
        self.count_steps = 0

        # linh tinh
        self.num_cameras = self.n
        self.num_targets = self.num_target
        self.num_obstacles = self.env.num_obstacles
        self.action_space = self.env.action_space

        # phải cài thêm state :)
        self.state = self.env.reset()

        self.slave_rule = slave_rule
        if not self.slave_rule:
            self.device = torch.device('cpu')
            self.slave = A3C_Single(env, args)
            self.slave = self.slave.to(self.device)
            saved_state = torch.load(
                args.load_executor_dir, # 'trainedModel/best_executor.pth',
                map_location = lambda storage, loc: storage)
            self.slave.load_state_dict(saved_state['model'], strict=False)
            self.slave.eval()
        
    def reset(self):
        self.goals4cam = np.ones([self.num_cameras, self.num_targets])
        self.count_steps = 0
        self.state = self.env.reset()

        return self.env.reset()
    def get_hori_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180 - current_pose[2]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now
    
    def get_baseline_action(self, goals, i, state_multi):
        # take target_pose 
        target_pos_list = target_pose_array(self.env, state_multi)

        # take camera pose
        cam_pose = [0, 0, 0]
        cam_pose[0] = state_multi[i, 13]
        cam_pose[1] = state_multi[i, 14]

        # cam direction
        x = state_multi[i, 16]
        y = state_multi[i, 17]
        phi = math.atan2(x, y)
        cam_pose[2] = math.degrees(phi)

        # take limitations
        box_high = self.env.action_space[0].high
        box_low = self.env.action_space[0].low
        
        # take current view of camera
        current_view = math.degrees(state_multi[i, 18])

        # lọc ra goals của camera
        goal_ids = goal_id_filter(goals)
        if len(goal_ids) != 0:
            goal_ids = [index for index in goal_ids if target_pos_list[index][0] != -np.inf]

        if len(goal_ids) !=0:
            if self.slave_rule:
                target_pose = (target_pos_list[goal_ids]).mean(axis=0)
                angle_h = self.get_hori_direction(cam_pose, target_pose)
                view_h = VIEW_MEAN_DEGREES - current_view
                action_camera = [angle_h, view_h]
                action_camera = np.clip(action_camera, box_low, box_high)
            else:
                values, actions, entropies, log_probs = self.slave(state_multi, test=True)
                action_camera = actions[i]
        else:
            angle_h = 100
            view_h = VIEW_MEAN_DEGREES - current_view
            action_camera = [angle_h, view_h]
            action_camera = np.clip(action_camera, box_low, box_high)
        
        return action_camera

    def step(self, actions):
        self.goals4cam = np.squeeze(actions)
        gr, state_multi, cameras_info = self.simulate(self.goals4cam, keep=10)
        self.state = state_multi
        self.count_steps += 1
        done = False

        if self.count_steps > MAX_STEPS:
            done = True
        
        return state_multi, gr, done, cameras_info


    def simulate(self, GoalMap, keep=-1):
        state_multi = self.state
        gre = np.array([0.0])
        for _ in range(keep):
            actions = []
            for i in range(self.num_cameras):
                action_camera = self.get_baseline_action(GoalMap[i], i, state_multi)
                actions.append(action_camera)
            
            new_state, _, _, cameras_info = self.env.step(actions)
            state_multi = new_state
            coor_reward = cameras_info[0]['real_coverage_rate']
            gre += coor_reward
            self.state = state_multi
        
        return gre / keep, state_multi, cameras_info
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
    
    def seed(self, para):
        self.env.seed(para)
