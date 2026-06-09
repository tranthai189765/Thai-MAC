"""Coordinator environment (Pose-v1): high-level target assignment.

Wraps the single-team MATE camera environment. The coordinator emits a binary
goal map (which targets each camera should attend to); the low-level control is
produced either by a hand-crafted baseline controller (``slave_rule=True``) or
by a trained executor policy (:class:`~hitmac.model.A3C_Single`). The team
reward is the realized target coverage rate.
"""

import math

import mate
import torch
import numpy as np

from . import observation
from .model import A3C_Single
from .observation import target_pose_array
from .utils import goal_id_filter

# Mean field-of-view (degrees) the baseline controller drives each camera toward.
VIEW_MEAN_DEGREES = 45
MAX_STEPS = 100


class CoordinatorEnv:
    """Single-team camera environment driven by a high-level goal-assignment policy."""

    def __init__(self, args):
        env = mate.make('MultiAgentTracking-v0')
        env = mate.MultiCamera(env, target_agent=mate.GreedyTargetAgent(seed=0))
        self.env = env
        self.n = self.env.num_teammates
        self.num_target = self.env.num_opponents
        self.count_steps = 0

        # Aliases used throughout the model / observation code.
        self.num_cameras = self.n
        self.num_targets = self.num_target
        self.num_obstacles = self.env.num_obstacles
        self.action_space = self.env.action_space

        self.state = self.env.reset()

        # Use the rule-based controller unless a trained executor is provided.
        self.slave_rule = args.load_executor_dir is None
        if not self.slave_rule:
            self.device = torch.device('cpu')
            self.slave = A3C_Single(env, args).to(self.device)
            saved_state = torch.load(
                args.load_executor_dir,
                map_location=lambda storage, loc: storage)
            self.slave.load_state_dict(saved_state['model'], strict=False)
            self.slave.eval()

    def reset(self):
        self.goals4cam = np.ones([self.num_cameras, self.num_targets])
        self.count_steps = 0
        self.state = self.env.reset()
        return self.env.reset()

    def get_hori_direction(self, current_pose, target_pose):
        """Signed horizontal angle (degrees) from a camera's heading to a target."""
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180 - current_pose[2]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now

    def get_baseline_action(self, goals, i, state_multi):
        """Low-level (pan, zoom) action for camera ``i`` given its assigned goals."""
        target_pos_list = target_pose_array(self.env, state_multi)

        # Camera position.
        cam_pose = [0, 0, 0]
        cam_pose[0] = state_multi[i, 13]
        cam_pose[1] = state_multi[i, 14]

        # Camera heading.
        x = state_multi[i, 16]
        y = state_multi[i, 17]
        phi = math.atan2(x, y)
        cam_pose[2] = math.degrees(phi)

        box_high = self.env.action_space[0].high
        box_low = self.env.action_space[0].low

        # Current field of view of the camera (degrees).
        current_view = math.degrees(state_multi[i, 18])

        # Keep only this camera's assigned goals that are currently localized.
        goal_ids = goal_id_filter(goals)
        if len(goal_ids) != 0:
            goal_ids = [index for index in goal_ids if target_pos_list[index][0] != -np.inf]

        if len(goal_ids) != 0:
            if self.slave_rule:
                target_pose = (target_pos_list[goal_ids]).mean(axis=0)
                angle_h = self.get_hori_direction(cam_pose, target_pose)
                view_h = VIEW_MEAN_DEGREES - current_view
                action_camera = np.clip([angle_h, view_h], box_low, box_high)
            else:
                _, actions, _, _ = self.slave(state_multi, test=True)
                action_camera = actions[i]
        else:
            angle_h = 100
            view_h = VIEW_MEAN_DEGREES - current_view
            action_camera = np.clip([angle_h, view_h], box_low, box_high)

        return action_camera

    def step(self, actions):
        self.goals4cam = np.squeeze(actions)
        gr, state_multi, cameras_info = self.simulate(self.goals4cam, keep=10)
        self.state = state_multi
        self.count_steps += 1
        done = self.count_steps > MAX_STEPS
        return state_multi, gr, done, cameras_info

    def simulate(self, GoalMap, keep=-1):
        """Roll the low-level controllers forward for ``keep`` steps under a fixed goal map."""
        state_multi = self.state
        gre = np.array([0.0])
        cameras_info = None
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
