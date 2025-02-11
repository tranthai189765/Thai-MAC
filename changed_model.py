from __future__ import division
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from gym import spaces
import time

from perception import NoisyLinear, AttentionLayer
from utils import norm_col_init
import changed_observation as changed

def build_model(env, args, device):
    name = args.model

    if 'single' in name:
        model = A3C_Single(env, args, device)
    elif 'multi' in name:
        # print("this")
        model = A3C_Multi(env, args, device)

    model.train()
    
    return model


def sample_action(mu_multi, sigma_multi, device, test=False):
    # discrete
    logit = mu_multi
    prob = F.softmax(logit, dim=-1)
    log_prob = F.log_softmax(logit, dim=-1)
    entropy = -(log_prob * prob).sum(-1, keepdim=True)
    if test:
        action = prob.max(-1)[1].data
        action_env = action.cpu().numpy()  # np.squeeze(action.cpu().numpy(), axis=0)
    else:
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))  # [num_agent, 1] # comment for sl slave
        action_env = action.squeeze(0)

    return action_env, entropy, log_prob

class PolicyNet_Coor(nn.Module):
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNet_Coor, self).__init__()
        self.head_name = head_name
        self.device = device
        num_outputs = action_space.n

        if 'ns' in head_name:
            self.noise = True
            self.actor_linear = NoisyLinear(input_dim, num_outputs, sigma_init=0.017)
        else:
            self.noise = False
            self.actor_linear = nn.Linear(input_dim, num_outputs)

            # init layers
            self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.1)
            self.actor_linear.bias.data.fill_(0)

    def forward(self, x, test=False):
        mu = F.relu(self.actor_linear(x))
        sigma = torch.ones_like(mu)
        action, entropy, log_prob = sample_action(mu, sigma, self.device, test)
        return action, entropy, log_prob

    def sample_noise(self):
        if self.noise:
            self.actor_linear.sample_noise()
            self.actor_linear2.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.actor_linear.sample_noise()
            self.actor_linear2.sample_noise()

class PolicyNet(nn.Module):
    def __init__(self, input_dim, env, head_name, device):
        super(PolicyNet, self).__init__()
        self.head_name = head_name
        self.device = device
        self.env = env
        output_dim = 2

        self.sigma_linear = nn.Linear(input_dim, output_dim)
        self.mu_linear = nn.Linear(input_dim, output_dim)
        # init layers
        self.sigma_linear.weight.data = norm_col_init(self.sigma_linear.weight.data, 0.1)
        self.sigma_linear.bias.data.fill_(0)
        
        self.mu_linear.weight.data = norm_col_init(self.mu_linear.weight.data, 0.1)
        self.mu_linear.bias.data.fill_(0)
    
    def forward(self, x, test=False):
        mu = 2 * torch.tanh(self.mu_linear(x))
        sigma = F.softplus(self.sigma_linear(x)) + 0.001
        action, entropy, log_prob = self.choose_action(mu, sigma, test)

        return action, entropy, log_prob
    
    def choose_action(self, mu, sigma, test=False):
        box_high = self.env.action_space[0].high
        box_low = self.env.action_space[0].low
        print("mu = ", mu)
        m = torch.distributions.Normal(mu, sigma)
        if test:
            action_env = mu.numpy()
        else:
            action_env = m.sample().numpy()
        log_prob = m.log_prob(torch.tensor(action_env, dtype=torch.float32)).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)).sum(dim=-1, keepdim=True)
        action_env = np.clip(action_env, box_low, box_high)
        action_env = tuple(map(lambda x: x, action_env))

        return action_env, entropy, log_prob

class ValueNet(nn.Module):
    def __init__(self, input_dim, head_name, output_dim=1):
        super(ValueNet, self).__init__()
        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, output_dim, sigma_init=0.017)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, output_dim)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)
    
    def forward(self, x):
        value = self.critic_linear(x)

        return value
    
    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()
    
    def remove_noise(self):
        if self.noise:
            self.critic_linear.remove_noise()

class ValueNet_Coor(nn.Module):
    def __init__(self, input_dim, head_name, num=1):
        super(ValueNet_Coor, self).__init__()
        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

    def forward(self, x):
        value = self.critic_linear(x)
        return value

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()


class AMCValueNet(nn.Module):
    def __init__(self, input_dim, head_name, num=1, device=torch.device('cpu')):
        super(AMCValueNet, self).__init__()
        self.head_name = head_name
        self.device = device

        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        if 'onlyJ' in head_name:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(2 * input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

            self.attention = AttentionLayer(input_dim, input_dim, device)
        self.feature_dim = input_dim

    def forward(self, x, goal):
        _, feature_dim = x.shape
        value = []

        coalition = x.view(-1, feature_dim)
        n = coalition.shape[0]

        feature = torch.zeros([self.feature_dim]).to(self.device)
        value.append(self.critic_linear(torch.cat([feature, coalition[0]])))
        for j in range(1, n):
            _, feature = self.attention(coalition[:j].unsqueeze(0))
            value.append(self.critic_linear(torch.cat([feature.squeeze(), coalition[j]])))  # delta f = f[:j]-f[:j-1]

        # mean and sum
        value = torch.cat(value).sum()

        return value.unsqueeze(0)

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()


class A3C_Single(torch.nn.Module):
    def __init__(self, environment, args, device=torch.device('cpu')):
        super(A3C_Single, self).__init__()
        self.head_name = args.model
        self.n = 4 
        self.env = environment
        self.device = device
        self.obs_dim = 5
        self.osc_dim = 4
        lstm_out = args.lstm_out

        self.encoder_targets = AttentionLayer(self.obs_dim, int(lstm_out/2), device)
        self.encoder_obstacles = AttentionLayer(self.osc_dim, int(lstm_out/2), device)

        self.critic = ValueNet(lstm_out, self.head_name, 1)
        self.actor = PolicyNet(lstm_out, self.env, self.head_name, device)

        self.train()
    
    def forward(self, x, test=False):
        input_targets = changed.joint_camera_observation_over_targets(self.env, x)
        input_obstacles = changed.joint_camera_observation_over_obstacles(self.env, x)

        if test:
            data_targets = Variable(input_targets).to(torch.float32)
            data_obstacles = Variable(input_obstacles).to(torch.float32)
        else:
            data_targets = Variable(input_targets, requires_grad = True).to(torch.float32)
            data_obstacles = Variable(input_obstacles, requires_grad = True).to(torch.float32)

        _, feature_targets = self.encoder_targets(data_targets)
        _, feature_obstacles = self.encoder_obstacles(data_obstacles)
        print("feature_targets.shape = ", feature_targets.shape)
        print("feature_obstacles.shape = ",feature_obstacles.shape)
        feature = torch.cat((feature_targets, feature_obstacles), dim=-1)

        actions, entropies, log_probs = self.actor(feature, test)
        values = self.critic(feature)

        return values, actions, entropies, log_probs
    
    def sample_noise(self):
        self.critic.sample_noise()
    
    def remove_noise(self):
        self.critic.remove_noise()

class EncodeLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out=32, head_name= 'lstm', device=None):
        super(EncodeLinear, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(inplace=True)
        )

        self.head_name = head_name
        self.feature_dim = dim_out
        self.train()
    
    def forward(self, inputs):
        # print("inputs.shape = ", inputs.shape)
        x = inputs
        feature = self.features(x)

        return feature
    
class A3C_Multi(torch.nn.Module):
    def __init__(self, env, args, device=torch.device('cpu')):
        super(A3C_Multi, self).__init__()
        self.env = env
        # self.num_agents = self.env.num_cameras
        self.obs_dim = 5
        self.num_agents = 4
        self.osc_dim = 4
        # self.num_obstacles = 5
        self.num_targets = self.env.num_targets
        
        lstm_out = args.lstm_out
        head_name = args.model 
        self.head_name  = head_name

        self.encoder_targets = EncodeLinear(self.obs_dim, lstm_out, head_name, device)
        # self.encoder_obstacles = EncodeLinear(self.osc_dim, int(lstm_out/2), head_name, device)
        feature_dim = self.encoder_targets.feature_dim

        self.attention_targets = AttentionLayer(feature_dim, lstm_out, device)
        # self.attention_obstacles = AttentionLayer(feature_dim, int(lstm_out/2), head_name)
        feature_dim = self.attention_targets.feature_dim

        self.actor = PolicyNet_Coor(feature_dim, spaces.Discrete(2), head_name, device)
        if 'shap' in head_name:
            self.ShapleyVcritic = AMCValueNet(feature_dim, head_name, 1, device)
        else:
            self.critic = ValueNet_Coor(feature_dim, head_name, 1)

        self.train()
        self.device = device
    
    def forward(self, x, test=False):
        input_targets = changed.joint_camera_observation_over_targets(self.env, x)
        # input_obstacles = changed.joint_camera_observation_over_obstacles(self.env, x)

        if test:
            data_targets = Variable(input_targets).to(torch.float32)
            # data_obstacles = Variable(input_obstacles).to(torch.float32)
        else:
            data_targets = Variable(input_targets, requires_grad = True).to(torch.float32)
            # data_obstacles = Variable(input_obstacles, requires_grad = True).to(torch.float32)
        
        #print("data_targets = ", data_targets)
        feature_targets = self.encoder_targets(data_targets)
        # print("done targets")
        # feature_obstacles = self.encoder_obstacles(data_obstacles)
        # print("feature_targets.shape = ", feature_targets.shape)
        # print("feature_obstacles.shape = ",feature_obstacles.shape)

        feature_targets = feature_targets.reshape(-1, self.encoder_targets.feature_dim).unsqueeze(0)
        # feature_obstacles = feature_obstacles.reshape(-1, self.encoder_obstacles.feature_dim).unsqueeze(0)

        # print("feature_targets.shape = ", feature_targets.shape)
        # print("feature_obstacles.shape = ",feature_obstacles.shape)

        new_feature_targets, global_new_feature_targets = self.attention_targets(feature_targets)

        new_feature_targets = new_feature_targets.squeeze()

        # new_feature_obstacles, global_new_feature_obstacles = self.attention_obstacles(feature_obstacles)

        # feature = torch.cat((new_feature_targets, new_feature_obstacles), dim=1)

        # feature = feature.reshape(feature.shape[0], -1)

        # global_feature = torch.cat((global_new_feature_targets, global_new_feature_obstacles), dim=-1)


        # print("feature.shape = ", feature.shape)
        # print("global_feature.shape = ", global_feature.shape)


        actions, entropies, log_probs = self.actor(new_feature_targets, test)
        # print("actions = ", actions)
        actions = actions.reshape(self.num_agents, self.num_targets, -1)
        # print("actions = ", actions)
        # time.sleep(1000000)

        if 'shap' not in self.head_name:
            values = self.critic(global_new_feature_targets)
        else:
            values = self.ShapleyVcritic(new_feature_targets, actions)  # shape [1,1]
        
        return values, actions, entropies, log_probs

    
    def sample_noise(self):
        self.actor.sample_noise()
        if 'shap' in self.head_name:
            self.ShapleyVcritic.sample_noise()
        else:
            self.critic.sample_noise()


    def remove_noise(self):
        self.actor.remove_noise()
        if 'shap' in self.head_name:
            self.ShapleyVcritic.remove_noise()
        else:
            self.critic.remove_noise()




        

