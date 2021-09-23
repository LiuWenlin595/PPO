from math import log
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from network import ActorCritic
from rollout import RolloutBuffer

# 配置显卡
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_actor,
                 lr_critic,
                 critic_coef,
                 entropy_coef,
                 gamma,
                 use_gae,
                 gae_lambda,
                 k_epochs,
                 eps_clip,
                 has_continuous_action_space,
                 action_std_init=0.6,
                 use_value_clip=False,
                 mini_batch=4000,
                 batch_size=10000):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.mini_batch = mini_batch

        self.buffer = RolloutBuffer(mini_batch, batch_size)

        # 需要用到两个网络, 因为策略比
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, use_orth=False).to(device)
        self.optimizer = torch.optim.Adam([{
            'params': self.policy.actor.parameters(),
            'lr': lr_actor
        }, {
            'params': self.policy.critic.parameters(),
            'lr': lr_critic
        }])
        self.max_grad_norm = 1

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, use_orth=False).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 复制网络

        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.use_value_clip = use_value_clip
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
            state_value = self.policy.critic(state)

        # 因为GAE需要用到这些数据, 而act()只返回action
        # 所以在这里存一波数据到buffer, reward和done在train文件中存
        # self.buffer.states.append(state)
        # self.buffer.actions.append(action)
        # self.buffer.logprobs.append(action_logprob)
        # self.buffer.states_value.append(state_value)
        # print(action, type(action))
        # print(state, )

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten(), action_logprob, state_value
        else:
            return action.item(), action_logprob, state_value

    # 更新模型
    def update(self):
        if self.use_gae:
            returns, actions, logprobs, states = self.calc_gae_return()
        else:
            returns, actions, logprobs, states = self.calc_lambda_return()
        """trick2, reward normalizing"""
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)  # TODO, 改成running reward, 而且好像应该是给reward做norm而不是return

        # list 转 tensor
        old_actions = torch.squeeze(torch.stack(actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(logprobs, dim=0)).detach().to(device)
        old_states = torch.squeeze(torch.stack(states, dim=0)).detach().to(device)

        # 进行k轮update policy
        for _ in range(self.k_epochs):

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 处理state_values的张量维度和reward相同
            state_values = torch.squeeze(state_values)

            # 计算策略比
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算PPO的约束loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            """trick1, value function clipping"""
            if self.use_value_clip:
                # 0.5就相当于epsilon, 是我瞎写的, 需要根据实际任务而定
                _, old_state_values, _ = self.policy_old.evaluate(old_states, old_actions)
                old_state_values = torch.squeeze(old_state_values)
                value_clip = old_state_values + torch.clamp(state_values - old_state_values, -0.5, 0.5)
                critic_loss = torch.min(self.MseLoss(state_values, returns), self.MseLoss(value_clip, returns))
            else:
                critic_loss = self.MseLoss(state_values, returns)

            # 总的loss = actor loss + critic loss + entropy loss
            loss = -torch.min(surr1, surr2) + self.critic_coef * critic_loss - self.entropy_coef * dist_entropy

            # 梯度更新
            self.optimizer.zero_grad()
            loss.mean().backward()
            """trick9, global gradient clipping"""
            # max_grad_norm的值也是我瞎写的
            # nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # 将新的权重赋值给policy old
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清除buffer, 开始下一轮的收集数据训练(改了buffer之后不需要清空了)
        # self.buffer.clear()

    # 计算MiniBatch的GAE, return = V + A(GAE)
    def calc_gae_return(self):
        actions, logprobs, states, next_states, states_value, rewards, is_done = self.buffer.sample()
        returns = []
        for t in range(self.buffer.mini_batch - 1, -1, -1):
            if is_done[t] or t == self.buffer.mini_batch - 1:  # episode最后一帧或者batch最后一帧
                gae = 0
                next_value = self.policy.critic(torch.FloatTensor(next_states[t]).to(device))
                delta = rewards[t] + self.gamma * next_value - states_value[t]
            else:
                delta = rewards[t] + self.gamma * states_value[t + 1] - states_value[t]
            gae += self.gamma * self.gae_lambda * delta
            returns.insert(0, gae + states_value[t])
        return returns, actions, logprobs, states

    # 计算MiniBatch的LambdaReturn, return = reward + gamma * return_next
    def calc_lambda_return(self):
        actions, logprobs, states, next_states, states_value, rewards, is_done = self.buffer.sample()
        returns = []
        for t in range(self.buffer.mini_batch - 1, -1, -1):
            lambda_return = 0
            if is_done[t]:  # 遇到done的时候重新计算
                lambda_return = 0
            lambda_return = rewards[t] + self.gamma * lambda_return
            returns.insert(0, lambda_return)
        return returns, actions, logprobs, states

    # 保存模型
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    # 加载模型
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
