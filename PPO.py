import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

# 配置显卡
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.state_value = []
        self.rewards = []
        self.is_terminals = []
        self.terminal_state_value = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.state_value[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.terminal_state_value[:]


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, use_orth):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        """trick3, 正交初始化"""
        """trick8, tanh做激活函数"""
        if use_orth:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), np.sqrt(2))
        else:
            init_ = lambda m: m

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                            init_(nn.Linear(state_dim, 64)),
                            nn.Tanh(),
                            init_(nn.Linear(64, 64)),
                            nn.Tanh(),
                            init_(nn.Linear(64, action_dim)),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            init_(nn.Linear(state_dim, 64)),
                            nn.Tanh(),
                            init_(nn.Linear(64, 64)),
                            nn.Tanh(),
                            init_(nn.Linear(64, action_dim)),
                            nn.Softmax(dim=-1)
                        )

        # critic
        self.critic = nn.Sequential(
                        init_(nn.Linear(state_dim, 64)),
                        nn.Tanh(),
                        init_(nn.Linear(64, 64)),
                        nn.Tanh(),
                        init_(nn.Linear(64, 1))
                    )
        
    def set_action_std(self, new_action_std):
        """设置方差"""
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """actor, 输入状态, 输出动作"""
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        """actor + critic, 输入状态, 输出动作+动作值函数+熵"""
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # 输出单动作的环境需要特殊处理一下数据
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, critic_coef, entropy_coef, gamma, use_gae, gae_lambda,
                 k_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, use_value_clip=False):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        
        self.buffer = RolloutBuffer()

        # 需要用到两个网络, 因为策略比
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init,
                                  use_orth=False).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        self.max_grad_norm = 1

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init,
                                      use_orth=False).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

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
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_value.append(state_value)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        """GAE计算 rewards = V + A(GAE)"""
        if self.use_gae:
            gae = 0
            for step in reversed(range(len(self.buffer.rewards))):
                # 计算某一个episode最后一步的gae
                if self.buffer.is_terminals[step]:
                    gae = 0
                    # 因为buffer存的是s_i, a_i, V_i, r_i, done_i, 最后一步得到的V(s_i+1)会记录进buffer.terminal_state_value
                    next_value = self.buffer.terminal_state_value.pop()
                    delta = self.buffer.rewards[step] + self.gamma * next_value - self.buffer.state_value[step]
                elif step == len(self.buffer.rewards) - 1:
                    # 很无奈, 最后一步必须拿出来单独处理, 因为没办法计算它的V', 只能用V近似代替
                    # 后续可以重新写一下buffer, 整成s_i+1, a_i, V_i, r_i, done_i的形式
                    delta = self.buffer.rewards[step] + self.gamma * self.buffer.state_value[step]\
                            - self.buffer.state_value[step]
                else:
                    delta = self.buffer.rewards[step] + self.gamma * self.buffer.state_value[step+1] \
                            - self.buffer.state_value[step]

                gae = gae + self.gamma * self.gae_lambda * delta
                rewards.insert(0, gae + self.buffer.state_value[step])
        else:
            # 蒙特卡洛采样计算 rewards = G
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
                # buffer里存的是几个episode的数据, 所以每次遇到terminal都要重新计算reward
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

        """trick2, reward normalizing"""
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # list 转 tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # 进行k轮update policy
        for _ in range(self.k_epochs):

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 处理state_values的张量维度和reward相同
            state_values = torch.squeeze(state_values)
            
            # 计算策略比
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算PPO的约束loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            """trick1, value function clipping"""
            if self.use_value_clip:
                # 0.5就相当于epsilon, 是我瞎写的, 需要根据实际任务而定
                _, old_state_values, _ = self.policy_old.evaluate(old_states, old_actions)
                old_state_values = torch.squeeze(old_state_values)
                value_clip = old_state_values + torch.clamp(state_values - old_state_values, -0.5, 0.5)
                critic_loss = torch.min(self.MseLoss(state_values, rewards), self.MseLoss(value_clip, rewards))
            else:
                critic_loss = self.MseLoss(state_values, rewards)

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

        # 清除buffer, 开始下一轮的收集数据训练
        self.buffer.clear()

    # 保存模型
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    # 加载模型
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))