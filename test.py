import time
import gym
from PPO import PPO


def test():
    """和train.py的结构基本一样, 就不写注释了, 有linux的同学可以试一下下面的其他游戏"""

    env_name = "CartPole-v1"
    has_continuous_action_space = False
    max_ep_len = 400
    action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500
    # action_std = 0.1            # 需要设置成和训练结束时的std一样

    # env_name = "RoboschoolWalker2d-v1"
    # has_continuous_action_space = True
    # max_ep_len = 1000
    # action_std = 0.1            # 需要设置成和训练结束时的std一样

    render = True               # 是否render
    frame_delay = 0             # 是否每一帧停顿一些时间, 可以render的更清楚

    total_test_episodes = 10    # 测试轮次

    k_epochs = 80
    eps_clip = 0.2
    gamma = 0.99

    lr_actor = 0.0003
    lr_critic = 0.001

    use_gae = False
    gae_lambda = 0.95

    critic_coef = 0.5
    entropy_coef = 0.01

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]

    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, critic_coef, entropy_coef, gamma, use_gae, gae_lambda,
                    k_epochs, eps_clip, has_continuous_action_space, action_std, use_value_clip=False)

    random_seed = 0             # 这个随机种子的设置和加载模型有关, 需要和下面的 checkpoint_path 对准
    run_num_pretrained = 0      # 也是需要和 checkpoint_path 对准

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    test_running_reward = 0

    for ep in range(total_test_episodes):
        ep_reward = 0
        state = env.reset()

        for t in range(max_ep_len):

            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # 删除堆区数据, 不写也没啥事
        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))

    env.close()

    print("============================================================================================")
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")


if __name__ == '__main__':
    test()
