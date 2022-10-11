'''
导入包
'''
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

'''
定义Actor网络，即策略网络
'''
class Actor(nn.Module): 
    def __init__(self,n_states,bound):
        super(Actor, self).__init__()
        self.n_states = n_states
        self.bound = float(bound)
        self.policy_noise = 1e-6           # 限制最小标准差的噪声

        # 定义策略网络各层
        self.layer = nn.Sequential(
            nn.Linear(self.n_states,128),
            nn.ReLU()
            )

        self.mu_out = nn.Linear(128,1)
        self.sigma_out = nn.Linear(128,1)

    ## 前向传播函数
    def forward(self, x):
        x = F.relu(self.layer(x))
        mu = self.bound*torch.tanh(self.mu_out(x))
        sigma = F.softplus(self.sigma_out(x))
        sigma = torch.clamp(sigma, min=self.policy_noise,max=1e10)
        return mu, sigma

'''
定义Critic网络，即价值网络
'''   
class Critic(nn.Module):  
    def __init__(self,n_states):
        super(Critic, self).__init__()
        self.n_states = n_states
        
        # 定义价值网络各层
        self.layer = nn.Sequential(
            nn.Linear(self.n_states,128),
            nn.ReLU(),
            nn.Linear(128,1)
            )

    ## 前向传播函数
    def forward(self,x):
        v = self.layer(x)
        return v

'''
定义PPO类
'''
class PPO(nn.Module):
    def __init__(self,env,n_states,n_actions,bound, 
                 lr_actor=1e-4,lr_critic=1e-4,batch_size=32,epsilon=0.2, 
                 gamma=0.9,a_update_steps=10,c_update_steps=10):
        super().__init__()
        self.env = env                      # 环境模型
        self.n_states = n_states            # 状态维数
        self.n_actions = n_actions          # 动作维数
        self.bound = bound                  # 动作幅值
        self.lr_actor  = lr_actor           # Actor网络学习率
        self.lr_critic = lr_critic          # Critic网络学习率
        self.batch_size= batch_size         # 批大小
        self.epsilon = epsilon              # 剪切参数
        self.gamma   = gamma                # 折扣率参数    
        self.a_update_steps = a_update_steps# 批次数据的Actor网络训练次数
        self.c_update_steps = c_update_steps# 批次数据的Critic网络训练次数
        self.env.seed(10)                   # 设置随机数种子
        torch.manual_seed(10)               # 设置随机数种子

        # 创建策略网络
        self.actor = Actor(n_states, bound)
        self.actor_old = Actor(n_states, bound)
        self.actor_optim = torch.optim.Adam(
                self.actor.parameters(),lr=self.lr_actor)

        # 创建价值网络
        self.critic_model = Critic(n_states)
        self.critic_optim = torch.optim.Adam(
                self.critic_model.parameters(),lr=self.lr_critic)

    ## 输出随机策略动作
    def choose_action(self, s):
        s = torch.FloatTensor(s)
        mu, sigma = self.actor(s)                       # 返回均值、标准差
        dist = torch.distributions.Normal(mu,sigma)    # 得到正态分布 
        action = dist.sample()                          # 采样输出动作
        return np.clip(action,-self.bound,self.bound) # 限制动作区间

    ## 计算状态价值估计
    def discount_reward(self,rewards, s_):
        s_ = torch.FloatTensor(s_)
        target = self.critic_model(s_).detach()             
        target_list = []
        # 从后向前回溯计算价值估计
        for r in rewards[::-1]:
            target = r+self.gamma*target
            target_list.append(target)
        target_list.reverse()
        target_list = torch.cat(target_list)       

        return target_list

    ## 单步Actor网络训练
    def actor_learn(self,states,actions,advantage):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).reshape(-1,1)

        mu, sigma = self.actor(states)
        pi = torch.distributions.Normal(mu,sigma)

        old_mu, old_sigma = self.actor_old(states)
        old_pi = torch.distributions.Normal(old_mu,old_sigma)
        
        # 计算新旧策略概率比
        ratio = torch.exp(pi.log_prob(actions)-old_pi.log_prob(actions))
        surr = ratio * advantage.reshape(-1, 1)     # 代理指标
        # 剪切函数处理，得到损失函数
        loss = -torch.mean(torch.min(
                surr,torch.clamp(ratio,1-self.epsilon,1+self.epsilon
                                 )*advantage.reshape(-1,1)))

        self.actor_optim.zero_grad()    # 梯度归零
        loss.backward()                 # 求各个参数的梯度值
        self.actor_optim.step()         # 误差反向传播更新参数

    ## 单步Critic网络训练
    def critic_learn(self,states,targets):
        states = torch.FloatTensor(states)
        # 计算预测价值估计
        v = self.critic_model(states).reshape(1,-1).squeeze(0)

        loss_func = nn.MSELoss()
        loss = loss_func(v,targets)     # 损失函数

        self.critic_optim.zero_grad()   # 梯度归零
        loss.backward()                 # 求各个参数的梯度值
        self.critic_optim.step()        # 误差反向传播更新参数

    ## 计算优势函数
    def cal_adv(self,states,targets):
        states = torch.FloatTensor(states)
        v = self.critic_model(states)                           
        advantage = targets-v.reshape(1,-1).squeeze(0)
        return advantage.detach()                               

    ## 智能体训练
    def update(self,states,actions,targets):
        # 更新旧Actor模型
        self.actor_old.load_state_dict(self.actor.state_dict())   
        advantage = self.cal_adv(states,targets)
        # 训练Actor网络多次
        for i in range(self.a_update_steps):                      
            self.actor_learn(states,actions,advantage)
        # 训练Critic网络多次
        for i in range(self.c_update_steps):                      
            self.critic_learn(states,targets)

    ## 训练函数
    def train(self,NUM_EPISODES=600,len_episode=200):
        # 外层循环直到最大迭代轮次
        rewards_history = []
        for episode in range(NUM_EPISODES):
            reward_sum = 0
            s = env.reset()
            states, actions, rewards = [], [], []
            # 内层循环，一次经历完整的模拟
            for t in range(len_episode):
                a = self.choose_action(s)
                s_, r, done, _ = env.step(a)
                reward_sum += r
                states.append(s)
                actions.append(a)
                rewards.append((r+8)/8)       # 对奖励函数进行调整
                s = s_

                # 训练数据量满足要求，进行训练
                if (t+1)%self.batch_size==0 or t==len_episode-1:  
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)

                    targets = self.discount_reward(rewards,s_)  # 奖励回溯
                    self.update(states,actions,targets)         # 网络更新
                    states, actions, rewards = [], [], []
    
            print('Episode {:03d} | Reward:{:.03f}'.format(
                    episode, reward_sum))
            rewards_history.append(reward_sum)

        # 图示训练过程
        plt.figure('train')
        plt.title('train')
        window = 10
        smooth_r = [np.mean(rewards_history[i-window:i+1]) if i > window 
                        else np.mean(rewards_history[:i+1]) 
                        for i in range(len(rewards_history))]
        plt.plot(range(NUM_EPISODES
                       ),rewards_history,label='accumulate rewards')
        plt.plot(smooth_r,label='smoothed accumulate rewards')
        plt.legend()
        filepath = 'train.png'
        plt.savefig(filepath, dpi=300)
        plt.show()     
    
    ## 测试函数
    def test(self,test_episodes=100):
        # 循环直到最大测试轮数
        rewards = []                        # 每一轮次的累积奖励
        for _ in range(test_episodes):
            reward_sum = 0
            state = self.env.reset()        # 环境状态初始化
            # 循环直到到达终止状态
            reward_sum = 0                  # 当前轮次的累积奖励
            while True:
                action = self.choose_action(state)
                next_state,reward,done,info = self.env.step(action)
                reward_sum += reward
                state = next_state 
                # 检查是否到达终止状态
                if done:                     
                    rewards.append(reward_sum)
                    break
        
        score = np.mean(np.array(rewards))  # 计算测试得分
        
        # 图示测试结果
        plt.figure('test')
        plt.title('test: score='+str(score))
        plt.plot(range(test_episodes),rewards,label='accumulate rewards')
        plt.legend()
        filepath = 'test.png'
        plt.savefig(filepath, dpi=300)
        plt.show()
        
        return score                        # 返回测试得分

    ## 一局仿真 
    def sim(self,EP_LEN=201,do_render=False):
        s = self.env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        for t in range(EP_LEN):             # 一个回合仿真
            if do_render==True:
                self.env.render()           # 动画，显示一帧
            else:
                pass
    
            a = self.choose_action(s)            
            s_, r, done, info = self.env.step(a)
    
            if done:
                break
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)
            s = s_
        buffer_s = np.array(buffer_s)
        buffer_a = np.array(buffer_a)
        buffer_r = np.array(buffer_r)
        batch = buffer_s,buffer_a,buffer_r
        self.env.close()
        return batch

'''
主程序
'''
if __name__ == '__main__':   
    # 导入环境
    env = gym.make('Pendulum-v0')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    bound = env.action_space.high[0]

    # 创建一个PPO类智能体
    agent = PPO(env, n_states, n_actions, bound) 
    agent.train()           # 训练
    agent.test()            # 测试
    batch = agent.sim(EP_LEN=200,do_render=True)      # 仿真