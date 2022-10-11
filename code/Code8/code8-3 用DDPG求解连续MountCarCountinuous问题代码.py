# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:24:00 2022

@author: Longqiang
"""

'''
导入包
'''
import numpy as np
import random
import gym
import torch
import copy
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt

'''
定义超参数
'''
NUM_EPISODES = 300
BUFFER_SIZE = 100
BATCH_SIZE = 32
TAU = 0.01

'''
经验回放池
'''
class ReplayBuffer():
    def __init__(self,env):
        self.env = env
        self.replay_buffer = deque()

    ## 往经验回放池中添加数据
    def add(self,state,action,reward,next_state,done):
        self.replay_buffer.append((state,action,reward,next_state,done))
        
        if len(self.replay_buffer) > BUFFER_SIZE: # 溢出，则删除最早数据
            self.replay_buffer.popleft()
        
    ## 从经验回放池中采样数据
    def sample(self):
        batch = random.sample(self.replay_buffer,BATCH_SIZE)
        return batch
    
    ## 采样指示器
    def is_available(self):
        if len(self.replay_buffer) >= BATCH_SIZE:
            return True
        else:
            return False

'''
定义Actor网络，即策略网络
'''
class Actor(nn.Module):
    def __init__(self,state_dim,action_dim,action_max):
        super(Actor,self).__init__()      
        self.action_max = torch.Tensor(action_max)
        
        # 定义策略网络各层
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(state_dim,32),
                nn.ReLU(),
                nn.Linear(32,32),
                nn.ReLU(),
                nn.Linear(32,action_dim)
                )
    
    ## 前向传播函数
    def forward(self,state):
        temp = self.linear_relu_stack(state)
        action = self.action_max*torch.tanh(temp)
        return action   
         
'''
定义Critic网络，即价值网络
'''
class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()
        
        # 定义价值网络各层
        self.sl_1 = nn.Linear(state_dim,32)
        self.al_1 = nn.Linear(action_dim,32)
        self.sal_2 = nn.Linear(32,32)
        self.sal_3 = nn.Linear(32,1)
        self.relu = nn.ReLU()
        
    ## 前向传播函数
    def forward(self,state,action):
        l1 = self.relu(self.sl_1(state))+self.relu(self.al_1(action))
        l2 = self.relu(self.sal_2(l1))
        qval = self.sal_3(l2)
        
        return qval
        
'''
定义DDPG类
'''
class DDPG():
    def __init__(self,env):
        self.env = env                  # 环境模型
        self.buffer = ReplayBuffer(env) # 创建经验回放池
        
        # 创建价值网络
        self.critic = Critic(self.env.state_dim,1)
        self.critic_t = copy.deepcopy(self.critic)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=1e-3)
        self.critic_loss = nn.MSELoss(reduction='mean')

        # 创建策略网络
        self.actor = Actor(self.env.state_dim,env.action_dim,env.action_max)
        self.actor_t = copy.deepcopy(self.actor)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=1e-3)
    
    ## 训练函数
    def train(self):
        # 外层循环直到最大迭代轮次
        rewards = []
        for ep in range(NUM_EPISODES):
            state = self.env.reset()
            reward_sum = 0
            # 内层循环，一次经历完整的模拟
            while True:
                action = self.actor.forward(torch.Tensor(state))
                action = action.detach().numpy()
                next_state,reward,done,_ = self.env.step(action)
                self.buffer.add(state,action,reward,next_state,done)
                reward_sum += reward
                
                # 判断训练数据量是否大于BATCH_SIZE
                if self.buffer.is_available():
                    # 抽样并转化数据
                    batch = self.buffer.sample() 
                    state_arr = np.array([x[0] for x in batch])
                    action_arr = np.array([x[1] for x in batch])
                    reward_arr = np.array([x[2] for x in batch])
                    next_state_arr = np.array([x[3] for x in batch])
                    done_arr = np.array([x[4] for x in batch])
                    state_ten = torch.Tensor(state_arr)
                    action_ten = torch.Tensor(action_arr)
                    reward_ten = torch.Tensor(reward_arr)
                    next_state_ten = torch.Tensor(next_state_arr)
                    done_ten = torch.Tensor(done_arr)
                    
                    # 训练价值网络
                    q_pred = self.critic(state_ten,action_ten)
                    q_targ = torch.zeros(q_pred.shape)
                    q_pred_next = self.critic_t(
                            next_state_ten,self.actor_t(next_state_ten))
                    q_pred_next = q_pred_next.detach()
                    for i in range(len(state)):
                        if done_ten[i]:
                            q_targ[i] = reward_ten[i]
                        else:
                            q_targ[i] = reward_ten[i]+\
                                        self.env.gamma*q_pred_next[i]
                    critic_loss = self.critic_loss(q_pred,q_targ)
                    
                    self.critic_opt.zero_grad()     # 梯度归零
                    critic_loss.backward()          # 求各个参数的梯度值
                    self.critic_opt.step()          # 误差反向传播更新参数
                    
                    # 训练策略网络
                    actor_loss = -self.critic(
                            state_ten,self.actor(state_ten)).mean()
                    
                    self.actor_opt.zero_grad()      # 梯度归零
                    actor_loss.backward()           # 求各个参数的梯度值
                    self.critic_opt.step()          # 误差反向传播更新参数
                                    
                    # 目标网络参数软更新
                    for param,t_param in zip(self.critic.parameters(),
                                             self.critic_t.parameters()):
                        t_param.data.copy_(
                                TAU*param.data+(1-TAU)*t_param.data)
                    for param,t_param in zip(self.actor.parameters(),
                                             self.actor_t.parameters()):
                        t_param.data.copy_(
                                TAU*param.data+(1-TAU)*t_param.data)                
                
                if done: # 回合结束
                    rewards.append(reward_sum)
                    break
                else: # 继续下一次交互
                    state = next_state
    
        # 图示训练过程
        plt.figure('train')
        plt.title('train')
        window = 10
        smooth_r = [np.mean(rewards[i-window:i+1]) if i > window 
                        else np.mean(rewards[:i+1]) 
                        for i in range(len(rewards))]
        plt.plot(range(NUM_EPISODES),rewards,label='accumulate rewards')
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
                action = self.actor.forward(torch.Tensor(state))
                action = action.detach().numpy()
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

'''
主程序
'''
if __name__ == '__main__':
    # 导入环境
    env = gym.make('MountainCarContinuous-v0')  # MountainCarContinuous-v0 Pendulum-v0
    env.gamma = 0.99                                # 补充定义折扣系数
    env.state_dim = env.observation_space.shape[0]  # 状态空间维度
    env.action_dim = env.action_space.shape[0]      # 动作空间维度
    env.action_max = env.action_space.high          # 动作空间上限   
    
    agent = DDPG(env)       # 创建一个DDPG类智能体
    agent.train()           # 训练
    agent.test()            # 测试