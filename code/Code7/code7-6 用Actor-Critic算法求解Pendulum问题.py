# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 22:57:01 2022

@author: LONG QIANG
"""

#==============================================================================

import gym
import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt

'''
定义Actor网络，即策略网络
'''
class Actor(nn.Module):
    def __init__(self,input_size,output_size):
        super(Actor,self).__init__()      
        
        # 定义策略网络各层
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size,32),
                nn.ReLU(),
                nn.Linear(32,32),
                nn.ReLU(),
                nn.Linear(32,output_size)
                )
        # 优化器
        self.opt=torch.optim.Adam(self.parameters(),lr=1e-3)
        # 损失函数
        self.loss = nn.MSELoss(reduction='mean')
    
    ## 前向传播函数
    def forward(self,x):
        action = self.linear_relu_stack(x)
        return action   

    ## 训练函数
    def train(self,state,action,td_error):          
        # 损失函数
        pred = self.forward(state)
        loss = self.loss(pred,action)*td_error 
        
        # 训练策略网络
        self.opt.zero_grad()      # 梯度归零
        loss.backward()           # 求各个参数的梯度
        self.opt.step()           # 误差反向传播
         
'''
定义Critic网络，即价值网络
'''
class Critic(nn.Module):
    def __init__(self,input_size,output_size):
        super(Critic,self).__init__()
        
        # 定义价值网络各层
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size,32),
                nn.ReLU(),
                nn.Linear(32,32),
                nn.ReLU(),
                nn.Linear(32,output_size),
                )
        
        # 优化器
        self.opt=torch.optim.Adam(self.parameters(),lr=1e-3)
        # 损失函数
        self.loss = nn.MSELoss(reduction='mean')
        
    ## 前向传播函数
    def forward(self, x): 
        qval = self.linear_relu_stack(x)
        return qval
    
    ## 训练函数
    def train(self,gamma,state,reward,done,next_state):
        # 损失函数
        y_pred = self.forward(state)    # 计算预测值
        if done:                        # 计算目标值
            y_target = reward
        else:
            y_hat_next = self.forward(next_state)
            y_target = reward+gamma*y_hat_next
        td_error = y_target-y_pred
        loss = self.loss(y_pred,y_target) 
        
        # 训练价值网络
        self.opt.zero_grad()            # 梯度归零
        loss.backward()                 # 求各个参数的梯度值
        self.opt.step()                 # 误差反向传播更新参数
        
        return td_error                 # 返回TD误差

'''
定义AC策略梯度法类
'''
class AC():
    def __init__(self,env):
        self.env = env                  # 环境模型
        
        # 创建策略和价值网路实体
        self.actor = Actor(self.env.state_dim,1)
        self.critic = Critic(self.env.state_dim,1) 
    
    ## 训练函数
    def train(self,num_episodes=100):
        # 外层循环直到最大迭代轮次
        rewards = []
        for ep in range(num_episodes):
            state = self.env.reset()        # state.shape = (3,)
            state = np.array([state])       # state.shape = (1,3)
            reward_sum = 0
            # 内层循环，一次经历完整的模拟
            while True:
                                            # action.shape = (1,1), tensor
                action = self.actor.forward(torch.Tensor(state)) 
                next_state,reward,done,_ = self.env.step(
                        action.detach().numpy())
                                            # next_state.shape = (1,3)
                next_state = next_state.transpose() 
                reward_sum += reward
                
                # 训练价值网络
                state = torch.Tensor(state)
                reward = torch.Tensor(np.array([reward]))
                next_state = torch.Tensor(next_state)
                td_error = self.critic.train(
                        self.env.gamma,state,reward,done,next_state)
                
                # 训练策略网络
                td_error_nograd = torch.Tensor.detach(td_error)
                self.actor.train(state,action,td_error_nograd)
                if done:
                    rewards.append(reward_sum)
                    break
                else:
                    state = next_state
    
        # 图示训练过程
        plt.figure('train')
        plt.title('train')
        window = 10
        smooth_r = [np.mean(rewards[i-window:i+1]) if i > window 
                        else np.mean(rewards[:i+1]) 
                        for i in range(len(rewards))]
        plt.plot(range(num_episodes),rewards,label='accumulate rewards')
        plt.plot(smooth_r,label='smoothed accumulate rewards')
        plt.legend()
        filepath = 'train.png'
        plt.savefig(filepath, dpi=300)
        plt.show()    
    
    ## 测试函数
    def test(self,num_episodes=100):
        # 循环直到最大测试轮数
        rewards = []                        # 每一轮次的累积奖励
        for _ in range(num_episodes):
            reward_sum = 0
            state = self.env.reset()        # 环境状态初始化
            
            # 循环直到到达终止状态
            reward_sum = 0                  # 当前轮次的累积奖励
            while True:
                                            # epsilon-贪婪策略选定动作
                action = self.actor.forward(torch.Tensor([state]))
                                            # 交互一个时间步
                next_state,reward,end,info = self.env.step(
                        action.detach().numpy())
                
                reward_sum += reward        # 累积奖励
                state = next_state.transpose().squeeze()
                
                # 检查是否到达终止状态
                if end:                     
                    rewards.append(reward_sum)
                    break
        
        score = np.mean(np.array(rewards))  # 计算测试得分
        
        # 图示测试结果
        plt.figure('test')
        plt.title('test: score='+str(score))
        plt.plot(range(num_episodes),rewards,label='accumulate rewards')
        plt.legend()
        filepath = 'test.png'
        plt.savefig(filepath, dpi=300)
        plt.show()
        
        return score                        # 返回测试得分

'''
主程序
'''
if __name__ == '__main__':
    # 导入CartPole环境
    env = gym.make('Pendulum-v0')  # MountainCarCountinus-v0
    env.gamma = 0.99                                # 补充定义折扣系数
    env.state_dim = env.observation_space.shape[0]  # 状态维度
    
    agent = AC(env)         # 创建一个AC类智能体
    agent.train()           # 训练
    agent.test()            # 测试