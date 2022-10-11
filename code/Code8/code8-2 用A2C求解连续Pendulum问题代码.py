# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:23:31 2022

@author: Longqiang
"""

## 导入相应的模块
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.distributions import Normal
import matplotlib.pyplot as plt

## 定义A-C网络
class ACNet(nn.Module):
    def __init__(self,state_dim,action_dim,action_limit,device):
        super().__init__()
        self.state_dim = state_dim      # 状态空间维度
        self.action_dim = action_dim    # 动作空间维度
                                        # 动作空间上限
        self.action_limit = torch.as_tensor(
                action_limit,dtype=torch.float32,device=device) 
        self.device = device            # 计算设备
        
        # A-C网络公共部分
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim,64),
            nn.ReLU6(),
            nn.Linear(64,64),
            nn.ReLU6()
            )
        
        # Critic分支
        self.value = nn.Linear(64,1)            # 状态值
        
        # Actor分支
        self.mu = nn.Linear(64,action_dim)      # 均值向量
        self.sigma = nn.Linear(64,action_dim)   # 方差向量
        
    # 前向传播函数
    def forward(self,state):
        common = self.linear_relu_stack(state)  # 公共部分
        value = self.value(common)              # Critic分支计算价值
        mu = torch.tanh(self.mu(common))*self.action_limit  # Actor分支计算mu
        sigma = F.softplus(self.sigma(common))  # Actor分支计算sigma
        
        return value,mu,sigma

    # 根据高斯分布选择动作
    def select_action(self,state):
        _,mu,sigma = self.forward(state)
        pi = Normal(mu,sigma)                   # 高斯分布
        return pi.sample().cpu().numpy()        # 基于高斯分布抽样动作

    # 损失函数
    def loss_func(self,states,actions,v_t,beta):
        values,mu,sigma = self.forward(states)  # 计算预测值
        td = v_t-values                         # 价值TD误差
        value_loss = torch.squeeze(td**2)       # 价值损失函数部分
        pi = Normal(mu,sigma)                   # 高斯分布
        log_prob = pi.log_prob(actions).sum(axis=-1)    # 评价函数值
        entropy = pi.entropy().sum(axis=-1)     # 交叉熵
                                                # 策略损失函数部分
        policy_loss = -(log_prob*torch.squeeze(td.detach())+beta*entropy)
        
        return (value_loss+policy_loss).mean()  # 返回A-C网络损失
    
## 定义分布智能体
class Worker(mp.Process):
    def __init__(self,id,device,env,beta,
                 global_network_lock,global_network,global_optimizer,
                 global_T,global_T_MAX,t_MAX,global_episode,
                 global_return_display,global_return_record,
                 global_return_display_record):
        super().__init__()
        self.id = id            # 工作组的ID
        self.device = device    # 计算设备 cpu or gpu
        self.env = env          # 环境模型
        self.beta = beta        # 策略熵系数
        self.global_network_lock = global_network_lock 
        self.global_network = global_network        # 全局AC网络
        self.global_optimizer = global_optimizer    # 全局优化器
        self.global_T = global_T                    # 全局交互次数计数器
        self.global_T_MAX = global_T_MAX            # 最大全局交互次数
        self.t_MAX = t_MAX                          # 最大局部交互次数
        self.global_episode = global_episode        # 总回合数
        self.global_return_display = global_return_display
        self.global_return_record = global_return_record
        self.global_return_display_record = global_return_display_record

    def update_global(self,states,actions,rewards,
                      next_states,done,optimizer):
        if done:
            R = 0
        else:
            R,mu,sigma = self.global_network.forward(next_states[-1])
        length = rewards.size()[0]

        # 计算目标值
        v_t = torch.zeros([length,1],dtype=torch.float32,
                          device=self.device)
        for i in range(length,0,-1):        # 自后向前计算 v_t
            R = rewards[i-1]+self.env.gamma*R
            v_t[i-1] = R

        #损失函数
        loss = self.global_network.loss_func(states,actions,v_t,self.beta)
        
        # 全局A-C网络参数更新
        with self.global_network_lock.get_lock():   # 锁定线程
            optimizer.zero_grad()                   # 梯度归零
            loss.backward()                         # 误差反向传播
            optimizer.step()                        # 参数更新

    # 训练函数,线程从该函数开始执行，函数名不能改
    def run(self):
        t = 0
        state,done = self.env.reset(),False # 初始化
        episode_return = 0

        # 循环，直到规定的全局交互次数
        while self.global_T.value <= self.global_T_MAX:
            # 获取交互数据
            t_start = t
            buffer_states = []
            buffer_actions = []
            buffer_rewards = []
            buffer_next_states = []
            while not done and t - t_start != self.t_MAX:
                action = self.global_network.select_action(
                        torch.as_tensor(state,dtype=torch.float32,
                                        device=self.device))     # 选择动作
                next_state,reward,done,_ = self.env.step(action) # 交互一次
                episode_return += reward            # 累积奖励
                buffer_states.append(state)         # 状态buffer
                buffer_actions.append(action)       # 动作buffer
                buffer_next_states.append(next_state) # 下一状态buffer
                buffer_rewards.append(reward/10)    # 奖励buffer
                t += 1
                with self.global_T.get_lock():      # 锁定全局计数器线程
                    self.global_T.value += 1        # 更新全局交互次数
                    
                state = next_state                  # 状态更新，继续下一次交互
                
            # 根据buffer的数据来更新全局梯度信息
            self.update_global(
                torch.as_tensor(buffer_states,
                                dtype=torch.float32,device=self.device),
                torch.as_tensor(buffer_actions,
                                dtype=torch.float32,device=self.device),
                torch.as_tensor(buffer_rewards,
                                dtype=torch.float32,device=self.device),
                torch.as_tensor(buffer_next_states,
                                dtype=torch.float32,device=self.device),
                done,self.global_optimizer)

            # 回合结束
            if done:
                with self.global_episode.get_lock():    # 全局网络线程上锁
                    self.global_episode.value += 1      # 更新全局回合计算器
                    self.global_return_record.append(episode_return)
                    if self.global_episode.value == 1:
                        self.global_return_display.value = episode_return
                    else: 
                        self.global_return_display.value *= 0.99
                        self.global_return_display.value += 0.01*episode_return
                        self.global_return_display_record.append(
                                self.global_return_display.value)

                episode_return = 0 # 回报归零
                state,done = self.env.reset(),False # 环境初始化


if __name__ == "__main__":
    # 定义实验参数
    device = 'cpu'      # 'cuda'
    num_processes = 8   # 线程数
    beta = 0.01         # 交叉熵系数
    lr = 1e-4           # 学习率
    T_MAX = 1000000     # 最大全局交互次数
    t_MAX = 5           # 每一个回合最大交互次数

    # 设置进程启动方式为spawn
#    mp.set_start_method('spawn')

    # 定义环境和相关参数
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)                        # 定义环境
    env.gamma = 0.9                                 # 折扣系数
    env.obs_dim = env.observation_space.shape[0]    # 状态空间维度
    env.act_dim = env.action_space.shape[0]         # 动作空间维度
    env.act_limit = env.action_space.high           # 动作空间上限
    
    # 定义中心智能体A-C网络和优化器
    global_network = ACNet(env.obs_dim,
                           env.act_dim,env.act_limit,device).to(device) 
    global_network.share_memory()           
    optimizer = Adam(global_network.parameters(),lr=lr)
    
    # 多线程参数初始化
    global_network_lock = mp.Value('i', 0)
    global_episode = mp.Value('i', 0)           # 全局回合计数器
    global_T = mp.Value('i', 0)                 # 全局交互次数计数器
    global_return_display = mp.Value('d', 0)    # 计算光滑回报的中间量
    global_return_record = mp.Manager().list()  # 记录各回合的回报
    global_return_display_record = mp.Manager().list() # 光滑回报用于作图

    # 定义分布智能体
    workers = [Worker(i,device,env,beta,
                      global_network_lock,global_network,optimizer,
                      global_T,T_MAX,t_MAX,global_episode,
                      global_return_display,global_return_record,
                      global_return_display_record
                      )for i in range(num_processes)]

    [worker.start() for worker in workers]  # 各进程开始工作
    [worker.join() for worker in workers]   # 数据对齐

    # 保存模型
    torch.save(global_network, 'a2c_model.pth')

    # 实验结果可视化 
    plt.figure('train')
    plt.title('train')
    window = 10
    plt.plot(np.array(global_return_record),label='return')
    plt.plot(np.array(global_return_display_record),label='smooth return')
    plt.ylabel('return')
    plt.xlabel('episode')
    plt.legend()
    filepath = 'train.png'
    plt.savefig(filepath,dpi=300)
    plt.show()
    
    
    
    
    