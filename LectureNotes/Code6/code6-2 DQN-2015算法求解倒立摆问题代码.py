# Cartpole by DQN-2015 using pyTorch

#========================================

import gym
import numpy as np
import random
import copy
from collections import deque
import torch
from torch import nn
import matplotlib.pyplot as plt

'''
定义Q-网络类
'''
class NeuralNetwork(nn.Module):             # 继承于Torch的nn.Module类
    ## 类构造函数
    def __init__(self,input_size,output_size):  
        super(NeuralNetwork,self).__init__()
        self.flatten = nn.Flatten()         # 将输入拉直成向量
        # 定义Q-网络
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 20),  # 输入层到第1隐藏层的线性部分
                nn.ReLU(),                  # 第1隐藏层激活函数
                nn.Linear(20, 20),          # 第1隐藏层到第2隐藏层的线性部分
                nn.ReLU(),                  # 第2隐藏层激活函数
                nn.Linear(20, output_size), # 第2隐藏层到输出层
                )
    
    def forward(self, x):                   # 前向传播函数
        x = self.flatten(x)                 # 将输入拉直成向量
        logits = self.linear_relu_stack(x)  # 前向传播，预测x的值
        return logits                       # 返回预测值

'''
定义DQN2015智能体类
'''
class DQN2015():
    def __init__(self,env,epsilon=0.1,learning_rate=1e-3,
                 replay_size=1000,batch_size=32):
        self.replay_buffer = deque()        # 初始化经验回放池
        self.env = env                      # 环境模型
        self.epsilon = epsilon              # epsilon-贪婪策略的参数
        self.learning_rate = learning_rate  # 学习率
        self.replay_size = replay_size      # 经验回放池最大容量
        self.batch_size = batch_size        # 批量尺度
        
        self.create_Q_network()             # 生成Q-网络实体
        self.create_training_method()       # Q-网络优化器

    ## Q-网络生成函数
    def create_Q_network(self):
        # 创建预测Q-网络实体
        self.Q_network = NeuralNetwork(self.env.state_dim,
                                       self.env.aspace_size)
        # 创建目标Q-网络实体,直接复制预测Q-网络
        self.Q_network_t = copy.deepcopy(self.Q_network)
    
    ## Q-网络优化器生成函数
    def create_training_method(self):
                                            # 损失函数
        self.loss_fun = nn.MSELoss(reduction='mean') 
                                            # 随机梯度下降(SGD)优化器
        self.optimizer = torch.optim.SGD(self.Q_network.parameters(),
                                         lr=self.learning_rate) 

    ## epsilon-贪婪策略函数
    def egreedy_action(self,state):
        state = torch.from_numpy(np.expand_dims(state,0))
        state = state.to(torch.float32)
                                            # 计算所有动作值
        Q_value = self.Q_network.forward(state)     
                                            # 以epsilon设定动作概率
        A = np.ones(self.env.aspace_size)*self.epsilon/self.env.aspace_size
                                            # 选取最大动作值对应的动作
        best = np.argmax(Q_value.detach().numpy()) 
        A[best] += 1-self.epsilon           # 以1-epsilon的概率设定贪婪动作
                                            # 选择动作
        action = np.random.choice(range(self.env.aspace_size),p=A) 
        
        return action                       # 返回动作编号

    ## 经验回放技术
    def perceive(self,state,action,reward,next_state,done):        
        # 将动作改写成one-hot向量
        one_hot_action = np.eye(self.env.aspace_size)[action]        
        # 将新数据存入经验回放池
        self.replay_buffer.append((state,one_hot_action,
                                   reward,next_state,done))
        # 如果经验回放池溢出，则删除最早经验数据
        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer.popleft()
        # 经验回放池中数据量多于一个批量就可以开始训练Q-网络
        if len(self.replay_buffer) > self.batch_size:
            self.train_Q_network()

    ## Q-网络训练函数
    def train_Q_network(self):        
        # 从经验回放池中随机抽取一个批量
        minibatch = random.sample(self.replay_buffer,self.batch_size)
        state_batch = np.array([x[0] for x in minibatch])
        action_batch = np.array([x[1] for x in minibatch])        
        
        # 计算TD目标值
        y_batch = []
        for x in minibatch:     # 对minibatch中每一条MDP数据循环
            if x[4]:            # 如果已经到达终止状态
                y_batch.append(x[2])
            else:               # 如果尚未到达终止状态
                temp = torch.from_numpy(x[3]).unsqueeze(0).to(torch.float32)
                value_next = self.Q_network_t(temp)
                td_target = x[2]+self.env.gamma*torch.max(value_next)
                y_batch.append(td_target.item())
        y_batch = np.array(y_batch)
        
        # 将numpy.array数据转换为torch.tensor数据
        state_batch = torch.from_numpy(state_batch).to(torch.float32)
        action_batch = torch.from_numpy(action_batch).to(torch.float32)
        y_batch = torch.from_numpy(y_batch).to(torch.float32)

        self.Q_network.train()          # 声明训练过程
        
        # 预测批量值和损失函数
        pred = torch.sum(torch.multiply(self.Q_network(state_batch),
                                        action_batch),dim=1)
        loss = self.loss_fun(pred, y_batch)

        # 误差反向传播，训练Q-网络
        self.optimizer.zero_grad()      # 梯度归零
        loss.backward()                 # 求各个参数的梯度值
        self.optimizer.step()           # 误差反向传播修改参数    

    ## 训练函数
    def train(self,num_episodes=250,num_steps=1000):
        # 外层循环指导最大轮次
        rewards = []                    # 每一回合的累积奖励
        for episode in range(num_episodes):    
            state = self.env.reset()    # 环境初始化

            # 内层循环直到最大交互次数或到达终止状态
            reward_sum = 0              # 当前轮次的累积奖励
            for step in range(num_steps):
                                        # epsilon-贪婪策略选定动作
                action = self.egreedy_action(state) 
                                        # 交互一个时间步
                next_state,reward,done,_ = self.env.step(action)
                reward_sum += reward    # 累积折扣奖励
                                        # 经验回放技术，训练包括在这里
                self.perceive(state,action,reward,next_state,done) 
                state = next_state      # 更新状态
                if (step+1)%5 == 0:     # 目标Q-网路参数更新
                    self.Q_network_t.load_state_dict(
                           self.Q_network.state_dict())
                
                # 如果到达终止状态则结论本轮循环
                if done:
                    rewards.append(reward_sum)
                    break
                
        # 图示训练过程
        plt.figure('train')
        plt.title('train')
        plt.plot(range(num_episodes),rewards,label='accumulate rewards')
        plt.legend()
        filepath = 'train.png'
        plt.savefig(filepath, dpi=300)
        plt.show()

    ## 测试函数
    def test(self,num_episodes=100):
        # 循环直到最大测试轮数
        rewards = []                        # 每一回合的累积奖励
        for _ in range(num_episodes):
            reward_sum = 0
            state = self.env.reset()        # 环境状态初始化
            
            # 循环直到到达终止状态
            reward_sum = 0                  # 当前轮次的累积奖励
            while True:
                                            # epsilon-贪婪策略选定动作
                action = self.egreedy_action(state)
                                            # 交互一个时间步
                next_state,reward,end,info = self.env.step(action)
                reward_sum += reward        # 累积奖励
                state = next_state          # 状态更新
                
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
    # 加载环境
    env = gym.make('CartPole-v0')   # 导入CartPole环境
    env.gamma = 1                   # 折扣系数
                                    # 状态空间维度
    env.state_dim = env.observation_space.shape[0] 
                                    # 离散动作个数
    env.aspace_size = env.action_space.n 
    
    
    agent = DQN2015(env)            # 创建一个DQN2015智能体
    agent.train()                   #训练智能体
    agent.test()                    # 测试智能体