# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:14:43 2021

@author: Longqiang

基于值迭代的起始探索每次访问蒙特卡罗强化学习算法
解决21点问题
"""

import numpy as np
import blackjack
from collections import defaultdict
import matplotlib.pyplot as plt

'''
基于值迭代的起始探索每次访问蒙特卡罗强化学习算法类
'''
class StartExplore_EveryVisit_ValueIter_MCRL():
    ## 类初始化 
    def __init__(self,env,num_episodes=10000):
        self.env = env
        self.nA = env.action_space.n                        # 动作空间尺度
        self.r_Q = defaultdict(lambda: np.zeros(self.nA))   # 动作值函数
        self.r_sum = defaultdict(lambda: np.zeros(self.nA)) # 累积折扣奖励之和
        self.r_cou = defaultdict(lambda: np.zeros(self.nA)) # 累积折扣奖励次数
        self.policy = defaultdict(int)                      # 各状态下的策略
        self.num_episodes = num_episodes                    # 最大抽样轮次
        
    ## 策略初始化及改进函数，初始化为：点数小于18则继续叫牌，否则停牌
    def update_policy(self,state): 
        if state not in self.policy.keys():
            player, dealer, ace = state
            action = 0 if player >= 18 else 1   # 0：停牌，1：要牌
        else:
            action = np.argmax(self.r_Q[state]) # 最优动作值对应的动作
        
        self.policy[state] = action
        
    ## 蒙特卡罗抽样产生一条经历完整的MDP序列
    def mc_sample(self):
        onesequence = []                    # 经验轨迹容器
                
        # 基于贪婪策略产生一条轨迹
        state = self.env.reset()            # 起始探索产生初始状态
        while True:
            self.update_policy(state)       # 策略改进
            action = self.policy[state]     # 根据策略选择动作 
            next_state,reward,done,_ = env.step(action) # 交互一步
            onesequence.append((state,action,reward))   # 经验轨迹          
            state = next_state
            if done:                        # 游戏是否结束
                break
        
        return onesequence
         
    ## 蒙特卡罗每次访问策略评估一条序列
    def everyvisit_valueiter_mc(self,onesequence):
        # 访问经验轨迹中的每一个状态-动作对
        for k, data_k in enumerate(onesequence):
            state = data_k[0]               # 状态
            action = data_k[1]              # 动作
            # 计算累积折扣奖励
            G = sum([x[2]*np.power(env.gamma,i) for i, x 
                           in enumerate(onesequence[k:])])
            self.r_sum[state][action] += G        # 累积折扣奖励之和
            self.r_cou[state][action] += 1.0      # 累积折扣奖励次数
            self.r_Q[state][action] = self.r_sum[
                     state][action]/self.r_cou[state][action]

    ## 蒙特卡罗强化学习
    def mcrl(self):
        for i in range(self.num_episodes):
            # 起始探索抽样一条MDP序列
            onesequence = self.mc_sample()
            # 值迭代过程，结合了策略评估和策略改进
            self.everyvisit_valueiter_mc(onesequence)
            
        opt_policy = self.policy            # 最优策略
        opt_Q = self.r_Q                    # 最优动作值
        
        return opt_policy, opt_Q
    
    ## 绘制最优策略图像
    def draw(self,policy):
        true_hit = [(x[1],x[0]) for x in policy.keys(
                ) if x[2]==True and policy[x]==1]
        true_stick = [(x[1],x[0]) for x in policy.keys(
                ) if x[2]==True and policy[x]==0]
        false_hit = [(x[1],x[0]) for x in policy.keys(
                ) if x[2]==False and policy[x]==1]
        false_stick = [(x[1],x[0]) for x in policy.keys(
                ) if x[2]==False and policy[x]==0]
        
        plt.figure(1)
        plt.plot([x[0] for x in true_hit],
                 [x[1] for x in true_hit],'bo',label='HIT')
        plt.plot([x[0] for x in true_stick],
                 [x[1] for x in true_stick],'rx',label='STICK')
        plt.xlabel('dealer'), plt.ylabel('player')
        plt.legend(loc='upper right')
        plt.title('Usable Ace')
        filepath = 'code3-5 UsabelAce.png'
        plt.savefig(filepath, dpi=300)
        
        plt.figure(2)
        plt.plot([x[0] for x in false_hit],
                 [x[1] for x in false_hit],'bo',label='HIT')
        plt.plot([x[0] for x in false_stick],
                 [x[1] for x in false_stick],'rx',label='STICK')
        plt.xlabel('dealer'), plt.ylabel('player')
        plt.legend(loc='upper right')
        plt.title('No Usable Ace')
        filepath = 'code3-5 NoUsabelAce.png'
        plt.savefig(filepath, dpi=300)
        
'''
主程序
'''
if __name__ == '__main__':
    env = blackjack.BlackjackEnv()  # 导入环境模型
    env.gamma = 1                   # 补充定义折扣系数

    agent = StartExplore_EveryVisit_ValueIter_MCRL(
            env,num_episodes=1000000)
    opt_policy,opt_Q = agent.mcrl()
    for key in opt_policy.keys():
        print(key,": ",opt_policy[key],opt_Q[key])
    agent.draw(opt_policy)