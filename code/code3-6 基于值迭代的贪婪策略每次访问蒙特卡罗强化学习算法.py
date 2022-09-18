# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:14:43 2021

@author: Longqiang

基于值迭代的epsilon-贪婪策略每次访问蒙特卡罗强化学习算法
解决21点问题
"""

import numpy as np
import blackjack
from collections import defaultdict
import matplotlib.pyplot as plt

'''
基于值迭代的epsilon-贪婪策略每次访问蒙特卡罗强化学习算法类
'''
class SoftExplore_EveryVisit_ValueIter_MCRL():
    ## 类初始化 
    def __init__(self,env,num_episodes=10000,epsilon=0.1):
        self.env = env
        self.nA = env.action_space.n                        # 动作空间维度
        self.Q_bar = defaultdict(lambda: np.zeros(self.nA)) # 动作值函数
        self.G_sum = defaultdict(lambda: np.zeros(self.nA)) # 累积折扣奖励之和
        self.G_cou = defaultdict(lambda: np.zeros(self.nA)) # 累积折扣奖励次数
        self.g_policy = defaultdict(int)                    # 贪婪策略
                                                            # epsilon贪婪策略
        self.eg_policy = defaultdict(lambda: np.zeros(self.nA))  
        self.num_episodes = num_episodes                    # 最大抽样轮次
        self.epsilon = epsilon   
        
    ## 策略初始化及改进函数，初始化为：点数小于18则继续叫牌，否则停牌
    def update_policy(self,state): 
        if state not in self.g_policy.keys():
            player, dealer, ace = state
            action = 0 if player >= 18 else 1       # 0：停牌，1：要牌
        else:
            action = np.argmax(self.Q_bar[state])   # 最优动作值对应的动作

        # 贪婪策略
        self.g_policy[state] = action 
        # 对应的epsilon贪婪策略
        self.eg_policy[state] = np.ones(self.nA)*self.epsilon/self.nA
        self.eg_policy[state][action] += 1-self.epsilon
        
        return self.g_policy[state], self.eg_policy[state]
        
    ## 蒙特卡罗抽样产生一条经历完整的MDP序列
    def mc_sample(self):
        onesequence = []                            # 经验轨迹容器
                
        # 基于epsilon-贪婪策略产生一条轨迹
        state = self.env.reset()                    # 初始状态
        while True:
            _,action_prob = self.update_policy(state)
            action = np.random.choice(np.arange(len(action_prob)),
                                      p=action_prob)
            next_state,reward,done,info = env.step(action)  # 交互一步
            onesequence.append((state,action,reward,info))  # 经验轨迹          
            state = next_state
            if done:                                # 游戏是否结束
                break
        
        return onesequence
         
    ## 蒙特卡罗每次访问策略评估一条序列
    def everyvisit_valueiter_mc(self,onesequence):
        # 访问经验轨迹中的每一个状态-动作对
        for k, data_k in enumerate(onesequence):
            state = data_k[0]
            action = data_k[1]
            # 计算累积折扣奖励
            G = sum([x[2]*np.power(env.gamma,i) for i, x 
                           in enumerate(onesequence[k:])])
            self.G_sum[state][action] += G          # 累积折扣奖励之和
            self.G_cou[state][action] += 1.0        # 累积折扣奖励次数
            self.Q_bar[state][action] = self.G_sum[
                       state][action]/self.G_cou[state][action]

    ## 蒙特卡罗强化学习
    def mcrl(self):
        for i in range(self.num_episodes):
            # 起始探索抽样一条MDP序列
            onesequence = self.mc_sample()

            # 值迭代过程，结合了策略评估和策略改进
            self.everyvisit_valueiter_mc(onesequence)
            
        opt_policy = self.g_policy              # 最优策略
        opt_Q = self.Q_bar                      # 最优动作值
        
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
        filepath = 'code3-6 UsabelAce.png'
        plt.savefig(filepath, dpi=300)
        
        plt.figure(2)
        plt.plot([x[0] for x in false_hit],
                 [x[1] for x in false_hit],'bo',label='HIT')
        plt.plot([x[0] for x in false_stick],
                 [x[1] for x in false_stick],'rx',label='STICK')
        plt.xlabel('dealer'), plt.ylabel('player')
        plt.legend(loc='upper right')
        plt.title('No Usable Ace')
        filepath = 'code3-6 NoUsabelAce.png'
        plt.savefig(filepath, dpi=300)

'''
主程序
'''
if __name__ == '__main__':
    env = blackjack.BlackjackEnv()  # 导入环境模型
    env.gamma = 1                   # 补充定义折扣系数

    agent = SoftExplore_EveryVisit_ValueIter_MCRL(
            env,num_episodes=1000000,epsilon=0.1)
    opt_policy,opt_Q = agent.mcrl()
    for key in opt_policy.keys():
        print(key,": ",opt_policy[key],opt_Q[key])
    agent.draw(opt_policy)