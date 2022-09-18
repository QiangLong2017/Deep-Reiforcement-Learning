# -*- coding: utf-8 -*-
"""
Created on Sun May 23 07:50:12 2021

@author: LONG QIANG

首次访问蒙特卡罗策略评估，计算动作值函数
简单策略为：如果player手中牌的点数大于等于18，则停止要牌，否则继续要牌。
"""
import numpy as np
import blackjack
from collections import defaultdict

'''
待评估的玩家策略：点数小于18则继续叫牌，否则停牌
'''
def simple_policy(state): 
    player, dealer, ace = state
    return 0 if player >= 18 else 1 # 0：停牌，1：要牌

'''
首次访问蒙特卡罗策略评估：算法3-1的具体实现
'''
def firstvisit_mc_actionvalue(env,num_episodes=50000):
    r_sum = defaultdict(float)      # 记录状态-动作对的累积折扣奖励之和
    r_count = defaultdict(float)    # 记录状态-动作对的累积折扣奖励次数
    r_Q = defaultdict(float)        # 动作值样本均值
    
    # 采样num_episodes条经验轨迹
    MDPsequence = []                # 经验轨迹容器
    for i in range(num_episodes):
        state = env.reset()         # 环境状态初始化
        
        # 采集一条经验轨迹
        onesequence = []
        while True:
            action = simple_policy(state) # 根基给定的简单策略选择动作
            next_state,reward,done,_ = env.step(action) # 交互一步
            onesequence.append((state, action, reward)) # MDP序列
            if done: # 游戏是否结束
                break
            state = next_state
        MDPsequence.append(onesequence)
        
    # 计算动作值，即策略评估
    for i in range(len(MDPsequence)):
        onesequence = MDPsequence[i]
        # 计算累积折扣奖励
        SA_pairs = []
        for j in range(len(onesequence)):
            sa_pair = (onesequence[j][0],onesequence[j][1])
            if sa_pair not in SA_pairs:
                SA_pairs.append(sa_pair)
                G = sum([x[2]*np.power(env.gamma, k) for 
                         k, x in enumerate(onesequence[j:])])
                r_sum[sa_pair] += G     # 合并累积折扣奖励
                r_count[sa_pair] += 1   # 记录次数
    for key in r_sum.keys():
        r_Q[key] = r_sum[key]/r_count[key] # 计算样本均值
        
    return r_Q, r_count

'''
主程序
'''
if __name__ == '__main__':
    env =blackjack.BlackjackEnv()                   # 定义环境模型
    env.gamma = 1.0                                 # 补充定义折扣系数
    r_Q, r_count = firstvisit_mc_actionvalue(env)   # 调用主函数
    # 打印结果
    for key, data in r_Q.items():
        print(key, r_count[key], ": ",data)