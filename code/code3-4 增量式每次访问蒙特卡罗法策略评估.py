# -*- coding: utf-8 -*-
"""
Created on Sun May 23 07:50:12 2021

@author: LONG QIANG

增量式每次访问蒙特卡罗策略评估，计算动作值函数
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
增量式每次访问蒙特卡罗策略评估：算法3-2在每次访问范式下的具体实现
'''
def everyvisit_incremental_mc_actionvalue(env,num_episodes=50000):
    r_count = defaultdict(float)    # 记录状态-动作对的累积折扣奖励次数
    r_Q = defaultdict(float)        # 动作值样本均值
    
    # 逐次采样并计算
    for i in range(num_episodes):
        # 采样一条经验轨迹
        state = env.reset()         # 环境状态初始化
        onesequence = []            # 一条经验轨迹容器
        while True:
            action = simple_policy(state) # 根基给定的简单策略选择动作
            next_state,reward,done,_ = env.step(action) # 交互一步
            onesequence.append((state, action, reward)) # MDP序列
            if done:
                break
            state = next_state

        # 逐个更新动作值样本均值            
        for j in range(len(onesequence)):
            sa_pair = (onesequence[j][0],onesequence[j][1])
            G = sum([x[2]*np.power(env.gamma, k) for 
                         k, x in enumerate(onesequence[j:])])
            r_count[sa_pair] += 1   # 记录次数
            r_Q[sa_pair] += (1.0/r_count[sa_pair])*(G-r_Q[sa_pair])
        
    return r_Q, r_count

'''
主程序
'''
if __name__ == '__main__':
    env =blackjack.BlackjackEnv()                   # 定义环境模型
    env.gamma = 1.0                                 # 补充定义折扣系数
                                                    # 调用主函数
    r_Q,r_count = everyvisit_incremental_mc_actionvalue(env) 
    # 打印结果
    for key, data in r_Q.items():
        print(key, r_count[key], ": ",data)