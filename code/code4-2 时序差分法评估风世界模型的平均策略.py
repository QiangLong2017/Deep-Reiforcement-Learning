# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:27:17 2021

@author: LONG QIANG

时序差分策略评估算法
评估风世界平均策略
"""

import numpy as np
from collections import defaultdict

## 平均策略
def even_policy(env,state):
    action_prob = np.ones(env.aspace_size)/env.aspace_size
    return action_prob

## 时序差分策略评估
def TD_actionvalue(env,alpha=0.01,num_episodes=100):
    Q = defaultdict(lambda: np.zeros(env.aspace_size))      # 初始化动作值
    
    for _ in range(num_episodes): 
        state = env.reset() # 环境状态初始化
        action_prob = even_policy(env,state)                #平均策略
        action = np.random.choice(env.get_aspace(),p=action_prob)
        
        # 内部循环直到到达终止状态
        while True:                
            next_state,reward,end,info = env.step(action)   #交互一步
            action_prob = even_policy(env,next_state)       # 平均策略
            next_action = np.random.choice(env.get_aspace(),p=action_prob)
            Q[state][action] += alpha*(reward
                                 +env.gamma*Q[next_state][next_action]
                                 -Q[state][action])         # 时序差分更新
            if end:                 # 检查是否到达终止状态
                break
            
            state = next_state      # 更新动作和状态
            action = next_action
    
    return Q
            
## 主函数
if __name__ == '__main__':
    import WindyWorld
    env = WindyWorld.WindyWorldEnv()
    
    alpha,num_episodes = 0.01,100
    Q = TD_actionvalue(env,alpha,num_episodes)
    
    for state in env.get_sspace():
        print(state, ' : ',Q[state])
    
    
    