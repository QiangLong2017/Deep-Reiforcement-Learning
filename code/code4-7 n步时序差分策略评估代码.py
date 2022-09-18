# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:27:17 2021

@author: LONG QIANG

n步时序差分策略评估算法
评估风世界平均策略
"""

import numpy as np
from collections import defaultdict

## 平均策略
def even_policy(env,state):
    NA = env.aspace_size
    action_prob = np.ones(NA)/NA
    return action_prob

## 时序差分策略评估
def nstep_TD_actionvalue(env,nstep=3,alpha=0.1,delta=0.1):
    NA = env.aspace_size
    aspace = env.get_aspace()
    Q = defaultdict(lambda: np.zeros(NA))   # 动作值函数
    error = 0.0                             # 前后两次动作值最大差值
    
    # 外层循环直到动作值改变小于容忍系数
    while error <= delta:   
        state = env.reset()                 # 环境状态初始化
        nstep_mdp = []                      # 存储n步交互数据
        action_prob = even_policy(env,state)#平均策略
        action = np.random.choice(aspace,p=action_prob)
        
        # 内层循环直到到达终止状态
        while True:                
            next_state,reward,end,info = env.step(action)
            nstep_mdp.append((state,action,reward))
            if len(nstep_mdp) < nstep:
                if end == True:
                    # 还未到n步已到达终止状态，则直接退出
                    break
                else:
                    # 根据平均策略选择一个动作
                    action_prob = even_policy(env,next_state)
                    next_action = np.random.choice(aspace,p=action_prob)            
            if len(nstep_mdp) >= nstep:
                if end == False:
                    # 根据平均策略选择一个动作
                    action_prob = even_policy(env,next_state)
                    next_action = np.random.choice(aspace,p=action_prob)
                    
                    # 之前第n步的动作和状态
                    state_n,action_n = nstep_mdp[0][0],nstep_mdp[0][1]
                    Q_temp = Q[state_n][action_n]       # 临时保存旧值
                    
                    # 计算n步TD目标值G
                    Re = [x[2] for x in nstep_mdp]
                    Re_sum = sum([env.gamma**i*re for (i,re) in enumerate(Re)])
                    G = Re_sum+env.gamma**nstep*Q[next_state][next_action]
                    
                    # n步时序差分更新
                    Q[state_n][action_n] += alpha*(G-Q[state_n][action_n])
                    
                    # 更新最大误差
                    error = max(error,abs(Q[state_n][action_n]-Q_temp))
                    
                    # 删除n步片段中最早一条交互数据
                    nstep_mdp.pop(0)

                else: # 已到达终止状态，处理剩下不足n步的交互数据
                    for i in range(len(nstep_mdp)):
                        state_i = nstep_mdp[i][0]       # 状态
                        action_i = nstep_mdp[i][1]      # 动作
                        Q_temp = Q[state_i][action_i]   # 临时保存旧值
                        
                        # 计算剩下部分TD目标值G
                        Re = [x[2] for x in nstep_mdp[i:]]
                        G = sum([env.gamma**i*re for (i,re) in enumerate(Re)])
                        
                        # 时序差分更新
                        Q[state_i][action_i] += alpha*(G-Q[state_i][action_i])
                        
                        # 更新最大误差
                        error = max(error,abs(Q[state_i][action_i]-Q_temp))
                    
                    break # 本轮循环结束
            
            state = next_state  # 更新动作和状态
            action = next_action
    
    return Q

## 主函数
if __name__ == '__main__':
    import WindyWorld
    env = WindyWorld.WindyWorldEnv()
    
    Q = nstep_TD_actionvalue(env,nstep=3,alpha=0.1,delta=0.1)
    
    for state in env.get_sspace():
        print(state, ' : ',Q[state])
    
    
    
    