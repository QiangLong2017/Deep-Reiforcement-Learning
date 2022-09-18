# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:56:00 2021

@author: Longqiang

用n-step Sarsa算法求解WindyWorld问题

"""

import numpy as np
from collections import defaultdict

## 创建一个epsilon-贪婪策略
def create_egreedy_policy(env,Q,epsilon=0.1):
    # 内部函数
    def __policy__(state):
        NA = env.aspace_size
        A = np.ones(NA,dtype=float)*epsilon/NA  # 平均设置每个动作概率
        best = np.argmax(Q[state])              # 选择最优动作
        A[best] += 1-epsilon                    # 设定贪婪动作概率
        return A
    
    return __policy__ # 返回epsilon-贪婪策略函数
        
## n-step Sarsa算法主程序
def nstep_sarsa(env,num_episodes=500,alpha=0.1,epsilon=0.1,nstep=3): 
    NA = env.aspace_size
    aspace = env.get_aspace()
    # 初始化        
    Q = defaultdict(lambda: np.zeros(NA))                 # 动作值
    egreedy_policy = create_egreedy_policy(env,Q,epsilon) # 贪婪策略函数
    
    # 外层循环
    for _ in range(num_episodes):
        state = env.reset()                     # 环境状态初始化
        nstep_mdp = []                      # 存储n步交互数据
        action_prob = egreedy_policy(state)     # 产生当前动作
        action = np.random.choice(aspace,p=action_prob)
        
        # 内层循环直到到达终止状态
        while True:
            next_state,reward,end,info = env.step(action) # 交互一次
            nstep_mdp.append((state,action,reward))   # 保留交互数据
            if len(nstep_mdp) < nstep:
                if end == True:
                    # 还未到n步已到达终止状态，则直接退出
                    break
                else:
                    # 根据平均策略选择一个动作
                    action_prob = egreedy_policy(next_state)
                    next_action = np.random.choice(aspace,p=action_prob)            
            if len(nstep_mdp) >= nstep:
                if end == False:
                    # 根据平均策略选择一个动作
                    action_prob = egreedy_policy(next_state)
                    next_action = np.random.choice(aspace,p=action_prob)
                    
                    # 之前第n步的动作和状态
                    state_n,action_n = nstep_mdp[0][0],nstep_mdp[0][1]
                    
                    # 计算n步TD目标值G
                    Re = [x[2] for x in nstep_mdp]
                    Re_sum = sum([env.gamma**i*re for (i,re) in enumerate(Re)])
                    G = Re_sum+env.gamma**nstep*Q[next_state][next_action]
                    
                    # n步时序差分更新
                    Q[state_n][action_n] += alpha*(G-Q[state_n][action_n])
                    
                    # 删除n步片段中最早一条交互数据
                    nstep_mdp.pop(0)
                else: # 已到达终止状态，处理剩下不足n步的交互数据
                    for i in range(len(nstep_mdp)):
                        state_i = nstep_mdp[i][0] # 状态
                        action_i = nstep_mdp[i][1]# 动作
                        
                        # 计算剩下部分TD目标值G
                        Re = [x[2] for x in nstep_mdp[i:]]
                        G = sum([env.gamma**i*re for (i,re) in enumerate(Re)])
                        
                        # 时序差分更新
                        Q[state_i][action_i] += alpha*(G-Q[state_i][action_i])
                                            
                    break       # 本轮循环结束
            
            state = next_state  # 更新动作和状态
            action = next_action            

    # 用表格表示最终策略
    P_table = np.ones((env.world_height,env.world_width))*np.inf
    for state in env.get_sspace():
        P_table[state[0]][state[1]] = np.argmax(Q[state])
     
    # 返回最终策略和动作值
    return P_table,Q
    
## 主程序
if __name__ == '__main__':
    # 构造WindyWorld环境
    import WindyWorld
    env = WindyWorld.WindyWorldEnv() 
    
    # 调用Sarsa算法
    P_table, Q = nstep_sarsa(
            env,num_episodes=5000,alpha=0.1,epsilon=0.1,nstep=3)
    
    # 输出   
    print('P = ',P_table)
    for state in env.get_sspace():
        print('{}: {}'.format(state,Q[state]))
    
    