# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:56:00 2021

@author: Longqiang

用期望Sarsa算法求解WindyWorld问题

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
    
    return __policy__                           # 返回epsilon-贪婪策略函数
        
## 期望Sarsa主程序
def Qlearning(env,num_episodes=1000,alpha=0.1,epsilon=0.1):
    NA = env.aspace_size
    # 初始化    
    Q = defaultdict(lambda: np.zeros(NA))           # 动作值函数
    egreedy_policy = create_egreedy_policy(env,Q,epsilon) # 贪婪策略函数
    
    # 外层循环
    for _ in range(num_episodes):
        state = env.reset()                     # 状态初始化
                
        # 内层循环
        while True:
            action_prob = egreedy_policy(state) # 产生当前动作
            action = np.random.choice(np.arange(NA),p=action_prob)
            next_state,reward,end,info = env.step(action) # 交互一次
            Q_exp = np.dot(Q[next_state],egreedy_policy(next_state))
            Q[state][action] += alpha*(         # 策略评估
                    reward+env.gamma*Q_exp-Q[state][action])
            
            # 检查是否到达终止状态
            if end: 
                break                            
            
            # 更新状态，进入下一次循环
            state = next_state
    
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
    P_table, Q = Qlearning(env,num_episodes=5000,alpha=0.1,epsilon=0.1)
    
    # 输出   
    print('P = ',P_table)
    for state in env.get_sspace():
        print('{}: {}'.format(state,Q[state]))