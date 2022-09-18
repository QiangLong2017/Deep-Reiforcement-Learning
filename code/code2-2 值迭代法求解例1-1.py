# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 00:00:17 2021

@author: LONG QIANG

值迭代算求解GridWorld问题
基于状态值函数
"""

import numpy as np

'''
创建一个随机确定性策略
'''
def create_random_greedy_policy(env):
    random_greedy_policy = {}               # 用字典表示策略
    for state in env.get_state_space():     # 遍历每一个状态
        random_greedy_policy[state] = np.zeros(env.action_space_size)
        # 随机选择一个动作，设置其概率为1
        random_greedy_policy[state][np.random.choice(range(env.action_space_size))] = 1.0
    
    return random_greedy_policy             # 返回策略

'''
迭代更新状态值函数
'''
def statevalue_update(env,V):
    V_new = np.zeros_like(V)                # 初始化新的状态值函数
    Psa = env.Psa()                         # 获取状态转移概率矩阵
    delta = 0                               # 值函数更新前后最大绝对差值
    epsilon = 0.001                         # 更新容忍系数
    no_value_change = True                  # 是否更新指示器
    
    # 对每一个状态进行循环
    for s_i,s in enumerate(env.get_state_space()):
        action_values = np.zeros(env.action_space_size)
        for a_i,a in enumerate(env.get_action_space()):
            for ns_i,ns in enumerate(env.get_state_space()):
                reward = env.Rsa(s,a,ns)    # (s,a)转移到ns的即时奖励
                prob = Psa[s_i,a_i,ns_i]    # (s,a)转移到ns的概率
                action_values[a_i] += prob*(reward+env.gamma*V[ns_i])
        V_new[s_i] = np.max(action_values)
        
        # 维持最大的增量
        delta = max(delta,np.abs(V_new[s_i]-V[s_i]))
        
    # 检查是否满足终止条件
    if delta >= epsilon:
        no_value_change = False
    
    return V_new, no_value_change

'''
策略改进函数，用贪婪法求解最优策略
'''
def policy_update(env,V):
    Psa = env.Psa()                                 # 获取状态转移概率矩阵
    policy = create_random_greedy_policy(env)       # 初始化策略

    # 求解最优策略        
    for s_i,s in enumerate(env.get_state_space()):  # 对每一个状态进行循环            
        action_values = np.zeros(env.action_space_size)
        for a_i,a in enumerate(env.get_action_space()):
            for ns_i,ns in enumerate(env.get_state_space()):
                reward = env.Rsa(s,a,ns)            # (s,a)转移到ns的即时奖励
                prob = Psa[s_i,a_i,ns_i]            # (s,a)转移到ns的概率
                action_values[a_i] += prob*(reward+env.gamma*V[ns_i])
            
        # 求解贪婪策略
        best_action = np.argmax(action_values)
        policy[s] = np.eye(env.action_space_size)[best_action]
        
    return policy

'''
将policy表示成矩阵形式
'''
def policy_express(env,policy):
    policy_mat = np.zeros((env.grid_height,env.grid_width))
    for s in env.get_state_space():
        policy_mat[s[0]][s[1]] = np.argmax(policy[s])
    
    return policy_mat
    
'''
值迭代主程序,该函数是算法2-4的具体实现
'''
def value_iteration(env,episode_limit=100):
    V = np.zeros(env.state_space_size)
    
    # 迭代法求解最优状态值
    for i in range(episode_limit):
        print('第{}次迭代'.format(i))        
        V,no_value_change = statevalue_update(env,V)
        print('V=',V)
        if no_value_change:
            print('Iteration terminate with stable state value.')
            break
        
    # 计算最优策略
    policy = policy_update(env,V)

    # 将决策表示成矩阵形式
    policy_mat = policy_express(env,policy)
            
    # 返回最优策略和对应状态值
    return policy,policy_mat, V

'''
主程序
'''
if __name__ == '__main__':
    import GridWorld
    env = GridWorld.GridWorldEnv()
    policy_opt,policy_mat,V_opt = value_iteration(env,episode_limit=100)
    print(policy_mat)
    print(V_opt)