"""
基于MCTS的单次策略选择
"""

import numpy as np
import random

## 树外策略
def offtree_policy(env):    
    return random.choice(env.get_aspace())  # 随机选择一个动作

## 树中策略
def create_ontree_policy(env):              # 如表3-1所示
    ontree_policy = {}
    for state in env.get_sspace():
        ontree_policy[state] = None
    ontree_policy[3] = 2
    ontree_policy[5] = 1
    ontree_policy[8] = 2    
    ontree_policy[12] = 1
    ontree_policy[17] = 2    
    ontree_policy[20] = 1
    
    return ontree_policy

# 蒙特卡罗树搜索进行一次策略选择
def mcts(env,state_cur,ontree_policy,num_mdpseq=100):
    Q = {} # 当前状态下的各动作值容器
    for action_ in env.get_aspace(): # 遍历当前状态下的各动作
        Q[action_] = 0
        for i in range(num_mdpseq):  # 每个状态-动作对生成相同数目的完整mdp序列
            env.state = state_cur
            action = action_
            while True:                     # 生成搜索树
                state,reward,done,info = env.step(action)
                # 如果到达终止状态
                if done:
                    Q[action_] += reward    # 回溯搜索树
                    break
                
                # 根据树外或树中策略选择动作
                if ontree_policy[state] == None:
                    action = offtree_policy(env)
                else:
                    action = ontree_policy[state]
                    
    action_opt = np.argmax(np.array([x for x in Q.values()]))+1
    
    return Q,action_opt

# main function
if __name__ == '__main__':
    import Count21
    env = Count21.Count21Env()
    
    ontree_policy = create_ontree_policy(env)
    num_mdpseq = 100000
    state_cur = 10
    Q, action_opt = mcts(env,state_cur,ontree_policy,num_mdpseq)
    
    print(Q,action_opt)

    


            
            
            
            
            
            
            
            