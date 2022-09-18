"""
基于MCTS的单次策略选择
"""

import numpy as np
import random
from collections import defaultdict

class MCTS_RL():
    def __init__(self,env,episode_max=100,num_mdpseq=100):
        self.env = env
        self.episode_max = episode_max
        self.num_mdpseq = num_mdpseq
        
        self.ontree_policy = self.create_ontree_policy()

    ## 创建初始树中策略
    def create_ontree_policy(self):
        ontree_policy = {} # 用字典表示策略
        for state in env.get_sspace(): # 遍历每一个状态
            ontree_policy[state] = None # 初始化为无策略
    
        return ontree_policy # 返回策略

    ## 添加或改进策略到树中策略
    def update_ontree_policy(self,state,action):
        if self.ontree_policy[state] == None:
            self.ontree_policy[state] = action
        else:
            self.ontree_policy[state] = action
    
    ## 树外策略
    def offtree_policy(self):    
        return random.choice(self.env.get_aspace())  # 随机选择一个动作
    
    ## 蒙特卡罗树搜索进行一次策略选择
    def mcts(self,state_cur):
        Q = {} # 当前状态下的各动作值容器
        for action_ in self.env.get_aspace(): # 遍历当前状态下的各动作
            Q[action_] = 0
            for i in range(self.num_mdpseq):  # 每个状态-动作对生成相同数目的完整mdp序列
                self.env.state = state_cur
                action = action_
                while True:                     # 生成搜索树
                    state,reward,done,info = self.env.step(action)
                    # 如果到达终止状态
                    if done:
                        Q[action_] += reward    # 回溯搜索树
                        break
                
                    # 根据树外或树中策略选择动作
                    if self.ontree_policy[state] == None:
                        action = self.offtree_policy(env)
                    else:
                        action = self.ontree_policy[state]
                    
        action_opt = np.argmax(np.array([x for x in Q.values()]))+1
    
        return action_opt
    
    ## 基于MCTS的强化学习
    def mcts_rl(self):
        for i in range(self.episode_max):
            state_cur = self.env.reset()
            while True:
                action = self.mcts(state_cur)
                self.update_ontree_policy(state_cur,action)
                cur_state,reward,done,info = self.env.step(action)
                if done:
                    break
        
        return self.ontree_policy
                
                

# main function
if __name__ == '__main__':
    import Count21
    env = Count21.Count21Env()
    
    episode_max = 100
    num_mdpseq = 100
    agent = MCTS_RL(env,episode_max,num_mdpseq)
    
    ontree_policy = agent.mcts()
    for key in ontree_policy.keys():
        print('{0}: {1}'.format(key,ontree_policy[key]))