# -*- coding: utf-8 -*-
"""
Created on Thu May 27 08:18:51 2021

@author: LONG QIANG
"""

import random

"""
环境模型
"""

class Count21Env():    
    def __init__(self,num_max=21,num_least=1,num_most=3,gamma=1):
        self.num_max = num_max      # 数数的终点，也是终止状态
        self.num_least = num_least  # 每次最少数的数
        self.num_most = num_most    # 每次最多数的数
        self.start = 0              # 数数的起点，也是初始状态
        self.goal = num_max         # 终止状态
        self.state = None           # 当前状态
        self.gamma = gamma          # 折扣系数        

        self.sspace_size = self.num_max+1                 # 状态个数
        self.aspace_size = self.num_most-self.num_least+1 # 动作个数
            
    ## 获取状态空间
    def get_sspace(self):
        return [i for i in range(self.start, self.num_max+1)]
    
    ## 获取动作空间
    def get_aspace(self):
        return [i for i in range(self.num_least, self.num_most+1)]
    
    ## 环境初始化
    def reset(self):   
        self.state = self.start
        return self.state
    
    # 庄家策略，作为环境的一部分，随机选择一个动作
    def get_dealer_action(self):
        return random.choice(self.get_aspace())
        
    # 进行一个时间步的交互
    def step(self,action):
        self.state += action            # 玩家数action个数

        if self.state > self.goal:      # 超过终止状态，庄家获胜
            reward = -1                 # 庄家获胜，玩家得-1分
            end = True
            info = "Player count then lose"
        elif self.state == self.goal:   # 到达终止状态，玩家获胜
            reward = 1                  # 玩家获胜，玩家得1分
            end = True
            info = "Player count then win"
        else:                           # 庄家继续数数
            self.state += self.get_dealer_action()  # 庄家数数
            if self.state > self.goal:              # 超过终止状态，玩家获胜
                reward = 1                          # 玩家获胜，玩家得1分
                end = True
                info = "Dealer count then lose"
            elif self.state == self.goal:
                reward = -1                         # 庄家获胜，玩家得-1分
                end = True
                info = "Dealer count then win"
            else:
                reward = 0                          # 游戏继续，玩家的0分
                end = False
                info = "Keep Going"
        
        return self.state,reward,end,info            
        
"""
主程序
"""

if __name__ == '__main__':
    env = Count21Env(21,1,3)
    
    print(env.get_sspace())
    print(env.get_aspace())
    
    for _ in range(10):
        seq = []
        state = env.reset()
        seq.append(state)
        while True:
            action = random.choice(env.get_aspace())
            state, reward, done, info = env.step(action)
            seq.append(state)
            if done:
                seq.append((info,reward))
                print(seq)
                break