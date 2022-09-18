# -*- coding: utf-8 -*-
"""
Created on Thu May 27 08:18:51 2021

风世界的环境模型

@author: LONG QIANG
"""

import random

class WindyWorldEnv():  
    def __init__(self,world_height=7,world_width=10,
                 wind=[0,0,0,1,1,1,2,2,1,0],
                 start=(3,0),goal=(3,7),gamma=1):
        self.world_height = world_height    # 网格高度 
        self.world_width = world_width      # 网格宽度
        self.wind = wind                    # 网格各列的风强度
        self.start = start                  # 初始状态网格
        self.goal = goal                    # 终止状态网格
        self.gamma = gamma                  # 折扣系数
        self.state = None                   # 环境当前状态
             
        # 用数字表示各个动作
        self.action_up = 0 
        self.action_down = 1
        self.action_left = 2
        self.action_right = 3
        
        self.sspace_size = self.world_height*self.world_width   # 状态数
        self.aspace_size = 4                                    # 动作数
        
        # 默认设置随机数种子
        self.seed()
    
    ## 设置随机数种子
    def seed(self, seed=None):
        return random.seed(seed)
    
    ## 获取动作空间
    def get_aspace(self):
        return [self.action_up,self.action_down,
                self.action_left,self.action_right]
    
    ## 获取状态空间
    def get_sspace(self):
        state_space = []
        for i in range(self.world_height):
            for j in range(self.world_width):
                state_space.append((i,j))
        return state_space
    
    ## 环境状态初始化
    def reset(self):
        self.state = self.start
        return self.state
    
    ## 一个时间步的交互
    def step(self,action):
        i,j = self.state
        
        # 先考虑按照action移动的过程
        if action == self.action_up:
            next_state = (max(i-1,0),j)
        elif action == self.action_down:
            next_state = (min(i+1,self.world_height-1),j)
        elif action == self.action_left:
            next_state = (i,max(j-1,0))
        elif action == self.action_right:
            next_state = (i,min(j+1,self.world_width-1))
        
        # 再考虑按照风吹移动的过程
        wind_count = 0                      # 风吹移动计数器
        wind_max = self.wind[next_state[1]] # 最大风吹移动格数
        while True:            
            if next_state == self.goal:     # 若已经到达目标位置
                reward = 0
                end = True
                info = 'Game Over'
                break
            elif wind_count == wind_max:    # 风吹移动结束，未到目标位置
                reward = -1
                end = False
                info = 'Keep Going'
                break

            i,j = next_state
            next_state = (max(i-1,0),j)     # 风吹移动一格       
            wind_count += 1
        
        self.state = next_state
        
        return next_state,reward,end,info            
        
# =============================================================================
# main code
# =============================================================================

if __name__ == '__main__':
    env = WindyWorldEnv()
    
    print(env.get_sspace())
    print(env.get_aspace())
    env.reset()
    for _ in range(100):
        action = random.choice(env.get_aspace())
        next_state,reward,end,info = env.step(action)
        print(action,next_state,reward,end,info)
        if end:
            break
        
    
    