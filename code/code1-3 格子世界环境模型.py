import numpy as np
from gym.utils import seeding

class GridWorldEnv():    
    ## 初始化类
    def __init__(self,grid_height=3,grid_width=4,start=(0,0),goal=(0,3),
                 obstacle=(1,1)):
        self.grid_height = grid_height      # 网格高度
        self.grid_width = grid_width        # 网格宽度
        self.start = start                  # 初始状态
        self.goal = goal                    # 目标状态
        self.obstacle = obstacle            # 障碍物位置
        self.state = None                   # 环境的当前状态
        self.gamma = 0.9                    # 折扣系数
        self.seed()                         # 默认设置随机数种子                   
        
        # 用0,1,2,3分别表示上、下、左、右动作。
        self.action_up = 0
        self.action_down = 1
        self.action_left = 2
        self.action_right = 3
        
        # 状态和动作空间大小
        self.state_space_size = self.grid_height*self.grid_width
        self.action_space_size = 4     
        
    ## 获取整个状态空间，用格子的坐标表示状态，用一个列表储存
    def get_state_space(self):
        state_space = []
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                state_space.append((i,j))
        
        return state_space
    
    ## 将状态转为自然数编号
    def state_to_number(self,state):
        return self.get_state_space().index(state)
    
    ## 将自然数编号转为相应的状态
    def number_to_state(self,number):
        return self.get_state_space()[number]

    ## 获取整个动作空间，用一个列表储存
    def get_action_space(self):
        return [self.action_up,self.action_down,self.action_left,
                self.action_right]

    ## 设置随机数种子
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ## 状态转移概率矩阵
    def Psa(self):
        # 定义一个张量来储存状态转移概率,大部分状态转移概率为0
        Psa = np.zeros((self.state_space_size,self.action_space_size,
                       self.state_space_size)) 
        # 逐个赋值状态转移概率
        # 当前状态为(0,0)
        Psa[0,0,0],Psa[0,0,1] = 0.9,0.1
        Psa[0,1,0],Psa[0,1,1],Psa[0,1,4] = 0.1,0.1,0.8
        Psa[0,2,0],Psa[0,2,4] = 0.9,0.1
        Psa[0,3,0],Psa[0,3,4],Psa[0,3,1] = 0.1,0.1,0.8
        # 当前状态为(0,1)
        Psa[1,0,0],Psa[1,0,1],Psa[1,0,2] = 0.1,0.8,0.1
        Psa[1,1,0],Psa[1,1,1],Psa[1,1,2] = 0.1,0.8,0.1
        Psa[1,2,0],Psa[1,2,1] = 0.8,0.2
        Psa[1,3,1],Psa[1,3,2] = 0.2,0.8
        # 当前状态为(0,2)
        Psa[2,0,1],Psa[2,0,2],Psa[2,0,3] = 0.1,0.8,0.1
        Psa[2,1,1],Psa[2,1,6],Psa[2,1,3] = 0.1,0.8,0.1
        Psa[2,2,1],Psa[2,2,2],Psa[2,2,6] = 0.8,0.1,0.1
        Psa[2,3,2],Psa[2,3,3],Psa[2,3,6] = 0.1,0.8,0.1
        # 当前状态为(1,0)
        Psa[4,0,0],Psa[4,0,4] = 0.8,0.2
        Psa[4,1,8],Psa[4,1,4] = 0.8,0.2
        Psa[4,2,0],Psa[4,2,4],Psa[4,2,8] = 0.1,0.8,0.1
        Psa[4,3,0],Psa[4,3,4],Psa[4,3,8] = 0.1,0.8,0.1
        # 当前状态为(1,2)
        Psa[6,0,2],Psa[6,0,6],Psa[6,0,7] = 0.8,0.1,0.1
        Psa[6,1,10],Psa[6,1,6],Psa[6,1,7] = 0.8,0.1,0.1
        Psa[6,2,6],Psa[6,2,2],Psa[6,2,10] = 0.8,0.1,0.1
        Psa[6,3,2],Psa[6,3,7],Psa[6,3,10] = 0.1,0.8,0.1
        # 当前状态为(1,3)
        Psa[7,0,3],Psa[7,0,6],Psa[7,0,7] = 0.8,0.1,0.1
        Psa[7,1,11],Psa[7,1,6],Psa[7,1,7] = 0.8,0.1,0.1
        Psa[7,2,6],Psa[7,2,3],Psa[7,2,11] = 0.8,0.1,0.1
        Psa[7,3,3],Psa[7,3,7],Psa[7,3,11] = 0.1,0.8,0.1
        # 当前状态为(2,0)
        Psa[8,0,4],Psa[8,0,8],Psa[8,0,9] = 0.8,0.1,0.1
        Psa[8,1,8],Psa[8,1,9] = 0.9,0.1
        Psa[8,2,8],Psa[8,2,4] = 0.9,0.1
        Psa[8,3,4],Psa[8,3,9],Psa[8,3,8] = 0.1,0.8,0.1
        # 当前状态为(2,1)
        Psa[9,0,9],Psa[9,0,8],Psa[9,0,10] = 0.8,0.1,0.1
        Psa[9,1,9],Psa[9,1,8],Psa[9,1,10] = 0.8,0.1,0.1
        Psa[9,2,8],Psa[9,2,9] = 0.8,0.2
        Psa[9,3,10],Psa[9,3,9] = 0.8,0.2
        # 当前状态为(2,2)
        Psa[10,0,6],Psa[10,0,9],Psa[10,0,11] = 0.8,0.1,0.1
        Psa[10,1,9],Psa[10,1,10],Psa[10,1,11] = 0.1,0.8,0.1
        Psa[10,2,9],Psa[10,2,6],Psa[10,2,10] = 0.8,0.1,0.1
        Psa[10,3,6],Psa[10,3,10],Psa[10,3,11] = 0.1,0.1,0.8
        # 当前状态为(2,3)
        Psa[11,0,7],Psa[11,0,10],Psa[11,0,11] = 0.8,0.1,0.1
        Psa[11,1,10],Psa[11,1,11] = 0.1,0.9
        Psa[11,2,10],Psa[11,2,7],Psa[11,2,11] = 0.8,0.1,0.1
        Psa[11,3,11],Psa[11,3,7] = 0.9,0.1
       
        return Psa        
    
    ## 即时奖励函数
    def Rsa(self,s,a,s_):
        # 以曼哈顿距离的倒数作为及时奖励
#        if s_ == self.goal:
#            reward = 2
#        else:
#            dis = abs(s_[0]-self.goal[0])+abs(s_[1]-self.goal[1])
#            reward = 1.0/dis
        
        # 到达目标位置则奖励1，否则不奖励
        if s_ == self.goal:
            reward = 1
        else:
            reward = 0
        
        return reward
    
    ## 状态初始化
    def reset(self):
        self.state = (0,0)      
        return self.state
    
    ## 一个时间步的环境交互，返回下一个状态，即时奖励，是否终止，日志
    def step(self,action):
        s = self.state_to_number(self.state)
        a = action
        s_ = np.random.choice(np.array(range(self.state_space_size)),
                              p=self.Psa()[s,a]) # 依概率选择一个状态
        next_state = self.number_to_state(s_)
        reward = self.Rsa(self.state,a,next_state)
        if next_state == self.goal:
            end = True
            info = 'Goal Obtained'
        else:
            end = False
            info = 'Keep Going'        
        self.state = next_state
        
        return next_state,reward,end,info            
    
    ## 可视化模拟函数，仅仅占位，无功能
    def render(self):
        return None
    
    ## 结束环境函数，仅仅占位，无功能
    def close(self):
        return None
        
# =============================================================================
# main code
# =============================================================================

import random

if __name__ == '__main__':
    env = GridWorldEnv()
    
    print(env.get_state_space())        # 打印状态空间
    print(env.get_action_space())       # 打印动作空间
    print(env.Psa())                    # 打印概率转移矩阵
    
    # 进行若干次环境交互
    env.reset()
    for _ in range(100):
        action = random.choice(env.get_action_space())  # 随机选择动作
        next_state,reward,end,info = env.step(action)   # 一次环境交互
        print(next_state,reward,end,info)               # 打印交互结果
        if end == True:                                 # 到达目标状态
            break