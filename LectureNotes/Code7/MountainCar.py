"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import numpy as np
from gym import spaces
from gym.utils import seeding

class MountainCarEnv():
    def __init__(self):
        self.min_position = -1.2    # 最低点
        self.max_position = 0.6     # 最高点
        self.max_speed = 0.07       # 最大速度
        self.goal_position = -0.2   # 目标高度
        self.goal_velocity = 0      # 目标速度      
        self.force=0.001            # 推力
        self.gravity=0.0025         # 重量
        self.time = None            # 一个回合持续时间步

        self.low = np.array([self.min_position,
                             -self.max_speed],dtype=np.float32)
        self.high = np.array([self.max_position,
                              self.max_speed],dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low,
                                            self.high,dtype=np.float32)
        
        self.seed()
    
    def seed(self, seed=None):
        self.np_random,seed = seeding.np_random(seed)
        return seed

    def step(self, action):
        position,velocity = self.state
        velocity += (action-1)*self.force+math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity,-self.max_speed,self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): 
            velocity = 0
        
        self.state = [position, velocity]
        self.time += 1
        
        if position>=self.goal_position and velocity>=self.goal_velocity:
            done = True
            reward = 0
            info = 'Goal Obtained'
        elif self.time > 1000:
            done = True
            reward = -1
            info = 'Maximum Timesteps'
        else:
            done = False
            reward = -1
            info = 'Goal Obtained'

        return self.state, reward, done, info

    def reset(self):
        self.state = [self.np_random.uniform(low=-0.6, high=-0.4), 0]
        self.time = 0
        
        return self.state
    
if __name__ == '__main__':
    env = MountainCarEnv()
    s = env.reset()
    for i in range(2000):
        prob = np.random.rand(3,)
        prob = prob/np.sum(prob)
        a = np.random.choice(np.arange(3),p=prob)
        s_,r,end,info = env.step(a)
        print(i,s,a,s_,end,info)
        
        if end:
            break