import math
import numpy as np
from gym import spaces
from gym.utils import seeding

class MountainCarContinuousEnv():
    def __init__(self, goal_velocity=0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = -0.1 
        self.goal_velocity = 0
        self.power = 0.0015
        self.time = None

        self.low_state = np.array([self.min_position,
                                   -self.max_speed],dtype=np.float32)
        self.high_state = np.array([self.max_position, 
                                    self.max_speed],dtype=np.float32)
        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action,
                                       shape=(1,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state,
                                            dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def step(self, action):
        position,velocity = self.state
        force = np.clip(action,self.min_action,self.max_action)
        velocity += force*self.power-0.0025*math.cos(3*position)
        velocity = np.clip(velocity,-self.max_speed,self.max_speed)
        position += velocity
        position = np.clip(position,self.min_position,self.max_position)
        if (position==self.min_position and velocity<0): 
            velocity = 0

        self.state = [position, velocity]
        self.time += 1
        
        if position>=self.goal_position and velocity>=self.goal_velocity:
            done = True
            reward = 100
            info = 'Goal Obtained'
        elif self.time > 2000:
            done = True
            reward = math.pow(action,2)*0.1
            info = 'Maximum Timesteps'
        else:
            done = False
            reward = math.pow(action,2)*0.1
            info = 'Goal Obtained'

        return self.state, reward, done, info

    def reset(self):
        self.state = [self.np_random.uniform(low=-0.6,high=-0.4),0]
        self.time = 0
        
        return self.state

if __name__ == '__main__':
    env = MountainCarContinuousEnv()
    s = env.reset()
    for i in range(2000):
        a = env.min_action+np.random.rand()*(env.max_action-env.min_action)
        s_,r,end,info = env.step(a)
        print(i,s,a,s_,end,info)
        
        if end:
            break
        
    
    
    
    
    
    
    
    
    
    