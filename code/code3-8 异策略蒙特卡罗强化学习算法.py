# -*- coding: utf-8 -*-
"""
Created on Sun May 23 07:50:12 2021

@author: LONG QIANG

异策略蒙特卡洛强化学求解21点游戏，算法3-8具体实现
行为策略：平均选择要牌或停牌
"""
import numpy as np
import blackjack
from collections import defaultdict
import matplotlib.pyplot as plt

'''
异策略蒙特卡罗强化学习算法类
'''
class OffpolicyMCRL():
    ## 类初始化 
    def __init__(self,env,num_episodes=1000000):
        self.env = env
        self.nA = env.action_space.n                        # 动作空间维度
        self.Q_bar = defaultdict(lambda: np.zeros(self.nA)) # 动作值函数
        self.W_sum = defaultdict(lambda: np.zeros(self.nA)) # 累积重要性权重
        self.t_policy = defaultdict(lambda: np.zeros(self.nA))  # 目标策略
        self.b_policy = defaultdict(lambda: np.zeros(self.nA))  # 行为策略
        self.num_episodes = num_episodes                    # 最大抽样轮次
        
    ## 初始化及更新目标策略
    def target_policy(self,state): 
        if state not in self.t_policy.keys():
            player, dealer, ace = state
            action = 0 if player >= 18 else 1     # 0：停牌，1：要牌
        else:
            action = np.argmax(self.Q_bar[state]) # 最优动作值对应的动作
            self.t_policy[state] = np.eye(self.nA)[action]
            
        return self.t_policy[state]
    
    ## 初始化行为策略
    def behavior_policy(self,state):
        self.b_policy[state] = [0.5,0.5]
        return self.b_policy[state]
        
    ## 按照行为策略蒙特卡罗抽样产生一条经历完整的MDP序列
    def mc_sample(self):
        one_mdp_seq = []                # 经验轨迹容器
        state = self.env.reset()        # 初始状态
        while True:
            action_prob = self.behavior_policy(state)
            action = np.random.choice(np.arange(len(action_prob)),
                                      p=action_prob)
            next_state,reward,done,_ = env.step(action) # 交互一步
            one_mdp_seq.append((state,action,reward))   # 经验轨迹          
            state = next_state
            if done:  # 游戏是否结束
                break
        
        return one_mdp_seq
         
    ## 基于值迭代的增量式异策略蒙特卡罗每次访问策略评估和改进
    def offpolicy_everyvisit_mc_valueiter(self,one_mdp_seq):
        # 自后向前依次遍历MDP序列中的所有状态-动作对
        G = 0
        W = 1
        for j in range(len(one_mdp_seq)-1,-1,-1):
            state = one_mdp_seq[j][0]
            action = one_mdp_seq[j][1]
            G = G+env.gamma*one_mdp_seq[j][2]       # 累积折扣奖励
            W = W*(self.target_policy(state)[action]/self.behavior_policy(
                    state)[action])                 # 重要性权重
            if W == 0:                              # 权重为0则退出本层循环
                break
            self.W_sum[state][action] += W          # 权重之和
            self.Q_bar[state][action] += (
                    G-self.Q_bar[state][action])*W/self.W_sum[state][action]
            self.target_policy(state)               # 策略改进

    ## 蒙特卡罗强化学习
    def mcrl(self):
        for i in range(self.num_episodes):
            one_mdp_seq = self.mc_sample()          # 抽样一条MDP序列
                                                    # 蒙特卡罗策略评和策略改进
            self.offpolicy_everyvisit_mc_valueiter(one_mdp_seq)
        
        return self.t_policy, self.Q_bar            # 输入策略和动作值

    ## 绘制最优策略图像
    def draw(self,policy):
        true_hit = [(x[1],x[0]) for x in policy.keys(
                ) if x[2]==True and np.argmax(policy[x])==1]
        true_stick = [(x[1],x[0]) for x in policy.keys(
                ) if x[2]==True and np.argmax(policy[x])==0]
        false_hit = [(x[1],x[0]) for x in policy.keys(
                ) if x[2]==False and np.argmax(policy[x])==1]
        false_stick = [(x[1],x[0]) for x in policy.keys(
                ) if x[2]==False and np.argmax(policy[x])==0]
        
        plt.figure(1)
        plt.plot([x[0] for x in true_hit],
                 [x[1] for x in true_hit],'bo',label='HIT')
        plt.plot([x[0] for x in true_stick],
                 [x[1] for x in true_stick],'rx',label='STICK')
        plt.xlabel('dealer'), plt.ylabel('player')
        plt.legend(loc='upper right')
        plt.title('Usable Ace')
        filepath = 'code3-8 UsabelAce.png'
        plt.savefig(filepath, dpi=300)
        
        plt.figure(2)
        plt.plot([x[0] for x in false_hit],
                 [x[1] for x in false_hit],'bo',label='HIT')
        plt.plot([x[0] for x in false_stick],
                 [x[1] for x in false_stick],'rx',label='STICK')
        plt.xlabel('dealer'), plt.ylabel('player')
        plt.legend(loc='upper right')
        plt.title('No Usable Ace')
        filepath = 'code3-8 NoUsabelAce.png'
        plt.savefig(filepath, dpi=300)

'''
主程序
'''
if __name__ == '__main__':
    env = blackjack.BlackjackEnv()  # 导入环境模型
    env.gamma = 1                   # 补充定义折扣系数
    
    # 定义方法
    agent = OffpolicyMCRL(env)
    # 强化学习
    opt_policy,opt_Q = agent.mcrl()
    # 打印结果
    for key in opt_policy.keys():
        print(key,": ",opt_policy[key],opt_Q[key])
    agent.draw(opt_policy)