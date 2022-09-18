'''
增量式异策略每次访问蒙特卡罗策略评估，计算动作值函数
目标策略：如果player手中牌的点数大于等于18，则停止要牌，否则继续要牌
行为策略：平均选择要牌或停牌
'''

import numpy as np
import blackjack
from collections import defaultdict

'''
目标策略：点数小于18则继续叫牌，否则停牌
'''
def target_policy(state): 
    player, dealer, ace = state
    return 0 if player >= 18 else 1 # 0：停牌，1：要牌

'''
行为策略：均匀选择叫牌或停牌
'''
def behavior_policy(state): 
    if np.random.rand() <= 0.5:
        return 0 # 0：停牌
    else:
        return 1 # 1：要牌

'''
增量式异策略每次访问蒙特卡罗策略评估：算法3-7的具体实现
'''
def offpolicy_firstvisit_mc_actionvalue(env,num_episodes=1000000):
    G_count = defaultdict(float)    # 记录状态-动作对的累积折扣奖励次数
    W_sum = defaultdict(float)      # 记录状态-动作对的累积重要性权重
    Q_bar_ord = defaultdict(float)  # 一般重要性采样动作值估计
    Q_bar_wei = defaultdict(float)  # 加权重要性采样动作值估计
    
    for i in range(num_episodes):
        # 采集一条经验轨迹
        state = env.reset()         # 环境状态初始化
        one_mdp_seq = []            # 经验轨迹容器
        while True:
            action = behavior_policy(state)             # 按行为策略选择动作
            next_state,reward,done,_ = env.step(action) # 交互一步
            one_mdp_seq.append((state, action, reward)) # MDP序列
            if done: # 游戏是否结束
                break
            state = next_state
        
        # 自后向前依次遍历MDP序列中的所有状态-动作对
        G = 0
        W = 1
        for j in range(len(one_mdp_seq)-1,-1,-1):
            sa_pair = (one_mdp_seq[j][0],one_mdp_seq[j][1])
            G = G+env.gamma*one_mdp_seq[j][2]       # 累积折扣奖励
            W = W*(target_policy(sa_pair[0])/0.5)   # 重要性权重
            if W == 0:                              # 权重为0则退出本层循环
                break
            W_sum[sa_pair] += W                     # 权重之和
            G_count[sa_pair] += 1                   # 记录次数
                                                    # 一般重要性采样估计
            Q_bar_ord[sa_pair] += (G-Q_bar_ord[sa_pair])/G_count[sa_pair]   
                                                    # 加权重要性采样估计
            Q_bar_wei[sa_pair] += (G-Q_bar_ord[sa_pair])*W/W_sum[sa_pair]         
        
    return Q_bar_ord, Q_bar_wei

'''
主程序
'''
if __name__ == '__main__':
    env = blackjack.BlackjackEnv()  # 导入环境模型
    env.gamma = 1                   # 补充定义折扣系数
    
    Q_bar_ord,Q_bar_wei = offpolicy_firstvisit_mc_actionvalue(env) 

    print('Ordinary action value of ((13,2,True),1) is {}'.
          format(Q_bar_ord[((13,2,True),1)]))
    print('Weighted action value of ((13,2,True),1) is {}'.
          format(Q_bar_wei[((13,2,True),1)]))