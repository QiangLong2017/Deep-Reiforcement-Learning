import numpy as np

'''
创建一个随机确定性策略
'''
def create_random_greedy_policy(env):
    random_greedy_policy = {}               # 用字典表示策略
    for state in env.get_state_space():     # 遍历每一个状态
        random_greedy_policy[state] = np.zeros(env.action_space_size)
        # 随机选择一个动作，设置其概率为1
        action = np.random.choice(range(env.action_space_size))
        random_greedy_policy[state][action] = 1.0
    
    return random_greedy_policy             # 返回策略

'''
策略评估函数。该函数是算法2-1的具体实现。
函数输入环境模型和当前策略，输出状态值。
'''
def policy_evaluation(env,policy):
    theta=0.001                             # 容忍参数
    Psa = env.Psa()                         # 获取状态转移概率矩阵
    V = np.random.rand(env.state_space_size)# 初始化状态值
    
    # 迭代求解Bellman方程
    for _ in range(500):
        delta = 0 # 初始化绝对差值
        
        # 对每一个状态进行循环
        for s_i,s in enumerate(env.get_state_space()):
            v = 0                               # 初始化s对应的更新状态值
            for a_i,a in enumerate(env.get_action_space()):
                temp = 0
                for ns_i,ns in enumerate(env.get_state_space()):
                    reward = env.Rsa(s,a,ns)    # (s,a)转移到ns的即时奖励
                    prob = Psa[s_i,a_i,ns_i]    # (s,a)转移到ns的概率
                    temp += prob*(reward+env.gamma*V[ns_i])
                v += policy[s][a_i]*temp        # s对应的更新状态值
            
            delta = max(delta,np.abs(v-V[s_i])) # 维持更新前后最大绝对差值
            V[s_i] = v                          # 状态值更新
        
        if delta <= theta:                      # 检查是否满足终止条件
            break
    
    return V                                    # 返回状态值

'''
策略改进函数。该函数是算法2-2的具体实现。
函数输入环境模型、当前策略和状态值，
输出改进后的策略和新旧策略是否一样的指示值。
若新旧策略不一样，则“no_policy_change = False”，说明策略的确有改进。
若新旧策略一样，则“no_policy_change = True”，说明策略已经达到最优。
'''
def policy_update(env,policy,V):
    Psa = env.Psa()                             # 获取状态转移概率矩阵 
    policy_new = policy                         # 初始化一个新的策略
    
    # 策略更新标志，True：策略有更新，False：策略无更新
    no_policy_change = True
    
    # 对每一个状态进行循环
    for s_i,s in enumerate(env.get_state_space()): 
        old_action = np.argmax(policy[s])       # 当前贪心策略
            
        # 计算新的贪心策略
        action_values = np.zeros(env.action_space_size)
        for a_i,a in enumerate(env.get_action_space()):
            for ns_i,ns in enumerate(env.get_state_space()):
                reward = env.Rsa(s,a,ns)        # (s,a)转移到ns的即时奖励
                prob = Psa[s_i,a_i,ns_i]        # (s,a)转移到ns的概率
                action_values[a_i] += prob*(reward+env.gamma*V[ns_i])
            
        # 采用贪婪算法更新当前策略
        best_action = np.argmax(action_values)
        policy_new[s] = np.eye(env.action_space_size)[best_action]
        
        # 判断策略是否有改进
        if old_action != best_action:
            no_policy_change = False
    
    return policy_new, no_policy_change         # 返回新策略，策略改进标志

'''
将策略用矩阵表示。由于在计算过程中，策略是用Python字典格式来存储的，
不便于直观阅读，故在最终输出时将其转化为和环境网格配套的矩阵。
'''
def policy_express(env,policy):
    policy_mat = np.zeros((env.grid_height,env.grid_width))
    for s in env.get_state_space():
        policy_mat[s[0]][s[1]] = np.argmax(policy[s])
    
    return policy_mat
    
'''
策略迭代算法主程序。该函数是算法2-3的具体实现。
首先初始化确定性策略，然后策略评估和策略改进的交替循环，
每次循环都要判断策略是否的确有更新，
若无更新则说明算法已经收敛到最优策略，迭代终止。
'''
def policy_iteration(env,episode_limit=100):  
    policy = create_random_greedy_policy(env)   # 创建初始贪婪策略
    
    # 策略迭代过程
    for i in range(episode_limit):
        print('第{}次迭代'.format(i))
        # 评估当前策略
        V = policy_evaluation(env,policy)
        print('V=',V)
        # 更新当前策略
        policy,no_policy_change = policy_update(env,policy,V) 
        print('policy=',policy)
        # 若所有状态下的策略都不再更新，则停止迭代
        if no_policy_change:
            print('Iteration terminate with stable policy.')
            break
    
    # 将决策表示成矩阵形式
    policy_mat = policy_express(env,policy) 
            
    # 返回最优策略和对应状态值
    return policy,policy_mat, V

'''
主程序，主要调用策略迭代主程序函数
'''
if __name__ == '__main__':
    import GridWorld
    env = GridWorld.GridWorldEnv()
    policy_opt,policy_mat,V_opt = policy_iteration(env,episode_limit=100)

    print(policy_mat)
    print(V_opt)