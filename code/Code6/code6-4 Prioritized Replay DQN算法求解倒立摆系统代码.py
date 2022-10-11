# Cartpole by Prioritized Replay DQN using pyTorch

#========================================

import gym
import numpy as np
import copy
import torch
from torch import nn
import matplotlib.pyplot as plt

'''
定义Sum-Tree类
'''
class SumTree():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size  # SumTree页节点数量=经验回放池容量
                                        # 储存Sum-Tree的所有节点数值
        self.tree = np.zeros(2*buffer_size-1) 
                                        # 储存经验数据，对应所有叶节点
        self.Transition = np.zeros(buffer_size, dtype=object)  
        self.TR_index = 0               # 经验数据的索引
        
    ## 向Sum-Tree中增加一个数据    
    def add(self, priority, expdata):   # priority优先级，expdata经验数据
                                        # TR_index在树中的位置为ST_index
        ST_index = self.TR_index+self.buffer_size-1
                                        # 将expdata存入TR_index位置
        self.Transition[self.TR_index] = expdata
                                        # 将TR_index的优先级priority存入
                                        # SumTree的ST_index位置，并更新SumTree
        self.update(ST_index, priority)  
        self.TR_index += 1              # 指针往前跳动一个位置
        if self.TR_index >= self.buffer_size:
            self.TR_index = 0           # 若容量已满，将叶节点指针拨回0
            
    ## 在ST_index位置添加priority后，更新Sum-Tree
    def update(self, ST_index, priority):
                                        # ST_index位置的优先级改变量
        change = priority-self.tree[ST_index] 
        self.tree[ST_index] = priority  # 将优先级存入叶节点
        while ST_index != 0:            # 回溯至根节点
            ST_index = (ST_index-1)//2  # 父节点 
            self.tree[ST_index] += change

    ## 根据value抽样
    def get_leaf(self, value):
        parent_idx = 0                  # 父节点索引
        while True:     
            cl_idx = 2*parent_idx+1     # 左子节点索引
            cr_idx = cl_idx+1           # 右子节点索引        
            if cl_idx >= len(self.tree):# 检查是否已经遍历到底了 
                leaf_idx = parent_idx   # 父节点成为叶节点
                break                   # 已经到底了，停止遍历
            else:
                                        # value小于左子节点数值，遍历左子树
                if value <= self.tree[cl_idx]: 
                    parent_idx = cl_idx # 父节点更新，进入更下一层
                else:                   # 否则遍历右子树
                                        # 先减去左子节点数值
                    value -= self.tree[cl_idx] 
                    parent_idx = cr_idx # 父节点更新，进入更下一层
                                        # 将Sum-tree索引转成Transition索引
        TR_index = leaf_idx-self.buffer_size+1 
        
        return leaf_idx, self.tree[leaf_idx], self.Transition[TR_index]

    ## 根节点数值，即所有优先级总和
    def total_priority(self):
        return self.tree[0]

'''
定义经验回放技术类
'''
class Memory():
    def __init__(self, buffer_size):
        self.tree = SumTree(buffer_size)    # 创建一个Sum-Tree实例
        self.counter = 0                    # 经验回放池中数据条数
        self.epsilon = 0.01                 # 正向偏移以避免优先级为0
        self.alpha = 0.6                    # [0,1],优先级使用程度系数
        self.beta = 0.4                     # 初始IS值
        self.delta_beta = 0.001             # beta增加的步长
        self.abs_err_upper = 1.             # TD误差绝对值的上界

    ## 往经验回放池中装入一个新的经验数据
    def store(self, newdata):
                                            # 所有优先级中最大者
        max_priority = np.max(self.tree.tree[-self.tree.buffer_size:]) 
        if max_priority == 0:               # 设置首条数据优先级为优先级上界
            max_priority = self.abs_err_upper
                                            # 设置新数据优先级为当前最大优先级
        self.tree.add(max_priority, newdata)
        self.counter += 1

    ## 从经验回放池中取出batch_size个数据
    def sample(self, batch_size):
        # indexes储存取出的优先级在SumTree中索引，一维向量
        # samples存储去除的经验数据，二维矩阵
        # ISWeights储存权重，以为向量
        indexes,samples,ISWeights = np.empty(
                batch_size,dtype=np.int32),np.empty(
                (batch_size,self.tree.Transition[0].size)
                ),np.empty(batch_size)
        # 将优先级总和batch_size等分
        pri_seg = self.tree.total_priority()/batch_size 
        # IS值逐渐增加到1，然后保持不变
        self.beta = np.min([1., self.beta+self.delta_beta])  
        # 最小优先级占总优先级之比
        min_prob = np.min(self.tree.tree[-self.tree.buffer_size:]
                          )/self.tree.total_priority() 
        # 修正最小优先级占总优先级之比，当经验回放池未满和优先级为0时会用上
        if min_prob == 0: 
            min_prob = 0.00001
        for i in range(batch_size):
            a,b = pri_seg*i,pri_seg*(i+1)   # 第i段优先级区间
            value = np.random.uniform(a,b)  # 在第i段优先级区间随机生成一个数
            # 返回SumTree中索引，优先级数值，对应的经验数据
            index,priority,sample = self.tree.get_leaf(value)
                                            # 抽样出的优先级占总优先级之比
            prob = priority/self.tree.total_priority()
                                            # 计算权重
            ISWeights[i] = np.power(prob/min_prob,-self.beta)
            indexes[i],samples[i,:] = index,sample
            
        return indexes, samples, ISWeights

    ## 调整批量数据
    def batch_update(self, ST_indexes, abs_errors):
        abs_errors += self.epsilon  # 加上一个正向偏移，避免为0
                                    # TD误差绝对值不要超过上界
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
                                    # alpha决定在多大程度上使用优先级
        prioritys = np.power(clipped_errors, self.alpha) 
                                    # 更新优先级，同时更新树
        for index, priority in zip(ST_indexes, prioritys):
            self.tree.update(index, priority) 

'''
定义Q-网络类
'''
class NeuralNetwork(nn.Module):
    def __init__(self,input_size,output_size):
        super(NeuralNetwork,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 20),
                nn.ReLU(),                  
                nn.Linear(20, 20),          
                nn.ReLU(),                  
                nn.Linear(20, output_size), 
                )
    
    ## 前向传播函数    
    def forward(self, x):                   
        x = self.flatten(x)                 
        logits = self.linear_relu_stack(x)
        return logits

'''
定义Prioritized Replay DQN方法类
'''
class PriRepDQN():
    def __init__(self,env,epsilon=0.1,learning_rate=1e-1,
                 buffer_size=100,batch_size=32):
        self.replay_buffer = Memory(buffer_size) # 初始化经验回放池
        self.env = env                              
        self.epsilon = epsilon 
        self.learning_rate = learning_rate 
        self.buffer_size = buffer_size 
        self.batch_size = batch_size
        
        self.create_Q_network()         
        self.create_training_method()

    ## Q-网络生成函数
    def create_Q_network(self):
        self.Q_network = NeuralNetwork(
                self.env.state_dim,self.env.aspace_size)
        self.Q_network_t = copy.deepcopy(self.Q_network)
    
    ## Q-网络优化器生成函数
    def create_training_method(self):
        self.loss_fun = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.SGD(
                self.Q_network.parameters(),lr=self.learning_rate) 

    ## epsilon-贪婪策略函数
    def egreedy_action(self,state):
        state = torch.from_numpy(np.expand_dims(state,0))
        state = state.to(torch.float32)
        Q_value = self.Q_network.forward(state)     
        A = np.ones(self.env.aspace_size)*self.epsilon/self.env.aspace_size
        best = np.argmax(Q_value.detach().numpy()) 
        A[best] += 1-self.epsilon
        action = np.random.choice(range(self.env.aspace_size),p=A) 
        
        return action     

    ## 经验回放技术
    def perceive(self,state,action,reward,next_state,done):        
        one_hot_action = np.eye(self.env.aspace_size)[action]        
        expdata = np.hstack((state,one_hot_action,reward,next_state,done))
        self.replay_buffer.store(expdata)
        if self.replay_buffer.counter > self.batch_size:
            self.train_Q_network()

    ## Q-网络训练函数
    def train_Q_network(self):        
        # 从经验回放池中抽取一个批量
        ST_indexes,minibatch,ISWeights = self.replay_buffer.sample(
                self.batch_size)
        # 分离出状态批量和动作批量
        state_batch = minibatch[:,0:self.env.state_dim] 
        action_batch = minibatch[:,self.env.state_dim:self.env.state_dim
                                 +self.env.aspace_size]
        # 计算TD目标值
        y_batch = []
        for x in minibatch:
            if x[-1]:
                y_batch.append(x[self.env.state_dim+
                                 self.env.aspace_size])
            else:
                next_state = x[-self.env.state_dim-1:-1]
                temp = torch.from_numpy(
                        next_state).unsqueeze(0).to(torch.float32)
                value_next = self.Q_network_t(temp)
                td_target = x[2]+self.env.gamma*torch.max(value_next)
                y_batch.append(td_target.item())
        y_batch = np.array(y_batch)

        state_batch = torch.from_numpy(state_batch).to(torch.float32)
        action_batch = torch.from_numpy(action_batch).to(torch.float32)
        y_batch = torch.from_numpy(y_batch).to(torch.float32)

        self.Q_network.train()
        pred = torch.sum(torch.multiply(
                self.Q_network(state_batch),action_batch),dim=1)
        
        # Importance-Sample权重
        ISWeights = torch.from_numpy(ISWeights).to(torch.float32)
        pred, y_batch = ISWeights*pred,ISWeights*y_batch
        loss = self.loss_fun(pred,y_batch)
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step() 
        
        # 计算被抽取数据TD误差绝对值
        abs_errors = torch.abs(pred-y_batch).detach().numpy() 
        # 更新被抽取数据的优先级
        self.replay_buffer.batch_update(ST_indexes,abs_errors)          

    ## 训练函数
    def train(self,num_episodes=250,num_steps=1000):
        rewards = []
        for episode in range(num_episodes):    
            state = self.env.reset()
            reward_sum = 0
            for step in range(num_steps):
                action = self.egreedy_action(state) 
                next_state,reward,done,_ = self.env.step(action)
                reward_sum += reward
                self.perceive(state,action,reward,next_state,done) 
                state = next_state
                if (step+1)%5 == 0:
                    self.Q_network_t.load_state_dict(
                            self.Q_network.state_dict())                
                if done:
                    rewards.append(reward_sum)
                    break

        plt.figure('train')
        plt.title('train')
        plt.plot(range(num_episodes),rewards,label='accumulate rewards')
        plt.legend()
        filepath = 'train.png'
        plt.savefig(filepath, dpi=300)
        plt.show()

    ## 测试函数
    def test(self,num_episodes=100):
        rewards = []
        for _ in range(num_episodes):
            reward_sum = 0
            state = self.env.reset()
            reward_sum = 0
            while True:
                action = self.egreedy_action(state)
                next_state,reward,end,info = self.env.step(action)
                reward_sum += reward
                state = next_state
                if end:                     
                    rewards.append(reward_sum)
                    break
        score = np.mean(np.array(rewards))
        
        plt.figure('test')
        plt.title('test: score='+str(score))
        plt.plot(range(num_episodes),rewards,label='accumulate rewards')
        plt.legend()
        filepath = 'test.png'
        plt.savefig(filepath, dpi=300)
        plt.show()
        
        return score

'''
主程序
'''
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.gamma = 1
    env.state_dim = env.observation_space.shape[0] 
    env.aspace_size = env.action_space.n 
    
    agent = PriRepDQN(env)
    agent.train()
    agent.test()