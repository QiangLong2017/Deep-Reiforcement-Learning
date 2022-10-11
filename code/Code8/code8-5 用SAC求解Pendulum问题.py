'''
导入包
'''
import gym
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

'''
经验回放池
'''
class ReplayBuffer():
    def __init__(self,mem_size,input_shape,n_actions):
        self.mem_size = mem_size
        self.mem_cntr = 0                     # 经验回放池计数器
        self.state_memory = np.zeros((self.mem_size,input_shape))
        self.new_state_memory = np.zeros((self.mem_size,input_shape))
        self.action_memory = np.zeros((self.mem_size,n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool)
    
    ## 往经验回放池中添加数据
    def add(self,state,action,reward,state_new,done):  
        index = self.mem_cntr%self.mem_size  # 溢出，则替换最早数据
        self.state_memory[index] = state
        self.new_state_memory[index] = state_new
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
     ## 从经验回放池中采样数据
    def sample(self, batch_size):  
        max_mem = min(self.mem_cntr,self.mem_size)
        batch = np.random.choice(max_mem,batch_size)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_new = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states,actions,rewards,states_new,dones
    
'''
定义Actor网络，即策略网络
'''
class Actor(nn.Module):    
    def __init__(self,lr,input_dims,max_action,n_actions=2, 
                 fc1_dims=256,fc2_dims=256):
        super(Actor, self).__init__()     
        self.max_action = max_action
        self.policy_noise = 1e-6            # 添加策略噪声
        
        # 定义策略网络各层
        self.fc1 = nn.Linear(input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.fc_mu = nn.Linear(fc2_dims,n_actions)
        self.fc_sigma = nn.Linear(fc2_dims,n_actions)
        
        # 定义优化器
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    ## 前向传播函数
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        sigma = torch.clamp(sigma,min=self.policy_noise,max=1)
        
        return mu,sigma
    
    ## 动作采样与概率计算函数
    def sample_normal(self,state,reparameterize=True):
        mu,sigma = self.forward(state)
        probabilities = torch.distributions.Normal(mu,sigma)
        if reparameterize:   
            action_ = probabilities.rsample()   # 重参数处理
        else:
            action_ = probabilities.sample()
    
        action = torch.tanh(action_)*torch.tensor(
                self.max_action).to(self.device)
        log_probs = probabilities.log_prob(action_)
        log_probs -= torch.log(torch.tensor(self.max_action)*(
                1-torch.tanh(action_).pow(2))+self.policy_noise)
        log_probs = log_probs.sum(1,keepdim=True)

        return action,log_probs

'''
定义Critic网络，即价值网络
'''       
class Critic(nn.Module):          
    def __init__(self,lr,input_dims,n_actions,fc1_dims=256,fc2_dims=256):
        super(Critic, self).__init__()
        
        # 定义价值网络各层
        self.fc1 = nn.Linear(input_dims+n_actions,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.fc_q = nn.Linear(fc2_dims,1)
        
        # 定义优化器
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    ## 前向传播函数    
    def forward(self,state,action):
        action_value = F.relu(self.fc1(torch.cat([state,action],dim=1)))
        action_value = F.relu(self.fc2(action_value))
        q = self.fc_q(action_value)

        return q

'''
定义SAC类
'''
class SAC():    
    def __init__(self,env,state_dim,action_dim,
                lr_actor=3e-4,lr_critic=3e-4,tau=0.005,alpha=0.5,
                gamma=0.99,buffer_size=1e4,batch_size=128,
                fc1_dims=256, fc2_dims=256):   
        self.env = env                      # 环境模型
        self.gamma = gamma                  # 折扣率参数
        self.tau = tau                      # 软更新参数
        self.target_entropy=-action_dim     # 期望熵值
        self.batch_size = batch_size        # 批大小
        # 创建经验回放池
        self.buffer = ReplayBuffer(mem_size=int(buffer_size),
                      input_shape=state_dim,n_actions=action_dim)

        # 温度参数及优化器定义
        self.alpha = torch.tensor((alpha),
                     dtype=torch.float32,requires_grad=True) 
        self.alpha_optimizer = torch.optim.Adam((self.alpha,),lr=1e-4)
        
        # 创建策略网络
        self.actor = Actor(lr=lr_actor,input_dims=state_dim,
                     max_action=self.env.action_space.high,
                     n_actions=action_dim)
        
        # 创建价值网络
        self.critic_1 = Critic(lr=lr_critic,
                        input_dims=state_dim,n_actions=action_dim)
        self.critic_2 = Critic(lr=lr_critic,
                        input_dims=state_dim,n_actions=action_dim)
        self.target_critic_1 = Critic(lr=lr_critic,
                        input_dims=state_dim,n_actions=action_dim)
        self.target_critic_2 = Critic(lr=lr_critic,
                        input_dims=state_dim,n_actions=action_dim)

    ## 输出随机策略动作
    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state,reparameterize=False)
        return actions.cpu().detach().numpy()[0]    
         
    ## 训练函数
    def train(self,NUM_EPISODES=100):
        # 外层循环直到最大迭代轮次
        rewards_history = []       
        for i in range(NUM_EPISODES):
            observation = self.env.reset()
            done = False
            score = 0
            # 内层循环，一次经历完整的模拟
            while not done:
                action = self.choose_action(observation)
                observation_new,reward,done,_ = self.env.step(action)
                score += reward
                self.buffer.add(observation,
                                action,reward,observation_new,done)

                # 判断训练数据量是否大于BATCH_SIZE
                if self.batch_size < self.buffer.mem_cntr:
                    # 抽样并转化数据
                    states,actions,rewards,states_new,dones = self.buffer.sample(self.batch_size)
                    rewards = torch.tensor(rewards,dtype=torch.float).to(self.actor.device)
                    states = torch.tensor(states,dtype=torch.float).to(self.actor.device)
                    states_new = torch.tensor(states_new,dtype=torch.float).to(self.actor.device)
                    actions = torch.tensor(actions,dtype=torch.float).to(self.actor.device)
                    
                    # 训练价值网络
                    actions_sample,log_probs = self.actor.sample_normal(
                            states_new,reparameterize=False)
                    q1_new_policy = self.target_critic_1.forward(
                            states_new,actions_sample)
                    q2_new_policy = self.target_critic_2.forward(
                            states_new,actions_sample)
                    critic_value = torch.min(
                            q1_new_policy,q2_new_policy).view(-1)
                    value_target = critic_value-self.alpha*log_probs
                    q_hat = rewards + self.gamma*value_target
                    q1_old_policy = self.critic_1.forward(
                            states,actions).view(-1)
                    q2_old_policy = self.critic_2.forward(
                            states,actions).view(-1)
                    critic_1_loss = 0.5 * F.mse_loss(q1_old_policy,q_hat)
                    critic_2_loss = 0.5 * F.mse_loss(q2_old_policy,q_hat)
                    critic_loss = critic_1_loss+critic_2_loss
                    self.critic_1.optimizer.zero_grad()     
                    self.critic_2.optimizer.zero_grad()     
                    critic_loss.backward()                  
                    self.critic_1.optimizer.step()        
                    self.critic_2.optimizer.step()         
                    
                     # 训练策略网络
                    actions_sample,log_probs = self.actor.sample_normal(
                            states,reparameterize=True)
                    log_probs = log_probs.view(-1)
                    q1_new_policy = self.critic_1.forward(
                            states,actions_sample)
                    q2_new_policy = self.critic_2.forward(
                            states,actions_sample)
                    critic_value = torch.min(
                            q1_new_policy,q2_new_policy).view(-1)           
                    actor_loss = self.alpha*log_probs-critic_value
                    actor_loss = torch.mean(actor_loss)
                    self.actor.optimizer.zero_grad()    
                    actor_loss.backward()               
                    self.actor.optimizer.step()         
                    
                    ## 温度参数自适应调节                   
                    obj_alpha = (self.alpha*(
                            -log_probs-self.target_entropy).detach()).mean()
                    self.alpha_optimizer.zero_grad()      
                    obj_alpha.backward()                    
                    self.alpha_optimizer.step()           
                    
                    # 目标网络参数软更新
                    for param,t_param in zip(self.critic_1.parameters(),
                            self.target_critic_1.parameters()):
                        t_param.data.copy_(
                            self.tau*param.data+(1-self.tau)*t_param.data)
                    for param,t_param in zip(self.critic_2.parameters(),
                            self.target_critic_2.parameters()):
                        t_param.data.copy_(
                            self.tau*param.data+(1-self.tau)*t_param.data)  
                                    
                observation = observation_new
                  
            rewards_history.append(score)          
            print('episode: {:^3d} | score: {:^10.2f} |'.format(i, score))
             
    
        # 图示训练过程
        plt.figure('train')
        plt.title('train')
        window = 10
        smooth_r = [np.mean(rewards_history[i-window:i+1]) if i > window 
                        else np.mean(rewards_history[:i+1]) 
                        for i in range(len(rewards_history))]
        plt.plot(range(NUM_EPISODES),rewards_history,
                 label='accumulate rewards')
        plt.plot(smooth_r,label='smoothed accumulate rewards')
        plt.legend()
        filepath = 'train.png'
        plt.savefig(filepath, dpi=300)
        plt.show()    
            
    
    ## 测试函数
    def test(self,test_episodes=100):
        # 循环直到最大测试轮数
        rewards = []                        # 每一轮次的累积奖励
        for _ in range(test_episodes):
            reward_sum = 0
            state = self.env.reset()        # 环境状态初始化
            
            # 循环直到到达终止状态
            reward_sum = 0                  # 当前轮次的累积奖励
            while True:
                action = self.choose_action(state)
                next_state,reward,done,info = self.env.step(action)
                reward_sum += reward
                state = next_state 
                
                # 检查是否到达终止状态
                if done:                     
                    rewards.append(reward_sum)
                    break
        
        score = np.mean(np.array(rewards))  # 计算测试得分
        
        # 图示测试结果
        plt.figure('test')
        plt.title('test: score='+str(score))
        plt.plot(range(test_episodes),rewards,label='accumulate rewards')
        plt.legend()
        filepath = 'test.png'
        plt.savefig(filepath, dpi=300)
        plt.show()
        
        return score                        # 返回测试得分
    
    ## 一个回合的仿真    
    def sim(self,EP_LEN=201,do_render=False):
        s = env.reset()
        buffer_s, buffer_a, buffer_r,buffer_sn = [], [], [],[]
        for t in range(EP_LEN):            # 一个回合仿真
            if do_render==True:
                self.env.render()          # 动画显示一帧
            else:
                pass
    
            a = self.choose_action(s)       
            s_, r, done, info = self.env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)
            buffer_sn.append(s_)
            s = s_
        batch = buffer_s,buffer_a,buffer_r,buffer_sn
        env.close()
        return batch        

'''
定义SAC类
'''       
if __name__ == '__main__':
    # 导入环境
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 创建一个SAC类智能体
    agent = SAC(env,state_dim,action_dim) 
    agent.train()           # 训练
    agent.test()            # 测试
    batch = agent.sim(EP_LEN=200,do_render=True)   # 仿真