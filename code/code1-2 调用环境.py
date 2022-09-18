import gym                    	# 导入Gym包

env = gym.make('CartPole-v1') # 生成环境
state = env.reset()           	# 环境初始化   
# 进行1000次交互
for _ in range(1000):
    env.render()             	# 渲染画面
    # 从动作空间随机获取一个动作
    action = env.action_space.sample()
    # 智能体与环境进行一步交互
    state, reward, done, info = env.step(action)
    # 判断当前局是否结束
    if done:
        state = env.reset() 	# 一局结束,环境重新初始化
        
env.close()                 	# 关闭环境