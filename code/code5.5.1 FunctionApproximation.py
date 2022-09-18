# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 20:21:11 2022

@author: 龙强
"""

# In[导入包]
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# In[超参数]
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 40

# In[原函数]
def fun(x):
    return x*x+3*x+4

x = np.linspace(-np.pi,np.pi,100)
y = fun(x)

# In[创建神经网络]
class NeuNet(nn.Module):
    def __init__(self,in_size,out_size):
        nn.Module.__init__(self)
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
                     nn.Linear(in_size,20),
                     nn.ReLU(),
                     nn.Linear(20,40),
                     nn.ReLU(),
                     nn.Linear(40,20),
                     nn.ReLU(),
                     nn.Linear(20,out_size),
                     )
    def forward(self,x):
        self.flatten(x)
        return self.layers(x)

model = NeuNet(1,1)  
  
# In[损失函数和优化器]
loss = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(),lr=LR)

# In[训练函数]
def train(model,loss,opt):
    x_batch = -np.pi+2*np.pi*np.random.rand(BATCH_SIZE,1)   # 训练输入
    y_tar_batch = fun(x_batch)                              # 目标输出
    
    x_batch = torch.from_numpy(x_batch).float()             # 数据格式转换
    y_tar_batch = torch.from_numpy(y_tar_batch).float()     # 数据格式转换
    y_pre_batch = model(x_batch).float()                    # 预测输入
            
    loss_fn = loss(y_tar_batch,y_pre_batch)                 # 损失函数
        
    model.train()                                           # 声明训练
    opt.zero_grad()                                         # 梯度归零
    loss_fn.backward()                                      # 误差反向传播
    opt.step()                                              # 参数调整
        
# In[测试函数]
def test(model):
    model.eval()
    with torch.no_grad():
        y_pre_test = model(torch.from_numpy(x).float().unsqueeze(dim=1))
        loss_value = loss(torch.from_numpy(y).float(),y_pre_test.float())
    print('loss_fn = ',loss_value)
    
    return loss_value

# In[训练和测试]
Loss = []
for i in range(EPOCHS):
    print('EPOCH {}--------------'.format(i))
    train(model,loss,opt)
    loss_value = test(model)
    Loss.append(loss_value)
print('DONE')

# In[作图比较]
with torch.no_grad():
    y_test = model(torch.from_numpy(x).float().unsqueeze(dim=1))
    y_test = y_test.squeeze().numpy()

plt.figure(1)
plt.plot(Loss)
plt.xlabel('EPOCHS')
plt.ylabel('Loss')
plt.title('Loss via EPOCHS')
plt.savefig('loss.jpg')

plt.figure(2)
plt.plot(x,y,label='real')
plt.plot(x,y_test,label='approximated')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Real vs approximated graph')
plt.legend()
plt.savefig('graph.jpg')
plt.show()

               
        
        
        
        
        
    
