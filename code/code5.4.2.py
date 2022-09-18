# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:08:58 2022

@author: LONG QIANG
"""

import torch
import numpy as np

# In[张量的维度]
t = torch.Tensor(np.arange(24)).reshape((2,3,4))
print(t)

print(t.sum(dim=0)) # print(t.sum(axis=0))
print(t.sum(dim=1))
print(t.sum(dim=2))

# In[张量的维度重组]
t = torch.Tensor(np.arange(24)).reshape((2,3,4))
t1 = t.view(2,2,6)
t2 = t.reshape(8,-1)
print(t)
print(t1)
print(t2)

# In[维的添加和压缩]
t = torch.Tensor(np.arange(6)).reshape((2,3))
t1 = t.unsqueeze(dim=0)
t2 = t1.unsqueeze(dim=3)
t3 = t1.squeeze()
t4 = t2.squeeze()
print(t,t.shape)
print(t1,t1.shape)
print(t2,t2.shape)
print(t3,t3.shape)
print(t4,t4.shape)

t5 = t2.squeeze(dim=3)
print(t5,t5.shape)

# In[张量转置]
t = torch.Tensor(np.arange(6)).reshape((2,3))
t_t = t.t()
print(t_t,t_t.shape)

t = torch.Tensor(np.arange(24)).reshape((2,3,4))
t_trans = t.transpose(0,1) # 第0和1维转置
print(t,t.shape)
print(t_trans,t_trans.shape)

t_perm = t.permute((1,0,2)) # 将维度按照1,0,2方式转置，即（3,2,4）
print(t_perm,t_perm.shape)

# In[张量的广播]
t1 = torch.Tensor(np.arange(6)).reshape((2,3))
t2 = torch.ones((3,))
t3 = t1+t2
print(t3)

# In[张量的拼接和拆分]
t1 = torch.Tensor(np.arange(6)).reshape((3,2))
t2 = torch.cat((t1,t1),dim=0)
print(t2)

t3 = torch.split(t1,[1,2])
print(t3)

t4 = torch.hsplit(t1,2)
print(t4)

t5 = torch.hstack((t1,t1))
print(t5)

t6 = torch.vsplit(t1,(1,2))
print(t6)

t7 = torch.vstack((t1,t1))
print(t7)
