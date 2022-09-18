# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:08:58 2022

@author: LONG QIANG
"""

import torch
import torch.nn as nn

# In[线性层]
B = 10                              # batch-size
linear_layer = nn.Linear(20,30)     # 创建线性层实例
x = torch.randn(B,20)               # 输入批量数据，单个输入维度为20
y = linear_layer(x)                 # 线性层映射，输出y
print(y.size(),y.dtype)             # 查看输出的尺寸

dir(linear_layer)           # 查询linear_layer的所有属性和功能函数名
print(linear_layer.weight)  # 查询权重
print(linear_layer.bias)    # 查询偏置

# In[ReLU激活函数]
B = 10                  # batch-size
relu = nn.ReLU()        # 创建ReLU函数实体
x = torch.randn(B,20)   # 输入批量数据，单个输入维度为20
y = relu(x)             # ReLU函数映射
print(x.shape,y.shape)  # 输入输出尺寸一样

# In[MSELoss函数]
B = 3                                   # batch-size
loss = nn.MSELoss()                     # 创建MSELoss函数，reduction='mean'
loss_sum = nn.MSELoss(reduction='sum')  # 创建MSELoss函数，reduction='sum'
loss_none = nn.MSELoss(reduction='none')# 创建MSELoss函数，reduction='none'
y_hat = torch.randn((B,5))              # 预测输出
y_tar = torch.randn((B,5))              # 目标输出
out = loss(y_hat,y_tar)                 # 损失函数值
out_sum = loss_sum(y_hat,y_tar)
out_none = loss_none(y_hat,y_tar)
print(out,out.shape)
print(out_sum,out_sum.shape)
print(out_none,out_none.shape)
