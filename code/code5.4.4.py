# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:08:58 2022

@author: LONG QIANG
"""

import torch

# In[搭建计算图]
x = torch.ones((5,),requires_grad=True)     # 变量，需要计算梯度 
W = torch.ones((5,5),requires_grad=False)   # 参数，不需要计算梯度
b = torch.ones((5,))                        # 参数，默认不需要计算梯度
Q = torch.matmul(torch.matmul(x,W),x)       # 二次项，中间结果
L = torch.matmul(b,x)                       # 一次项，中间结果
C = torch.matmul(b,b)                       # 常数项
y = Q+L+C                                   # 前向传播，建立计算图
                                            # 查看需要求梯度的量
print(x.requires_grad,Q.requires_grad,C.requires_grad,y.requires_grad)
print(y.grad_fn) # 对最终结果y的梯度函数
print(Q.grad_fn) # 对中间结果Q的梯度函数
print(C.grad_fn) # 对中间结果C的梯度函数
print(x.grad_fn) # 对中间结果L的梯度函数

# In[梯度计算和查看]
y.backward(retain_graph=True)       # 自动梯度计算
print(x.grad)                       # 查看y对于x的梯度
print(b.grad)                       # 查看y对于b的梯度
#print(Q.grad)                      # 查看y对于Q的梯度

# In[梯度累加]
y.backward(retain_graph=True) 
print(x.grad)

# In[梯度归零]
x.grad.zero_()
print(x.grad)
y.backward(retain_graph=True)
print(x.grad)

# In[关闭自动梯度计算]
y = torch.matmul(b,x)
print(y.requires_grad)

with torch.no_grad():
    y1 = torch.matmul(b,x)
print(y1.requires_grad)

y2 = y.detach()
print(y2.requires_grad)


