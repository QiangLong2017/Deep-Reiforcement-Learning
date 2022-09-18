# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:08:58 2022

@author: LONG QIANG
"""

import torch
import numpy as np

# In[Tensor默认数据类型] 
a = torch.rand((3,))
print(a.dtype)

b = torch.randint(1,10,(3,))
print(b.dtype)

# In[Tensor指定数据类型]
a = torch.rand((3,),dtype=torch.float64)
print(a.dtype)

b = torch.randint(1,10,(3,),dtype=torch.int32)
print(b.dtype)

# In[Tensor数据类型转换]
a = torch.rand((3,))
print(a,a.dtype)

b = a.int()
print(b,b.dtype)

c = b.float()
print(c,c.dtype)


# In[ndarray转Tensor]
a = np.array([1.,2.,3.])
tensor_a1 = torch.from_numpy(a)
tensor_a2 = torch.FloatTensor(a)
print(a,a.dtype)
print(tensor_a1,tensor_a1.dtype)
print(tensor_a2,tensor_a2.dtype)

# In[Tensor转ndarray]
t = torch.rand((3,))
array_t1 = t.numpy()
array_t2 = np.array(t)
print(t,t.dtype)
print(array_t1,array_t1.dtype)
print(array_t2,array_t2.dtype)