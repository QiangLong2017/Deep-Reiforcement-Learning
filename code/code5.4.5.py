# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:08:58 2022

@author: LONG QIANG
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# In[下载数据集]
training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
        )

test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
        )

# In[查看数据结构]
print(training_data.data.shape,training_data.targets.shape)
print(test_data.data.shape,test_data.data.shape)
print(training_data.data.dtype,training_data.targets.dtype)
print(training_data.classes)

# In[作图]
labels = training_data.classes
figure = plt.figure(figsize=(8,8))
cols,rows = 3,3
for i in range(1,cols*rows+1):
    sample_idx = torch.randint(len(training_data),size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels[label])
    plt.axis('off')
    plt.imshow(img.squeeze(),cmap='gray')
plt.show()

# In[训练数据加载]
batch_size = 32
training_data_loader = DataLoader(training_data,batch_size,shuffle=True)
test_data_loader = DataLoader(test_data,batch_size,shuffle=True)

for X,y in training_data_loader:
    print('Shape of X is ',X.shape)
    print('Shape of y is ',y.shape)
    break

    
    
    
    
    
    
    
    
    
    
    
    
