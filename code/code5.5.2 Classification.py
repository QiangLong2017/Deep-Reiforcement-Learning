# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 18:13:25 2022

@author: 龙强
"""

# In[导入包]
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# In[超参数]
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 5

# In[数据下载]
training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        )

test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        )

# In[数据加载]
train_dataloader = DataLoader(training_data,batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data,batch_size=BATCH_SIZE)

# In[创建网络]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeuralNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.flatten = nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
                nn.Linear(28*28,512),
                nn.ReLU(),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512,10)
                )
        
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

# In[损失函数和优化器]
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=LR)

# In[训练函数]
def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred,y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch%100 ==0:
            loss, current = loss.item(),batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# In[测试函数]    
def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss,correct = 0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device),y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

# In[模型训练和测试]
for t in range(EPOCHS):
    print(f"------Epoch {t+1}------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
print("Done!")

# In[训练结果展示]
classes = training_data.classes
model.eval()
x,y = test_data[0][0],test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted,actual = classes[pred[0].argmax(0)],classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
