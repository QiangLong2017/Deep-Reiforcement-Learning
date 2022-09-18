# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:08:58 2022

@author: LONG QIANG
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# In[数据准备]
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
batch_size = 64
train_dataloader = DataLoader(training_data,batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

# In[构建模型]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeuNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.flatten = nn.Flatten()     # 将多维张量拉直成1维张量
                                        # 定义网络拓扑
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28,512),
                nn.ReLU(),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512,10),
                )
    def forward(self,x):                # 前向传播函数
        x=self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model =NeuNet().to(device)              # 创建一个神经网络实体
print(model) 
        
# In[损失函数和优化器]
loss_fn = nn.CrossEntropyLoss(reduction='mean')
opt = torch.optim.SGD(model.parameters(),lr=1e-3)
    
# In[训练函数]
def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train()                               # 声明以下是训练环境
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        pred = model(X)                         # 计算预测值
        loss = loss_fn(pred,y)                  # 计算损失函数
        opt.zero_grad()                         # 梯度归零
        loss.backward()                         # 误差反向传播
        opt.step()                              # 调整参数
        
        if batch%100 == 0:                      # 每隔100批次打印训练进度
            loss,current = loss.item(),batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
# In[测试函数]
def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()                                # 声明模型评估状态
    test_loss,correct = 0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device),y.to(device)
            pred= model(X)                       # 计算预测值
            test_loss += loss_fn(pred,y).item()  # 计算误差
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    correct /= size                             # 预测分类正确率
    test_loss /= num_batches                    # 测试数据凭据误差
    print('Accuracy is {}, Average loss is {}'.format(correct,test_loss))

# In[训练和测试]
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, opt)
    test(test_dataloader, model, loss_fn)
print("Done!") 

# In[访问模型参数]
model.state_dict()
model.state_dict()['linear_relu_stack.4.bias']

# In[保存模型和模型参数]
torch.save(model,'model')                               # 保存整个模型
torch.save(model.state_dict(),'model_parameter.pth')    # 保存模型参数


# In[重载模型和模型参数]
model1 = torch.load('model')    # 直接重载整个模型，包括网络拓扑和参数
model2 = NeuNet()               # 只重载参数需要先创建一个相同网络拓扑的初始模型
model2.load_state_dict(torch.load('model_parameter.pth')) # 重载模型参数

with torch.no_grad():
    for X,y in train_dataloader:
        print(model(X)[0])
        print(model1(X)[0])
        print(model2(X)[0])
        break            
            
# In[torchvision预训练模型加载]
import torchvision.models as models
model_vgg16 = models.vgg16(pretrained=True)     # 创建一个已经训练好的Vgg16网络
                                                # 保存模型参数
torch.save(model_vgg16.state_dict(),'model_vgg16_parameter') 
model_vgg16_1 = models.vgg16()                  # 创建一个未训练的vgg16网络
                                                # 加载模型参数
model_vgg16_1.load_state_dict(torch.load('model_vgg16_parameter')) 
model_vgg16.eval()                              # 评估模型
model_vgg16_1.eval()                            # 评估模型


    
    
    
    
    
    
    
