# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 17:20:25 2021

@author: LONG QIANG

用蒙特卡罗法计算pi值
"""

import random
import math

random.seed(0)  # 初始化随机数种子
n = 1000000     # 投点数
k = 0           # 落在圆内的点计数器
for _ in range(n):
    x1,x2 = random.random(),random.random()     # 随机生成一个点
    if math.sqrt(x1*x1+x2*x2)<=1:               # 如果点落在圆内
        k += 1 
print("pi = ",4*k/n)

