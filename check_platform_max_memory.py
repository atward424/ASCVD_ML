# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:53:25 2020

@author: Andrew
"""
#%%
import platform

print(platform.architecture()[0])

import numpy as np

a = np.random.rand(100000, 100)
b = np.random.rand(100000, 1000)
c = np.random.rand(100000, 10000)
d = np.random.rand(100000, 100000)
e = np.random.rand(100000, 1000000)
f = np.random.rand(1000000, 1000000)
g = np.random.rand(1000000, 10000000)