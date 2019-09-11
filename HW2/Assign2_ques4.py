#!/usr/bin/env python
# coding: utf-8

# In[1]:


def step(state, action):
    if state == A:
        return A1, 10
    if state == B:
        return B1, 5
    state = np.array(state)
    new_state = (state+action).tolist()
    x, y = new_state
    if x < 0 or x >= gridsize or y < 0 or y >= gridsize:
        reward = -1.0
        new_state = state
    else:
        reward = 0
    return new_state, reward


# In[2]:


def OptimalPolicy(state, action , v):
    if state == A:
        x,y = A1
        value = v[x,y]
        return value
    if state == B:
        x,y = B1
        value = v[x,y]
        return value
    state = np.array(state)
    next_state = (state+action).tolist()
    x, y = next_state
    if x < 0 or x >= gridsize or y < 0 or y >= gridsize:
        value = 0
        next_state = state
    else:
        value = v[x,y]
    return  value


# In[3]:


import numpy as np
gridsize = 5
A = [0,1]
B = [0,3]
A1 = [4,1]
B1 = [2,3]
discount = 0.9
v= np.zeros((gridsize, gridsize))
actions=[[0,-1],[-1,0],[0,1],[1,0]]
prob = 0.25 # probability of taking particular action
v= np.zeros((gridsize, gridsize))
while True:
    v1 = np.zeros(v.shape)
    for i in range(0,gridsize):
        for j in range(0,gridsize):
            temp = []
            for a in actions:
                (p,q),reward = step([i,j],a)
                temp.append(reward + discount*v[p,q])
            v1[i,j] = np.max(temp)
    if np.sum(np.abs(v - v1)) < 1e-4:
        break
    v = v1
    
print(v)


# In[4]:


# 0 means Left , 1 means UP , 2 means RIGHT , 3 means Down  
for i in range(0,gridsize):
    for j in range(0,gridsize):
        temp = []
        for a in actions:
            value = OptimalPolicy([i,j],a,v1)
            temp.append(value)
        print((temp))
        
        z = [p for p, t in enumerate(temp) if t == max(temp)]
        
        print("Optimal Policy for State:",(i,j),"---",z)


# In[ ]:




