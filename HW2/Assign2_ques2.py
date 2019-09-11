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
    if x < 0 or x >= gridsize or y < 0 or y >= gridsize: # condition for checking edges 
        reward = -1.0
        new_state = state
    else:
        reward = 0
    return new_state, reward


# In[2]:


import numpy as np
gridsize = 5
A = [0,1]
B = [0,3]
A1 = [4,1]
B1 = [2,3]
discount = 0.9
s = [[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24]]
s = np.array(s)
v= np.zeros((gridsize, gridsize))
actions=[[0,-1],[-1,0],[0,1],[1,0]]
prob = 0.25 # probability of taking particular action
while True:
    v1 = np.zeros(v.shape)
    for i in range(0,gridsize):
        for j in range(0,gridsize):
            for a in actions:
                (p,q),reward = step([i,j],a)
                v1[i,j] += prob*(reward + discount*v[p,q]) # compute Equation for each state. 
    if np.sum(np.abs(v - v1)) < 1e-4:
        break
    v = v1
    
print(v)

