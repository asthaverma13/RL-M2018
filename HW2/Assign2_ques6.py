#!/usr/bin/env python
# coding: utf-8

# In[1]:


# to check terminal states
def terminal_state(state):
    x, y = state
    return (x == 0 and y == 0) or (x == 3 and y == 3)


# In[2]:


# Function for taking Setps in Grid world problem
def Step(state, action):
    state = np.array(state)
    new_state = (state + action).tolist()
    x, y = new_state

    if x < 0 or x >= gridsize or y < 0 or y >= gridsize:
        new_state = state.tolist()

    reward = -1
    return new_state, reward


# In[3]:


# Function for computing optimal Policy
def OptimalPolicy(state, action , v): 
    state = np.array(state)
    new_state = (state+action).tolist()
    x, y = new_state
    if x < 0 or x >= gridsize or y < 0 or y >= gridsize:
        x,y = state
        value = v[x,y]
        new_state = state
    else:
        value = v[x,y]
    return  value


# In[4]:


# 0 - Left , 1 - UP , 2 - RIGHT , 3 - Down  
def VI():     #value Iteration
    new_state_values = np.zeros((gridsize, gridsize))
    state_values = np.zeros((gridsize, gridsize))
    it = 0
    it = it+1
    while True:
       
        for i in range(gridsize):
            for j in range(gridsize):
                temp = []
                if terminal_state([i, j]):
                    continue
                value = 0
                
                for action in actions:
                    (p,q ), reward = Step([i, j], action)
                    value += prob * (reward + state_values[p, q]) # computing value function
                new_state_values[i, j] = value
                
        if np.sum(np.abs(new_state_values - state_values)) < theta: # Termination Condition
            state_values = new_state_values.copy()
            break

        state_values = new_state_values.copy()
        print('Iteration - ',it)
        print(state_values)
        
        for i in range(0,gridsize):
            for j in range(0,gridsize):
                temp = []
                for a in actions:
                    if terminal_state([i,j]):
                        value1 = 0.0
                    value1 = OptimalPolicy([i,j],a,state_values)
                    temp.append(value1)
     
                z = [p for p, t in enumerate(temp) if t == max(temp)]
       
                print("Optimal Policy for State:",(i,j),"---",z)
            it += 1

   
    return state_values


# In[5]:


# policy Improvement
def policyImprovement(old_policy,values):
    s = 0
    new_policy = np.zeros((gridsize*gridsize,len(index)))
    for i in range(gridsize):
        for j in range(gridsize):
            if terminal_state([i, j]):
                continue
            q_optimal = values[i,j]
            z = []
            for ix,a in zip(index,actions):
                #P  = old_policy[n,inx]
                (p,q), reward = Step([i, j], a)
                q_sa = (reward + values[p,q]) # action value function computation
                if q_sa > q_optimal: # update policy condition
                    z.append(ix)
           
            if len(z)>0:
                for k, l in enumerate(z):
                    #y = np.array(k)
                    new_policy[s,l] = 1/len(z)
            else:
                new_policy[s,:] = old_policy[s,:]
            s = s+1        
    return new_policy


# In[6]:


import numpy as np
gridsize = 4
terminal_state1 = [0,0]
terminal_state2 = [3,3]
actions=[[0,-1],[-1,0],[0,1],[1,0]]
prob = 0.25
theta = 1e-4
values = VI() # run value iteration function


# In[7]:


index = np.array([0,1,2,3]) 
def valueIteration(state_values,policy):
    n=0
    for i in range(gridsize):
        for j in range(gridsize):
            if terminal_state([i, j]):
                continue
            value = 0
            for ix,a in zip(index,actions):
                P = policy[n,ix]
                (p,q), reward = Step([i, j], a)
                value +=  P*(reward + state_values[p, q]) # Value function computation 
            new_state_values[i, j] = value
    return new_state_values


# In[8]:


#policy improvement

itr = 1
maxiter = 3

new_state_values = np.zeros((gridsize, gridsize))
state_values = new_state_values.copy()
old_policy = 0.25*np.ones((gridsize*gridsize,len(index))) # Random Policy
while True:
    new_state_values = valueIteration(state_values,old_policy) # run value iteration for each state, Policy Evaluation
    new_policy = policyImprovement(old_policy,new_state_values) # Policy Improvement
    # 0 -->> Left , 1 -->> UP , 2 -->> RIGHT , 3 -->> Down  
    print("New State Values")
    print(new_state_values)
  
    print("Updated Policy with policy improvement")
    print(new_policy)
    if np.sum(np.abs(new_state_values - state_values)) < theta or itr > maxiter: # Termination Condition
        state_values = new_state_values.copy()
        break
    state_values = new_state_values.copy()
    old_policy = new_policy.copy()
    itr += 1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




