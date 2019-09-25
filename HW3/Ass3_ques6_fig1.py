#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


initial_value = np.zeros(7)
initial_value[1:6] = 0.5
initial_value[6] = 1
true_value = np.zeros(7)
true_value[1:6] = np.arange(1, 6) / 6.0
true_value[6] = 1
left = 0
right = 1

episodes = [0, 1, 10, 100]
current_values = np.copy(initial_value)
plt.figure(figsize=(30, 30))
for i in range(episodes[-1] + 1):
    if i in episodes:
        plt.plot(current_values, label=str(i) + ' episodes')
    values = current_values
    batch = False
    alpha = 0.1
    state = 3
    history = [state]
    rewards = [0]
    while True:
        old_state = state
        if np.random.binomial(1, 0.5) == left:
            state -= 1
        else:
            state += 1
        reward = 0
        history.append(state)
        if not batch:
            values[old_state] += alpha * (reward + values[state] - values[old_state])
        if state == 6 or state == 0:
            break
        rewards.append(reward)
plt.plot(true_value, label='True Values')
plt.xlabel('State',fontsize=32)
plt.ylabel('Estimated value',fontsize=32)
plt.legend(prop={'size': 32})
plt.savefig('Example_6_2_EstimatedValue.png')
plt.show()

