#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random
random.seed(42)


# In[2]:


# Target Policy for player
def target_policy_player(pl_ace_flag, pl_sum, dl_card):
    return pl_policy[pl_sum]
# Random behaviour Policy for player
def behavior_policy_player(pl_ace_flag, pl_sum, dl_card):
    if np.random.binomial(1, 0.5) == 1:
        return stand
    return hit
# draw card from the Deck.
def drawCard():
    card = np.random.randint(1, 13)
    if card > 10:
        card = 10    
    return card
# Game 
def blackjack(policy_player, initial_state=None, initial_action=None):
    pl_sum = 0
    pl_history = []
    pl_ace_flag = False # True --> player use ace as 11 otherwise 1
    dl_card1 = 0
    dl_card2 = 0
    dl_ace_flag = False
    if initial_state is None:
        n_ace = 0
        while pl_sum < 12:
            card = drawCard()
            if card == 1:
                n_ace += 1
                card = 11
                pl_ace_flag = True
            pl_sum += card
        if pl_sum > 21:
            pl_sum -= 10
            if n_ace == 1:
                pl_ace_flag = False
        #get dealer cards
        dl_card1 = drawCard()
        dl_card2 = drawCard()
    else:
        pl_ace_flag, pl_sum, dl_card1 = initial_state
        dl_card2 = drawCard()

    state = [pl_ace_flag, pl_sum, dl_card1]
        
    
    dl_sum = 0
    if dl_card1 == 1 and dl_card2 != 1:
        dl_sum += 11 + dl_card2
        dl_ace_flag = True
    elif dl_card1 != 1 and dl_card2 == 1:
        dl_sum += dl_card1 + 11
        dl_ace_flag = True
    elif dl_card1 == 1 and dl_card2 == 1:
        dl_sum += 1 + 11
        dl_ace_flag = True
    else:
        dl_sum += dl_card1 + dl_card2
   
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            action = policy_player(pl_ace_flag, pl_sum, dl_card1)
        pl_history.append([(pl_ace_flag, pl_sum, dl_card1), action])

        if action == stand:
            break
         # HIT   
        pl_sum += drawCard()
        # BUSTS
        if pl_sum > 21:
            #  Avoid busting
            if pl_ace_flag == True:
                pl_sum -= 10
                pl_ace_flag = False
            else:
                # Player loses
                return state, -1, pl_history
    # dealer's turn
    while True:
        # get action based on current sum
        action = dl_policy[dl_sum]
        if action == stand:
            break
        # HIT
        new_card = drawCard()
        if new_card == 1 and dl_sum + 11 < 21:
            dl_sum += 11
            dl_ace_flag = True
        else:
            dl_sum += new_card
        # BUST
        if dl_sum > 21:
            if dl_ace_flag == True:
            #  Avoid busting and continue
                dl_sum -= 10
                dl_ace_flag = False
            else:
            # otherwise dealer loses
                return state, 1, pl_history
     # compare the sum between player and dealer
    if pl_sum > dl_sum:
        return state, 1, pl_history
    elif pl_sum == dl_sum:
        return state, 0, pl_history
    else:
        return state, -1, pl_history
   


# In[3]:



def MS_ES(episodes):
   state_action_values = np.zeros((10, 10, 2, 2))
   state_action_pair_count = np.ones((10, 10, 2, 2))
   def behavior_policy(usable_ace, player_sum, dealer_card):
       usable_ace = int(usable_ace)
       player_sum -= 12
       dealer_card -= 1
       #action = int(action)
       values_ = state_action_values[player_sum, dealer_card, usable_ace, :] /                  state_action_pair_count[player_sum, dealer_card, usable_ace, :]
       return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
   for episode in range(episodes):
       # Random State
       initial_state = [bool(np.random.choice([0, 1])),np.random.choice(range(12, 22)),np.random.choice(range(1, 11))]
       initial_action = np.random.choice(actions) # Random Action
       current_policy = behavior_policy if episode else target_policy_player # Get the current policy
       _, reward, trajectory = blackjack(current_policy, initial_state, initial_action)
       for (usable_ace, player_sum, dealer_card), action in trajectory:
           usable_ace = int(usable_ace)
           player_sum -= 12
           dealer_card -= 1
           action = int(action)
           state_action_values[player_sum, dealer_card, usable_ace, action] += reward #Update state action Values
           state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1

   return state_action_values / state_action_pair_count


# In[4]:


# Define Actions
hit= 0
stand = 1
actions = [hit,stand]
# The Policy that Sticks if the player's sum is 20 or 21 and hit othewise
pl_policy = np.zeros(22)
for i in range(12, 20):
    pl_policy[i] = hit
pl_policy[20] = stand
pl_policy[21] = stand
# Dealer hits and sticks according to a fixed stratergy without choice: 
dl_policy = np.zeros(22)
for i in range(12, 17): 
    dl_policy[i] = hit
for i in range(17, 22):
    dl_policy[i] = stand # sticks on any sum of 17 or greater
state_action_values = MS_ES(500000) # Run Monte carlo for 500000 Episodes
state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)
# get the optimal policy
action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)
images1 = [state_value_usable_ace,state_value_no_usable_ace]
titles1 = ['Optimal value with usable Ace','Optimal value without usable Ace']
index = [1,2]
fig = plt.figure(figsize=plt.figaspect(1))
for image, title, a in zip(images1, titles1, index):
    ax = fig.add_subplot(2, 1, a, projection='3d')
    surf = ax.plot_surface(range(1,11), range(12,22), image, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.ylabel('player sum', fontsize=10)
    plt.xlabel('dealer showing', fontsize=10)
    plt.title(title, fontsize=10)

plt.savefig('5_2.png')
plt.close()

images2 = [action_usable_ace, action_no_usable_ace]
titles2 = ['Optimal policy with usable Ace', 'Optimal policy without usable Ace']
_, axes = plt.subplots(2, 1, figsize=(40, 30))
plt.subplots_adjust(wspace=0.1, hspace=0.2)
axes = axes.flatten()

for image, title, axis in zip(images2, titles2, axes):
    fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                      yticklabels=list(reversed(range(12, 22))))
    fig.set_ylabel('player sum', fontsize=30)
    fig.set_xlabel('dealer showing', fontsize=30)
    fig.set_title(title, fontsize=30)

plt.savefig('figure_5_2.png')
plt.close()


# In[ ]:




