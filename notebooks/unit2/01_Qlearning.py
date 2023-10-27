import numpy as np
import random
import tqdm

STATE_ROWS = 4
STATE_COLS = 5
ACTIONS = 4
TERMINATE_STATE = 12

#what you need

# based on formula

# gamma - the discount factor
# immediate reward
# learning rate

# q table will be -> # of states * # of actions table
# the reward will have the smae dimension of states
Q_TABLE = np.zeros((STATE_ROWS*STATE_COLS,ACTIONS), dtype='f')
REWARD_TABLE = np.array([0 for i in range(STATE_ROWS*STATE_COLS)])
# Set some reward
REWARD_TABLE[10] = 10
REWARD_TABLE[5] = 1
REWARD_TABLE[6] = 1
REWARD_TABLE[11] = 1
REWARD_TABLE[TERMINATE_STATE] = -10

print(REWARD_TABLE.reshape((STATE_ROWS,STATE_COLS)))

num_episode = 10000
learning_rate = 0.7
decay_rate = 0.01
max_steps = 30
discount_factor = 0.8

def mapStateAction(state:int,action:int)->(int,int): 
    if action == 0:
        if state-STATE_COLS >= 0:
            state = state-STATE_COLS
            return action,state
    if action == 1:
        if (state%STATE_COLS) != (STATE_COLS -1):
            state = state + 1
            return action,state
    if action == 2:
        if state+STATE_COLS < STATE_COLS*STATE_ROWS-1:
            state = state + STATE_COLS
            return action,state
    if action == 3:
        if (state%STATE_COLS) != 0:
            state = state -1
            return action,state
    return None


def randomPolicy(cur_state:int)->(int,int):
    # action up:0,right:1,down:2,left:3
    action = random.randint(0,ACTIONS-1)
    res = mapStateAction(cur_state,action)
    if res == None:
        res = randomPolicy(cur_state)
    action,state = res
    return action,state

def greedyPolicy(cur_state)->(int,int):
    action = np.argmax(Q_TABLE[cur_state][:])
    res = mapStateAction(cur_state,action)
    if res == None:
        res = randomPolicy(cur_state)
    action, state = res
    return action,state

def epsilonGreedy(cur_state:int,eps:float)->(int,int):
    eps_factor = np.random.uniform(0,1)
    if eps_factor <= eps:
        action,state = randomPolicy(cur_state)
    else:
        action,state = greedyPolicy(cur_state)
    return action, state


def train():
    eps = 1
    for i in range(num_episode):
        cur_state = random.randint(0,STATE_COLS*STATE_ROWS-1)
        t = 0
        while True:
            if cur_state == TERMINATE_STATE:
                break
            if t == max_steps:
                break
            action, next_state = epsilonGreedy(cur_state,eps)
            reward = REWARD_TABLE[next_state]
            #print(f"current state = {cur_state} \n next state = {next_state}, \n action = {action} \n Reward = {reward} \n at t={t}")
            Q_TABLE[cur_state][action] = Q_TABLE[cur_state][action] + learning_rate*(
                reward + 
                discount_factor*np.max(Q_TABLE[next_state])
                - Q_TABLE[cur_state][action]
            )
            t +=1
            cur_state = next_state
        eps *= decay_rate
        print(Q_TABLE)
        

train()