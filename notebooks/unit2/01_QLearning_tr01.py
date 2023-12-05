import numpy as np
import random
from tqdm import tqdm

STATE_ROWS = 4
STATE_COLS = 4
ACTIONS = 4
DEAD_STATE = 7
WIN_STATE = 4

REWARD_TABLE = np.zeros(STATE_COLS*STATE_ROWS)
REWARD_TABLE[3] = 1
REWARD_TABLE[WIN_STATE] = 10
REWARD_TABLE[DEAD_STATE] = -10
REWARD_TABLE[9] = 1

Q_TABLE = np.zeros((STATE_COLS*STATE_ROWS,ACTIONS))


def randomAction(state):
    action = random.randint(0,ACTIONS-1)
    if step(state,action) == state:
        action = randomAction(state)
    return action

def greedyAction(state):
    action = np.argmax(Q_TABLE[state][:])
    return action

def epsGreedy(state:int,eps:int)->int:
    epsFactor = random.uniform(0,1)
    if epsFactor <= eps:
        action = randomAction(state)
    else:
        action = greedyAction(state)
    return action


# actino 0,1,2,3 -> up,right,down,left
def step(state:int,action:int)-> int:
    if action == 0:
        if state-STATE_COLS >= 0:
            return state - STATE_COLS
    if action == 1:
        if state%STATE_COLS != STATE_COLS-1:
            return state + 1
    if action == 2:
        if state + STATE_COLS < STATE_COLS*STATE_ROWS:
            return state + STATE_COLS
    if action == 3:
        if state%STATE_COLS != 0:
            return state - 1
    return state

def train():
    max_episodes = 10000
    decay_rate = 0.999
    learning_rate = 0.7
    gamma = 0.90
    eps = 1.0
    max_step = 15
    terminal_state = {WIN_STATE,DEAD_STATE}
    for i in tqdm(range(max_episodes)):
        state = random.randint(0,STATE_ROWS*STATE_COLS-1)
        for j in range(max_step):
            if state in terminal_state:
                break
            action = epsGreedy(state,eps)
            new_state = step(state,action)
            reward = REWARD_TABLE[new_state]

            Q_TABLE[state][action] = Q_TABLE[state][action] +\
                                        learning_rate*(reward + gamma * np.max(Q_TABLE[new_state])
                                                       - Q_TABLE[state][action])

            state = new_state

        eps *=  decay_rate
    return Q_TABLE


print(f"Reward table is \n {REWARD_TABLE.reshape(STATE_ROWS,STATE_COLS)}")
print(f"Q-table is \n{train()}")

    
