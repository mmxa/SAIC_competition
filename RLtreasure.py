import numpy as np
import pandas as pd
import time

# np.random.seed(2)  # 伪随机序列

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.95  # greedy policy
ALPHA = 0.1  # learning rate
LAMBDA = 0.9  # discount
MAX_EPISODES = 15  # rounds
FRESH_TIME = 0.01


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    # print(table)
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or state_actions.all() == 0:
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions[state_actions == state_actions.max()].index
    return action_name


def get_env_feedback(s, a):
    if a == 'right':
        if s == N_STATES - 2:
            s = 'terminal'
            R = 1
        else:
            s += 1
            R = 0
    else:
        R = 0
        if s == 0:
            s = s
        else:
            s -= 1
    return s, R


def update_env(s, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if s == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(1)
        print('\r            ', end='')
    else:
        env_list[s] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        s = 0
        is_terminated = False
        update_env(s, episode, step_counter)

        while not is_terminated:
            a = choose_action(s, q_table)
            s_, R = get_env_feedback(s, a)
            q_predict = q_table.loc[s][a]
            if s_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[s_, :].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[s][a] += ALPHA * (q_target - q_predict)
            s = s_

            update_env(s, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('q_table:\n')
    print(q_table)

# build_q_table(N_STATES, ACTIONS)
