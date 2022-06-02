from collections import defaultdict
from random import uniform
from time import sleep

import gym

env = gym.make("FrozenLake-v0")
actions = list(range(env.action_space.n))
greedy_policy = {
    state: env.action_space.sample()
    for state in range(env.observation_space.n)
}
q_values = defaultdict(float)
EPSILON = 0.5
ALPHA = 0.05
GAMMA = 1
STEP_SLEEP = 0
EPISODE_SLEEP = 0
SUCCESS_SLEEP = 0
success_count = 0


def policy(state):
    if uniform(0, 1) < EPSILON:
        return env.action_space.sample()
    return greedy_policy[state]


while True:
    state = env.reset()
    action = policy(state)
    env.render()
    while True:
        next_state, reward, done, _ = env.step(action)
        next_action = policy(next_state)
        env.render()
        q_values[state, action] += \
            ALPHA * (reward + GAMMA * q_values[next_state, next_action]
                     - q_values[state, action])
        greedy_policy[state] = max(actions, key=lambda x: q_values[state, x])
        if done:
            if reward == 1:
                print("SUCCESS!")
                success_count += 1
                if success_count == 1000:
                    EPSILON = 0
                    STEP_SLEEP = 0.2
                    EPISODE_SLEEP = 1
                    SUCCESS_SLEEP = 3
                sleep(SUCCESS_SLEEP)
            sleep(EPISODE_SLEEP)
            break
        action = next_action
        state = next_state
        sleep(STEP_SLEEP)
