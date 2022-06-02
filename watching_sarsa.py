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
success_count = 0
RENDER = False

EPSILON = 0.5
ALPHA = 0.05
GAMMA = 1
STEP_SLEEP = 0
EPISODE_SLEEP = 0
SUCCESS_SLEEP = 0
SLOWDOWN_THRESHOLD = 100


def policy(state):
    if uniform(0, 1) < EPSILON:
        return env.action_space.sample()
    return greedy_policy[state]


while True:
    state = env.reset()
    action = policy(state)
    if RENDER:
        env.render()
    while True:
        next_state, reward, done, _ = env.step(action)
        next_action = policy(next_state)
        if RENDER:
            env.render()
        q_values[state, action] += \
            ALPHA * (reward + GAMMA * q_values[next_state, next_action]
                     - q_values[state, action])
        greedy_policy[state] = max(actions, key=lambda x: q_values[state, x])
        if done:
            if reward == 1:
                success_count += 1
                if RENDER:
                    print("SUCCESS!")
                else:
                    print(f"Successes: {success_count: 5}", end="\r")
                if success_count == SLOWDOWN_THRESHOLD:
                    RENDER = True
                    EPSILON = 0
                    STEP_SLEEP = 0.2
                    EPISODE_SLEEP = 1
                    SUCCESS_SLEEP = 3
                    print(f"Successes: {success_count: 5}")
                sleep(SUCCESS_SLEEP)
            sleep(EPISODE_SLEEP)
            break
        action = next_action
        state = next_state
        sleep(STEP_SLEEP)
