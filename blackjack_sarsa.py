from collections import defaultdict
from random import uniform
from time import sleep
import os

import gym

env = gym.make("Blackjack-v0")
actions = list(range(env.action_space.n))
greedy_policy = defaultdict(int)
q_values = defaultdict(float)
success_count = 0
RENDER = False

EPSILON = 0.5
ALPHA = 0.05
GAMMA = 1
SLOWDOWN_THRESHOLD = 10000


def policy(state):
    if uniform(0, 1) < EPSILON:
        return env.action_space.sample()
    return greedy_policy[state]


while True:
    state = env.reset()
    action = policy(state)
    episode = [(state, action, True, False)]
    while True:
        next_state, reward, done, _ = env.step(action)
        next_action = policy(next_state)
        episode.append((next_state, next_action, False, done))
        q_values[state, action] += \
            ALPHA * (reward + GAMMA * q_values[next_state, next_action]
                     - q_values[state, action])
        greedy_policy[state] = max(actions, key=lambda x: q_values[state, x])
        if done:
            if RENDER:
                os.system('clear')
                for (agent, dealer, ace), action, first, done in episode:
                    if first:
                        print(f"Dealer's hand: {dealer}\n")
                    print(f"Agent's hand: {agent}")
                    ace = "Yes" if ace else "No"
                    print(f"Usable ace: {ace}")
                    action = "Hit!" if action == 1 else "Stand!"
                    if not done:
                        print(f"Agent's action: {action}\n")
                    else:
                        if reward == 1:
                            print("\nWIN!")
                        elif reward == 0:
                            print("\nDRAW!")
                        elif agent > 21:
                            print("\nBUST!")
                        else:
                            print("\nLOSS! Dealer had a better hand!")
                input("\nPress enter to see another game!")
            if reward == 1:
                success_count += 1
                if not RENDER:
                    print(f"Successes: {success_count: 5}", end="\r")
                if success_count == SLOWDOWN_THRESHOLD:
                    RENDER = True
                    EPSILON = 0
                    print(f"Successes: {success_count: 5}")
            break
        action = next_action
        state = next_state
