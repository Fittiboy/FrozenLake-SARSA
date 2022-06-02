from collections import defaultdict
from random import uniform
from time import sleep
from pprint import pprint
import os

import gym

env = gym.make("Blackjack-v0")
actions = list(range(env.action_space.n))
greedy_policy = defaultdict(int)
q_values = defaultdict(float)
success_count = 0
RENDER = False

EPSILON = 0.5
ALPHA = 0.5
GAMMA = 1
SLOWDOWN_THRESHOLD = 1000000

wins = 0
games = 0


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
        best_next_action = max(actions, key=lambda x: q_values[next_state, x])
        q_values[state, action] += \
            ALPHA * (reward + GAMMA * q_values[next_state, best_next_action]
                     - q_values[state, action])
        greedy_policy[state] = max(actions, key=lambda x: q_values[state, x])
        ALPHA *= 0.99999
        if done:
            if RENDER:
                os.system('clear')
                games += 1
                for (agent, dealer, ace), action, first, over in episode:
                    if first:
                        print(f"Dealer's hand: {dealer}\n")
                    print(f"Agent's hand: {agent}")
                    ace = "Yes" if ace else "No"
                    print(f"Usable ace: {ace}")
                    action = "Hit!" if action == 1 else "Stand!"
                    if not over:
                        print(f"Agent's action: {action}\n")
                    else:
                        if reward == 1:
                            print("\nWIN!")
                            wins += 1
                        elif reward == 0:
                            print("\nDRAW!")
                        elif agent > 21:
                            print("\nBUST!")
                        else:
                            print("\nLOSS! Dealer had a better hand!")
                if wins:
                    print(f"\nWin%: {(wins/games)*100: 2}%\n")
                input("Hit enter to play another episode!")
            if reward == 1:
                success_count += 1
                if not RENDER:
                    print(f"Successes: {success_count: 8}", end="\r")
                if success_count == SLOWDOWN_THRESHOLD:
                    RENDER = True
                    EPSILON = 0
                    ALPHA = 0
                    print(f"Successes: {success_count: 8}")
                    sleep(1)
            break
        action = next_action
        state = next_state
