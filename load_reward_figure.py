import pickle
# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

makerList = ['o', '*', '+', 'D', 's', '-']
#labelList= ["greedy", "random", "e-greedy", "boltzmann", "bayesian"]
labelList = ["vallina actor-critic", "active actor-critic",
             "e-greedy", "boltzmann", "bayesian"]
# load the original
with open('ac_orig_cartpole1.pickle', 'rb') as fp:
    [a1, b1, lens1, rewards1] = pickle.load(fp)

with open('ac_active_cartpole.pickle', 'rb') as fp:
    [a2, b2, lens2, rewards2] = pickle.load(fp)

smoothing_window = 10

# plot the how episode length change over time
fig1 = plt.figure(figsize=(10, 5))
i = 0
x = range(len(lens1))
plt.plot(x, lens1, marker=makerList[i],
         label=labelList[i])  # plotting by columns
i = i + 1
x = range(len(lens1))
plt.plot(x, lens2, marker=makerList[i],
         label=labelList[i])  # plotting by columns

plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("Episode Length over Time")
ax = plt.gca()
ax.legend(loc='best')
plt.show()


# plot the how episode reward change over time
fig2 = plt.figure(figsize=(10, 5))
i = 0
x = range(len(rewards1))
rewards_smoothed = pd.Series(rewards1).rolling(
    smoothing_window, min_periods=smoothing_window).mean()
x = range(len(rewards_smoothed))
# plotting by columns
plt.plot(x, rewards_smoothed, marker=makerList[i], label=labelList[i])

i = i + 1
rewards_smoothed = pd.Series(rewards2).rolling(
    smoothing_window, min_periods=smoothing_window).mean()
plt.plot(x, rewards_smoothed, marker=makerList[i], label=labelList[i])
plt.xlabel("Episode")
plt.ylabel("Episode Reward (Smoothed)")
plt.title("Episode Reward over Time (Smoothed over window size {})".format(
    smoothing_window))


# Plot time steps and episode number
i = 0
x = range(len(lens1))
fig3 = plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(lens1), np.arange(len(lens1)),
         marker=makerList[i], label=labelList[i])

plt.xlabel("Time Steps")
plt.ylabel("Episode")
plt.title("Episode per time step")

ax = plt.gca()
ax.legend(loc='best')
plt.show()


# multiple line plot
'''
plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
'''
