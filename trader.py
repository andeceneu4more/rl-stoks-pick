
from tqdm import trange
from pandas_datareader import data as data_reader
from trader_agents import ImprovedTrader, SimpleTrader

import numpy as np
# TODO: add device as well

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def state_creator(data, timestep, window_size):
    starting_id = timestep - window_size + 1

    if starting_id >= 0:
        windowed_data = data[starting_id: timestep + 1]
    else:
        # if there are not data points we padd it by repeating the first element
        windowed_data = int(np.abs(starting_id)) * [data[0]] + data[0: timestep + 1]
    
    state = []
    for i in range(len(windowed_data) - 1):
        state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))

    return np.array([state])


dataset = data_reader.DataReader('AAPL', data_source="yahoo")
data = dataset['Close'].tolist()

window_size = 10
n_episodes = 1000

batch_size = 32
data_samples = len(data) - 1
sync_steps  = 1000


# TODO: put this in a function and make an inference for another set of yahoo stocks
trader = ImprovedTrader(state_size=window_size)

general_step = 0
best_mean_reward = -1

for episode in range(n_episodes):
  
    print("Episode: {}/{}".format(episode, n_episodes))
    state = state_creator(data, 0, window_size + 1)

    total_profit = 0
    trader.portfolio = []
    rewards = []    
  
    for t in trange(data_samples):
        action = trader.trade(state)
        
        next_state = state_creator(data, t + 1, window_size + 1)
        reward = 0

        if action == 1: # Buying
            trader.portfolio.append(data[t])
            # print(f"AI Trader bought: {data[t]}")
        
        elif action == 2 and len(trader.portfolio) > 0: # Selling
            buy_price = trader.portfolio.pop(0)
            
            profit = data[t] - buy_price
            reward = max(profit, 0)
            total_profit += profit
            # print(f"AI Trader sold: {data[t]}, Profit: {profit}")
        
        done = t == data_samples - 1
        trader.memory.append((state, action, reward, next_state, done))
        state = next_state

        rewards.append(reward)
        mean_reward = np.mean(rewards)
        if best_mean_reward < mean_reward:
            best_mean_reward = mean_reward
            trader.save_model()

        general_step += 1
        if general_step % sync_steps == 0:
            trader.sync_target()

    print("#" * 20)
    print(f"Episode {episode} ended with a total profit = {total_profit}")
    print("#" * 20)

    if len(trader.memory) > trader.replay_size:
        trader.batch_train(batch_size)