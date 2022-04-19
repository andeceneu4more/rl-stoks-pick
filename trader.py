from collections import deque

from tqdm import trange
from pandas_datareader import data as data_reader
from torch.optim import AdamW

import torch
import torch.nn as nn
import numpy as np
import os
import pdb


class Estimator(nn.Module):
    def __init__(self, state_size, action_space):
        super().__init__()
        self.state_size = state_size
        self.action_space = action_space

        self.model = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

    def forward(self, x):
        return self.model(x)


class SimpleTrader():
    def __init__(self, state_size, action_space=3, model_name="SimpleTrader"):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque()
        self.portfolio = []
        
        self.model_name = model_name
        self.model = Estimator(state_size, action_space)
        self.optimizer = AdamW(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

    def model_predict_proba(self, state):
        state = torch.tensor(state).float()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(state)
        return logits[0].numpy()

    def model_predict(self, state):
        logits = self.model_predict_proba(state)
        return np.argmax(logits)

    def save_model(self, exp_dir='logs/simple_trader'):
        model_path = os.path.join(exp_dir, 'best.pth')
        torch.save({
            "model": self.model.state_dict()
        }, model_path)

    def training_step(self, state, target):
        state = torch.tensor(state).float()
        target = torch.tensor(target).float()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(state)
        loss = self.loss_fn(logits, target)

        loss.backward()
        self.optimizer.step()
        

    def trade(self, state):
        rand = np.random.uniform(0, 1)
        if rand <= self.epsilon:
            return np.random.randint(low=0, high=self.action_space)

        action = self.model_predict(state)
        return action

    def batch_train(self, batch_size):
        batch = []
        # TODO: check if iterates all of them
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state, action, reward, next_state, done in batch:
            if not done:
                logits = self.model_predict_proba(next_state)
                reward = reward + self.gamma * np.max(logits)

            target = self.model_predict_proba(state)
            target[action] = reward

            target = np.array([target])            
            self.training_step(state, target)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

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

trader = SimpleTrader(state_size=window_size)

for episode in range(n_episodes):
  
    print("Episode: {}/{}".format(episode, n_episodes))
    state = state_creator(data, 0, window_size + 1)

    total_profit = 0
    trader.portfolio = []
  
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

    print("#" * 20)
    print(f"Episode {episode} ended with a total profit = {total_profit}")
    print("#" * 20)

    if len(trader.memory) > batch_size:
        trader.batch_train(batch_size)
        
    if episode % 10 == 0:
        trader.save_model()
    