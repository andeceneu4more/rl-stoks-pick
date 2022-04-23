from models import Estimator
from torch.optim import AdamW

import pdb
import torch
import torch.nn as nn
import numpy as np
import os

import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class EpsilonScheduller():
    def __init__(self, epsilon=1.0, epsilon_final=0.01, epsilon_decay=0.995):
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

    def get(self):
        return self.epsilon

    def step(self):
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay



class SimpleTrader():
    def __init__(self, state_size, action_space=3):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = []
        self.portfolio = []
        
        self.model = Estimator(state_size, action_space).to(device)
        self.optimizer = AdamW(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        
        self.gamma = 0.95
        self.epsilon_scheduller = EpsilonScheduller()

    def model_predict_proba(self, state):
        state = torch.tensor(state).float().to(device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(state)
        return logits[0].cpu().numpy()

    def model_predict(self, state):
        logits = self.model_predict_proba(state)
        return np.argmax(logits)

    def save_model(self, exp_dir='logs/simple_trader'):
        model_path = os.path.join(exp_dir, 'best.pth')
        torch.save({
            "model": self.model.state_dict()
        }, model_path)

    def training_step(self, state, target):
        state = torch.tensor(state).float().to(device)
        target = torch.tensor(target).float().to(device)

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(state)
        loss = self.loss_fn(logits, target)

        loss.backward()
        self.optimizer.step()
        

    def trade(self, state):
        rand = np.random.uniform(0, 1)
        if rand <= self.epsilon_scheduller.get():
            return np.random.randint(low=0, high=self.action_space)

        action = self.model_predict(state)
        return action

    def batch_train(self, batch_size):
        batch = []
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

        self.epsilon_scheduller.step()

class ImprovedTrader(SimpleTrader):
    def __init__(self, state_size, action_space=3):
        super().__init__(state_size, action_space)
        self.replay_size = 10000
        self.target_model = Estimator(state_size, action_space).to(device)
        self.target_model.eval()

    def sync_target(self):
        current_state_dict = self.model.state_dict()
        self.target_model.load_state_dict(current_state_dict)


    def training_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states).float().to(device)
        next_states = torch.tensor(next_states).float().to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        self.model.train()
        self.optimizer.zero_grad()

        state_action_probas = self.model(states).gather(1, actions.unsqueeze(-1))
        state_action_probas = state_action_probas.squeeze(-1)

        with torch.no_grad():
            next_state_probas, _ = self.target_model(next_states).max(axis=1)
            if done_mask.all().item() != 0:
                next_state_probas[done_mask] = 0.0

        expected_values = next_state_probas.detach() * self.gamma + rewards
        loss = self.loss_fn(state_action_probas, expected_values)

        loss.backward()
        self.optimizer.step()


    def batch_train(self, batch_size):
        replay_buffer = self.memory[-self.replay_size:]
        buffer_size = min(self.replay_size, len(replay_buffer))

        for i in range(0, buffer_size, batch_size):
            batch = replay_buffer[i: i + batch_size]

            states, actions, rewards, next_states, dones = list(zip(*batch))
            states = np.array(states).squeeze(1)
            next_states = np.array(next_states).squeeze(1)
            self.training_step(states, actions, rewards, next_states, dones)

        # ISSUE: check the book to be sure it should be done here?
        self.epsilon_scheduller.step()