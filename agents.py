from common import *
from models import BaseEstimator

class Agent(ABC):
    def __init__(self, 
        model             : torch.nn.Module,
        state_size        : int, 
        action_space      : int, 
        scheduler         : EpsilonScheduler,
        optimizer         : torch.nn.Module,
        loss_fn           : torch.nn.MSELoss,
    ):
        self.state_size   = state_size
        self.action_space = action_space
        self.portfolio    = []
        self.memory       = []

        self.model       = model
        self.scheduler   = scheduler
        self.optimizer   = optimizer
        self.loss_fn     = loss_fn
        self.gamma       = 0.95

    def model_predict_proba(self, state):
        state = torch.tensor(state).float().to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(state)
        return logits[0].cpu().numpy()

    def model_predict(self, state):
        logits = self.model_predict_proba(state)
        return np.argmax(logits)

    # Should be the same for all?
    def trade(self, state):
        rand = np.random.uniform(0, 1)
        if rand <= self.scheduler.get():
            # Predicting based on random
            return np.random.randint(low = 0, high = self.action_space)
        # Predicting based on AI   
        action = self.model_predict(state)
        return action

    # All agents should have the same default save model
    def save_model(self, model_path = "best.pth"):
        torch.save({"model": self.model.state_dict()}, model_path)

    @abstractmethod
    def training_step(self):
        pass

    @abstractmethod
    def batch_train(self):
        pass

class DQN(Agent):
    def __init__(self,
        model             : torch.nn.Module,
        state_size        : int, 
        action_space      : int, 
        scheduler         : EpsilonScheduler,
        optimizer         : torch.nn.Module,
        loss_fn           : torch.nn.MSELoss,
    ):
        super().__init__(model, state_size, action_space, scheduler, optimizer, loss_fn)

    def training_step(self, state, target):
        state = torch.tensor(state).float().to(DEVICE)
        target = torch.tensor(target).float().to(DEVICE)

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(state)
        loss   = self.loss_fn(logits, target)

        loss.backward()
        self.optimizer.step()
        
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

        self.scheduler.step()


class DQNFixedTargets(Agent):
    def __init__(self,
        model             : torch.nn.Module,
        state_size        : int, 
        action_space      : int, 
        scheduler         : EpsilonScheduler,
        optimizer         : torch.nn.Module,
        loss_fn           : torch.nn.MSELoss,
        target_model      : torch.nn.Module,
        replay_size       : int = 10000
    ):
        super().__init__(model, state_size, action_space, scheduler, optimizer, loss_fn)
        self.target_model = target_model
        self.replay_size  = replay_size

    def sync_target(self):
        current_state_dict = self.model.state_dict()
        self.target_model.load_state_dict(current_state_dict)

    def training_step(self, states, actions, rewards, next_states, dones):
        states      = torch.tensor(states).float().to(DEVICE)
        next_states = torch.tensor(next_states).float().to(DEVICE)
        actions     = torch.tensor(actions).to(DEVICE)
        rewards     = torch.tensor(rewards).to(DEVICE)
        done_mask   = torch.ByteTensor(dones).to(DEVICE)

        self.model.train()
        self.optimizer.zero_grad()

        state_action_probas = self.model(states).gather(1, actions.unsqueeze(-1))
        state_action_probas = state_action_probas.squeeze(-1)

        with torch.no_grad():
            next_state_probas, _ = self.target_model(next_states).max(axis = 1)
            if done_mask.all().item() != 0:
                next_state_probas[done_mask] = 0.0

        expected_values = next_state_probas.detach() * self.gamma + rewards
        loss = self.loss_fn(state_action_probas, expected_values)

        loss.backward()
        self.optimizer.step()

    def batch_train(self, batch_size):
        replay_buffer = self.memory[- self.replay_size :]
        buffer_size   = min(self.replay_size, len(replay_buffer))
        
        for i in range(0, buffer_size, batch_size):
            batch = replay_buffer[i: i + batch_size]
            states, actions, rewards, next_states, dones = list(zip(*batch))
            states = np.array(states).squeeze(1)
            next_states = np.array(next_states).squeeze(1)
            self.training_step(states, actions, rewards, next_states, dones)

        # ISSUE: check the book to be sure it should be done here?
        self.scheduler.step()
