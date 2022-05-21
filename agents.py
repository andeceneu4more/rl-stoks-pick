from common import *

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

    def append_state(self, state):
        self.memory.append(state)

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
        batch = self.memory[-batch_size:]

        for state, action, reward, next_state, done in batch:
            if not done:
                logits = self.model_predict_proba(next_state)
                reward = reward + self.gamma * np.max(logits)
            target = self.model_predict_proba(state)
            target[action] = reward

            target = np.array([target])            
            self.training_step(state, target)

        self.scheduler.step()

class DQNVanilla(Agent):
    def __init__(self,
        model             : torch.nn.Module,
        state_size        : int, 
        action_space      : int, 
        scheduler         : EpsilonScheduler,
        optimizer         : torch.nn.Module,
        loss_fn           : torch.nn.MSELoss,
        replay_size       : int = 10000
    ):
        super().__init__(model, state_size, action_space, scheduler, optimizer, loss_fn)
        self.replay_size  = replay_size

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
            next_state_probas, _ = self.model(next_states).max(axis = 1)
            if done_mask.all().item() != 0:
                next_state_probas[done_mask] = 0.0

        expected_values = next_state_probas.detach() * self.gamma + rewards
        loss = self.loss_fn(state_action_probas, expected_values)

        loss.backward()
        self.optimizer.step()

    def batch_train(self, batch_size):
        replay_index = max(0, len(self.memory) - self.replay_size)
        
        for i in range(replay_index, len(self.memory), batch_size):
            batch = self.memory[i: i + batch_size]
            states, actions, rewards, next_states, dones = list(zip(*batch))
            states = np.array(states).squeeze(1)
            next_states = np.array(next_states).squeeze(1)
            self.training_step(states, actions, rewards, next_states, dones)

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

        print("States:",states.size())
        print("Actions:",actions.unsqueeze(-1).size())
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
        replay_index = max(0, len(self.memory) - self.replay_size)
        
        for i in range(replay_index, len(self.memory), batch_size):
            batch = self.memory[i: i + batch_size]
            states, actions, rewards, next_states, dones = list(zip(*batch))
            states = np.array(states).squeeze(1)
            next_states = np.array(next_states).squeeze(1)
            self.training_step(states, actions, rewards, next_states, dones)

        self.scheduler.step()

class DQNPrioritizedTargets(Agent):
    
    def __init__(self,
        model             : torch.nn.Module,
        state_size        : int, 
        action_space      : int, 
        scheduler         : EpsilonScheduler,
        optimizer         : torch.nn.Module,
        loss_fn           : torch.nn.MSELoss,
        target_model      : torch.nn.Module,
        replay_size       : int = 10000,
        prob_alpha        : float = 0.6,
        beta_start        : float = 0.4,
        n_episodes        : int = 1000
    ):
        super().__init__(model, state_size, action_space, scheduler, optimizer, loss_fn)
        self.target_model = target_model
        self.replay_size  = replay_size
        self.memory       = []
        
        self.priorities   = []
        self.prob_alpha   = prob_alpha
        
        self.beta_start   = beta_start
        self.beta         = beta_start
        self.episode_id   = 0
        self.n_episodes   = n_episodes

    def detach_low_prio_states(self):
        """ gets rid of states with low priority if the memory exceeds the replay buffer size; made for optimization purpose and it replaces the buffer position from the book
        """
        if len(self.memory) > self.replay_size:
            prios = np.array(self.priorities)
            probs = prios ** self.prob_alpha
            probs /= probs.sum()

            indices = np.random.choice(len(self.memory), self.replay_size // 4, p=probs)
            buffer = [self.memory[idx] for idx in indices]
            priorities = [self.priorities[idx] for idx in indices]
            
            self.memory = buffer
            self.priorities = priorities

    def sync_target(self):
        current_state_dict = self.model.state_dict()
        self.target_model.load_state_dict(current_state_dict)

    def append_state(self, state):
        """ appends to the memory in the limit of replay size; If the limit is archieved, the states are replaced from left to right
        """
        self.memory.append(state)
        prio_max = np.max(self.priorities) if len(self.priorities) > 0 else 1
        self.priorities.append(prio_max)
        

    def sample_batch(self, batch_size):
        """ using formulae from the article/book it selects the memorized states in a specific order and returns the batch_weights
        """
        prios = np.array(self.priorities)
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        """ updates the priorities with the weighted loss computed on the specific batch 
        """
        for idx, prio in zip(batch_indices, batch_priorities.tolist()):
            self.priorities[idx] = prio

    def training_step(self, states, actions, rewards, next_states, dones, batch_weights):
        """ same as before with weighted L1Loss extracted from batch_weights
        """
        states        = torch.tensor(states).float().to(DEVICE)
        next_states   = torch.tensor(next_states).float().to(DEVICE)
        actions       = torch.tensor(actions).to(DEVICE)
        rewards       = torch.tensor(rewards).to(DEVICE)
        done_mask     = torch.ByteTensor(dones).to(DEVICE)
        batch_weights = torch.tensor(batch_weights).to(DEVICE)

        self.model.train()
        self.optimizer.zero_grad()

        state_action_probas = self.model(states).gather(1, actions.unsqueeze(-1))
        state_action_probas = state_action_probas.squeeze(-1)

        with torch.no_grad():
            next_state_probas, _ = self.target_model(next_states).max(axis = 1)
            if done_mask.all().item() != 0:
                next_state_probas[done_mask] = 0.0

        expected_values = next_state_probas.detach() * self.gamma + rewards
        losses = batch_weights * (state_action_probas - expected_values) ** 2
        loss = losses.mean()

        loss.backward()
        self.optimizer.step()
        return losses + 1e-5

    def batch_train(self, batch_size):
        self.beta = min(1.0, self.beta_start + self.episode_id * (1.0 - self.beta_start) / self.n_episodes)
        self.episode_id += 1

        n_batches = self.replay_size // batch_size
        for _ in range(n_batches):
            batch, batch_indices, batch_weights = self.sample_batch(batch_size)
            states, actions, rewards, next_states, dones = list(zip(*batch))
            states = np.array(states).squeeze(1)
            next_states = np.array(next_states).squeeze(1)

            sampled_prios = self.training_step(states, actions, rewards, next_states, dones, batch_weights)
            self.update_priorities(batch_indices, sampled_prios)

        self.scheduler.step()


class DQNDouble(Agent):
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
        """ same as fixed targets, but for extracting the probabilities: instead of choosing the best action from the target model, it uses the best actions evaluated by the training model 
        """
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
            model_qp_out = self.model(next_states)
            next_state_actions = torch.argmax(model_qp_out, axis = 1)

            next_state_probas = self.target_model(next_states)
            next_state_probas = next_state_probas.gather(1, next_state_actions.unsqueeze(-1))
            next_state_probas = next_state_probas.squeeze(-1)

            if done_mask.all().item() != 0:
                next_state_probas[done_mask] = 0.0

        expected_values = next_state_probas.detach() * self.gamma + rewards
        loss = self.loss_fn(state_action_probas, expected_values)

        loss.backward()
        self.optimizer.step()

    def batch_train(self, batch_size):
        replay_index = max(0, len(self.memory) - self.replay_size)
        
        for i in range(replay_index, len(self.memory), batch_size):
            batch = self.memory[i: i + batch_size]
            states, actions, rewards, next_states, dones = list(zip(*batch))
            states = np.array(states).squeeze(1)
            next_states = np.array(next_states).squeeze(1)
            self.training_step(states, actions, rewards, next_states, dones)

        self.scheduler.step()