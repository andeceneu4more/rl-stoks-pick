# Trading RL (from [tutorial](https://www.mlq.ai/deep-reinforcement-learning-for-trading-with-tensorflow-2-0/))

In trading we have an action space of 3: Buy, Sell, and Sit

We set the experience replay memory to deque with 2000 elements inside it
We create an empty list with inventory which contains the stocks we've already bought
We need to set an gamma parameter to 0.95, which helps to maximize the current reward over the long-term

The epsilon parameter is used to determine whether we should use a random action or to use the model for the action. We start by setting it to 1.0 so that it takes random actions in the beginning when the model is not trained. Over time we want to decrease the random actions and instead we can mostly use the trained model, so we set epsilon_final to 0.01


Let's first look at how we can translate the problem of stock market trading to a reinforcement learning environment.

Each point on a stock graph is just a floating number that represents a stock price at a given time.
Our task is to predict what is going to happen in the next period, and as mentioned there are 3 possible actions: buy, sell, or sit.
This is regression problem - let's say we have a window_size = 5 so we use 5 states to predict our target, which is a continuous number.
Instead of predicting real numbers for our target we instead want to predict one of our 3 actions.

# High Priority Tasks
1. ~~Split train, validation, test and integration~~ &#8594; **Assignee**: Adrian Iordache, **Status**: Done
2. Refactoring trader_agents.py for hyperparameter optimization (LR, loss_fn, optimizer) &#8594; **Assignee**: Adrian Iordache, **Status**: In progress
3. Monitoring and ploting results (profit, rewards, loss) &#8594; **Assignee**: Adrian Iordache, **Status**: In progress
4. Prioritized Experience Replay (already have this, integration for vanilla) - experiments and results for validation and test &#8594; **Assignee**: Manea Andrei, **Status**: In progress 
5. Another evaluation metrics (backtesting, [pyfolio](https://github.com/quantopian/pyfolio), [FinRL](https://github.com/AI4Finance-Foundation/FinRL/blob/master/tutorials/1-Introduction/FinRL_StockTrading_Fundamental.ipynb)) &#8594; **Assignee**: Sichitiu Marian, Gîdea Andrei, **Status**: In progress
6. OpenAI Gym Integration ? &#8594; **Assignee**: Manea Andrei, **Status**: In progress
7. Vanilla DQN - experiments and results for validation and test &#8594; **Assignee**: Manea Andrei, Adrian Iordache, **Status**: Waiting
8. DQN with fixed targets (target network)(already have this) - experiments and results for validation and test  &#8594; **Assignee**:  Manea Andrei, Adrian Iordache, **Status**: Waiting
9. Double DQN  - experiments and results for validation and test  &#8594; **Assignee**: Manea Andrei, **Status**: In progress 
10. Dueling Double DQN Architectures - experiments and results for validation and test &#8594; **Assignee**: Manea Andrei, **Status**: In progress 
11. Searching for other features as input ([stockstats](https://pypi.org/project/stockstats/)) &#8594; **Assignee**: Dobre Bogdan, Sichitiu Marian, Gîdea Andrei, **Status**: In progress
12. Improved Estimator Networks, Convolutional 1d vs Fully Connected, maybe LSTM or transformer encoder or transofrmer encoder decoder &#8594; **Assignee**: Sichitiu Marian, Gîdea Andrei, **Status**: In progress

# Optional Tasks
1. Throw away all episodes with a reward below the boundary
2. Random offset to start the experiments
3. Commision of the broker (0.1%)
4. reward += 100.0 * (close - self.open_price) / self.open_price


## C. Algorithm design (Class Requirements)
• Introduction

• What is the problem?

• Why can't any of the existing techniques effectively tackle this problem?

• What is the intuition behind the technique that you developed?

• Techniques to tackle the problem

• Brief review of previous work concerning this problem (i.e., the 4-8 papers that you read)
 - https://arxiv.org/pdf/1511.05952.pdf (prioritized replay)
 - https://arxiv.org/pdf/2011.09607.pdf (FinRL)
 - https://arxiv.org/pdf/2203.02628.pdf (If we use truncation as well)
 - https://arxiv.org/pdf/1703.01327.pdf (if we use TD)
 - others from https://github.com/tuxedcat/PER-D3QN

• Describe the technique that you developed

• Brief description of the existing techniques that you will compare to

• Evaluation

• Analyze and compare (empirically or theoretically) your new approach to existing approaches

• Conclusion:

• Can your new technique effectively tackle the problem?

• What future research do you recommend?
