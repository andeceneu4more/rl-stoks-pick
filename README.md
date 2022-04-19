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


## C. Algorithm design

1. procesari mai bune si deep models 

2. split up de actiune + alte metadate de extras (+ de verificat in alte paper-uri)

3. alte metode de RL (+ ce avantaje avem in gym) + implementare

4. api news for bert (finbert)

5. Metode de evaluare, pe langa reward

mesaj metoda + date - algorithm design