from common import *
from models import BaseEstimator
from agents import DQN, DQNFixedTargets

CFG = {
    "action_space"  : 3,
    "learning_rate" : 0.001,
    "n_episodes"    : 1000,
    "window_size"   : 10,
    "batch_size"    : 32,
    "sync_steps"    : 1000
}

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

def train_fn(trader, train_data, window_size, global_step, batch_size, sync_target):
    train_profit = 0
    trader.portfolio, train_rewards = [], []    

    # Standard "data" iteration
    for timestep, current_stock_price in enumerate(train_data):
        # We discard the last sample because we don't have a next state for it
        if timestep == len(train_data) - 1: break

        # Generate the current state to be predicted, 
        # [states] variable represents the input features for DNN to determine the action
        state  = state_creator(train_data, timestep, window_size + 1)    
        action = trader.trade(state)
        reward = 0
        
        if action == 1: # Buying
            trader.portfolio.append(current_stock_price)
            #print(f"AI Trader bought: {data[t]}")
        
        elif action == 2 and len(trader.portfolio) > 0: # Selling
            buy_price     = trader.portfolio.pop(0)
            profit        = current_stock_price - buy_price
            reward        = max(profit, 0)
            train_profit += profit
            # print(f"AI Trader sold: {data[t]}, Profit: {profit}")
        
        # The penultimate element is the last valid one 
        # Because we need one more sliding window after as the next state for prediction
        done = timestep == len(train_data) - 2
        next_state = state_creator(train_data, timestep + 1, window_size + 1)
        trader.memory.append((state, action, reward, next_state, done))
        train_rewards.append(reward)

        global_step += 1
        if global_step % sync_target == 0:
            trader.sync_target()

        if global_step % trader.replay_size == 0:
            trader.batch_train(batch_size)

    return trader, np.mean(train_rewards), train_profit, global_step

def valid_fn(trader, valid_data, window_size):
    valid_profit = 0
    trader.portfolio, valid_rewards = [], []    

    for timestep, current_stock_price in enumerate(valid_data):
        state  = state_creator(valid_data, timestep, window_size + 1)    
        action = trader.model_predict(state)
        reward = 0

        if action == 1: # Buying
            trader.portfolio.append(current_stock_price)
            #print(f"AI Trader bought: {data[t]}")
        
        elif action == 2 and len(trader.portfolio) > 0: # Selling
            buy_price     = trader.portfolio.pop(0)
            profit        = current_stock_price - buy_price
            reward        = max(profit, 0)
            valid_profit += profit
            # print(f"AI Trader sold: {data[t]}, Profit: {profit}")

        valid_rewards.append(reward)

    return np.mean(valid_rewards), valid_profit

def main():
    data = pd.read_csv(PATH_TO_DATA)

    train_data = data[data['Split'] == 0]["Adj_Close"].tolist()
    valid_data = data[data['Split'] == 1]["Adj_Close"].tolist()
    test_data  = data[data['Split'] == 2]["Adj_Close"].tolist()
    
    model        = BaseEstimator(CFG['window_size'], CFG['action_space']).to(DEVICE)
    target_model = BaseEstimator(CFG['window_size'], CFG['action_space']).to(DEVICE)
    optimizer    = AdamW(model.parameters(), lr = CFG['learning_rate'])
    loss_fn      = nn.MSELoss()    
    scheduler    = EpsilonScheduler()


    trader = DQNFixedTargets(
        model        = model,
        target_model = target_model,
        state_size   = CFG['window_size'],
        action_space = CFG['action_space'],
        scheduler    = scheduler,
        optimizer    = optimizer,
        loss_fn      = loss_fn,
    )

    global_step = 0
    best_mean_reward = -1
    # Standard "epochs" iteration
    for episode in range(CFG['n_episodes']):
        print(40 * "="+ f" Episode: {episode + 1}/{CFG['n_episodes']} " + 40 * "=")

        # One train iteration over the training set
        trader, train_reward, train_profit, global_step = train_fn(
            trader      = trader, 
            train_data  = train_data, 
            window_size = CFG['window_size'], 
            global_step = global_step, 
            batch_size  = CFG['batch_size'], 
            sync_target = CFG['sync_steps']
        )

        # One valid iteration over the validation set
        valid_reward, valid_profit = valid_fn(trader, valid_data, CFG['window_size'])

        # Saving the best model based on the reward on the validation set
        if best_mean_reward < valid_reward:
            best_mean_reward = valid_reward
            trader.save_model(
                model_name = f"model_episode_{episode}_profit_{RD(valid_profit)}_reward_{RD(valid_profit)}.pth"
            )

        print("=" * 40 + f" Episode: {episode + 1}, Train Profit: {RD(train_profit)}, Train Reward: {RD(train_reward)}")
        print("=" * 40 + f" Episode: {episode + 1}, Valid Profit: {RD(valid_profit)}, Valid Reward: {RD(valid_profit)}")

if __name__ == "__main__":
    main()