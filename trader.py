from common import *
from trader_agents import ImprovedTrader, SimpleTrader

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

def main():
    data = datareader.DataReader('AAPL', data_source = "yahoo")
    data = data['Close'].tolist()

    n_episodes  = 1000
    window_size = 10

    batch_size   = 32
    data_samples = len(data)
    sync_steps   = 1000

    # TODO: put this in a function and make an inference for another set of yahoo stocks
    trader = ImprovedTrader(state_size = window_size)

    general_step = 0
    best_mean_reward = -1
    # Standard "epochs" iteration
    for episode in range(n_episodes):
        print(30 * "="+ f" Episode: {episode + 1}/{n_episodes} " + 30 * "=")
        
        total_profit = 0
        trader.portfolio, episode_rewards = [], []    
        
        # Standard "data" iteration
        for timestep, current_stock_price in enumerate(data):
            # We discard the last sample because we don't have a next state for it
            if timestep == len(data) - 1: break

            # Generate the current state to be predicted, 
            # [states] variable represents the input features for DNN to determine the action
            state  = state_creator(data, timestep, window_size + 1)    
            action = trader.trade(state)
            reward = 0
            
            if action == 1: # Buying
                trader.portfolio.append(current_stock_price)
                #print(f"AI Trader bought: {data[t]}")
            
            elif action == 2 and len(trader.portfolio) > 0: # Selling
                buy_price     = trader.portfolio.pop(0)
                profit        = current_stock_price - buy_price
                reward        = max(profit, 0)
                total_profit += profit
                # print(f"AI Trader sold: {data[t]}, Profit: {profit}")
            
            # The penultimate element is the last valid one 
            # Because we need one more sliding window after as the next state for prediction
            done = timestep == len(data) - 2
            next_state = state_creator(data, timestep + 1, window_size + 1)
            trader.memory.append((state, action, reward, next_state, done))
            episode_rewards.append(reward)

            general_step += 1
            if general_step % sync_steps == 0:
                trader.sync_target()

            if general_step % trader.replay_size == 0:
                trader.batch_train(batch_size)

        # This should be here, not inside the data iteration loop
        mean_reward = np.mean(episode_rewards)
        if best_mean_reward < mean_reward:
            best_mean_reward = mean_reward
            trader.save_model(
                model_name = f"model_episode_{episode}_profit_{RD(total_profit)}_reward_{RD(mean_reward)}.pth"
            )

        print("=" * 30 + f" Episode: {episode + 1}, Total Profit: {RD(total_profit)}")
        print("=" * 30 + f" Episode: {episode + 1}, Mean Reward: {RD(mean_reward)}")

if __name__ == "__main__":
    main()