from site import USER_BASE
from common import *
from getters import *
import matplotlib.pyplot as plt

plt.style.use(["ggplot"])

# REFACTOR: moved them in the common.py
# USER          = "andreig"
# STAGE         = 0 # if we change the structure of the GlobalLogger.csv we increase the stage number
SAVE_TO_LOG     = True # change this to False if you don't want to save the experiment

GLOBAL_LOGGER = GlobalLogger(
    path_to_global_logger = f'logs/{USER}/stage-{STAGE}/global_logger.csv',
    save_to_log = SAVE_TO_LOG
)

CFG = {
    "id"            : GLOBAL_LOGGER.get_version_id(),
    "trader"        : "DQNFixedTargets",
    "estimator"     : "BaseEstimator",

    """
    https://pypi.org/project/stockstats/

    How to set "features_used"


    General rules {

        Retrieve the symbol with 2 arguments ( <columnName>_<windowSize>_<statistics> )
            -> 5 periods simple moving average of the high price: "high_5_sma"
            -> 10 periods exponential moving average of the close: "close_10_ema"
            -> 1 period delta of the high price: "high_-1_d" The - symbol stands for looking backwards.

        Retrieve the symbol with 1 arguments ( <statistics>_<windowSize> )
            -> 6 periods RSI: "rsi_6"
            -> 10 periods CCI: "cci_10"
            -> 13 periods ATR: "atr_13"

    }


    Change of the Close
        -> "change" is the change of the close price in percentage.

    Delta of Periods ( <column>_<window>_d or <column>_delta as a shortcut to <column>_-1_d )
        -> "close_-1_d" retrieves the close price delta between current and prev. period.
        -> "close_delta" is the same as "close_-1_d"
        -> "high_2_d" retrieves the high price delta between current and 2 days later

    Shift Periods
        Shift the column backward or forward. It takes 2 parameters:
            <> the name of the column to shift
            <> periods to shift, can be negative

            Example:
                In [15]: df[['close', 'close_-1_s', 'close_2_s']]
                Out[15]:
                          close    close_-1_s  close_2_s
                date
                20040817  11.20       11.20      10.53
                20040818  10.29       11.20      10.55
                20040819  10.53       10.29      10.10
                20040820  10.55       10.53      10.25

    RSI - Relative Strength Index (https://en.wikipedia.org/wiki/Relative_strength_index)
        RSI chart the current and historical strength or weakness of a stock. It takes a window parameter.

        -> "rsi" 14 periods RSI
        -> "rsi_6" 6 periods RSI

    Stochastic RSI (https://www.investopedia.com/terms/s/stochrsi.asp)
        Stochastic RSI gives traders an idea of whether the current RSI value is overbought or oversold. It takes a window parameter.

        Examples:
            -> "stochrsi" retrieve the Stochastic RSI of 14 periods
            -> "stochrsi_6" retrieve the Stochastic RSI of 6 periods

    Log Return of the Close (https://en.wikipedia.org/wiki/Rate_of_return)
        Logarithmic return = ln( close / last close)

        From wiki:
            For example, if a stock is priced at 3.570 USD per share at the close on one day, and at 3.575 USD per share 
            at the close the next day, then the logarithmic return is: ln(3.575/3.570) = 0.0014, or 0.14%.

        -> "log-ret" access this column.

    RSV - Raw Stochastic Value
        RSV is essential for calculating KDJ. It takes a window parameter.

        From wiki:
            KDJ indicator is a technical indicator used to analyze and predict changes in stock trends and 
            price patterns in a traded asset. KDJ indicator is otherwise known as the random index. 
            It is a very practical technical indicator which is most commonly used in market trend analysis of short-term stock.

        -> "rsv" or "rsv_6" to access it.

    SMMA - Smoothed Moving Average
        It takes two parameters, column and window.

        For example, use "close_7_smma'] to retrieve the 7 periods smoothed moving average of the close price.

    TRIX - Triple Exponential Average (https://www.investopedia.com/articles/technical/02/092402.asp)
        The triple exponential average is used to identify oversold and overbought markets.

        The algorithm is:

            TRIX = (TripleEMA - LastTripleEMA) -  * 100 / LastTripleEMA
            TripleEMA = EMA of EMA of EMA
            LastTripleEMA =  TripleEMA of the last period

        It takes two parameters, column and window. By default, the column is close, the window is 12.


        Examples:

            -> "trix" stands for 12 periods Trix for the close price.
            -> "middle_10_trix" stands for the 10 periods Trix for the typical(middle) price.

    TEMA - Another Triple Exponential Average (https://www.forextraders.com/forex-education/forex-technical-analysis/triple-exponential-moving-average-the-tema-indicator/)
        Tema is another implementation for the triple exponential moving average.

        TEMA=(3 x EMA) - (3 x EMA of EMA) + (EMA of EMA of EMA)

        It takes two parameters, column and window. By default, the column is close, the window is 5.

        Examples:

            "tema" stands for 12 periods TEMA for the close price.
            "middle_10_tema" stands for the 10 periods TEMA for the typical(middle) price.

    VR - Volume Variation Index (https://help.eaglesmarkets.com/hc/en-us/articles/900002867026-Summary-of-volume-variation-index)
        It is the strength index of trading volume.

        Examples:

            "vr" retrieves the 26 periods VR.
            "vr_6" retrieves the 6 periods VR.

    Typical Price (https://en.wikipedia.org/wiki/Typical_price)
        It's the average of high, low and close.

            -> "middle" access this value.

    Simple Moving Average (https://www.investopedia.com/terms/m/mean.asp)
        Follow the pattern <columnName>_<window>_sma to retrieve simple moving average.

        -> "high_5_sma" 5 periods simple moving average of the high price

    Moving Standard Deviation (https://www.investopedia.com/terms/s/standarddeviation.asp)
        Follow the pattern <columnName>_<window>_mstd to retrieve the moving STD.

        -> "high_5_mstd" 5 periods moving standard deviation of the high price

    Moving Variance (https://www.investopedia.com/terms/v/variance.asp)
        Follow the pattern <columnName>_<window>_mvar to retrieve the moving VAR.

        -> "high_5_mvar" 5 periods moving variance of the high price

    Volume Weighted Moving Average (https://www.investopedia.com/articles/trading/11/trading-with-vwap-mvwap.asp)
        It's the moving average weighted by volume.

        Examples:

        -> "vwma" retrieves the 14 periods VWMA
        -> "vwma_6" retrieves the 6 periods VWMA

    CHOP - Choppiness Index (https://www.tradingview.com/education/choppinessindex/)
        The Choppiness Index determines if the market is choppy.

        It has a parameter for window size. The default window is 14. Change it with StockDataFrame.CHOP.

        Examples:

        -> "chop" retrieves the 14 periods CHOP
        -> "chop_6" retrieves the 6 periods CHOP

    MFI - Money Flow Index (https://www.investopedia.com/terms/m/mfi.asp)
        The Money Flow Index identifies overbought or oversold signals in an asset.

        It has a parameter for window size. The default window is 14. Change it with StockDataFrame.MFI.

        Examples:

        -> "mfi" retrieves the 14 periods MFI
        -> "mfi_6" retrieves the 6 periods MFI

    KAMA - Kaufman's Adaptive Moving Average (https://school.stockcharts.com/doku.php?id=technical_indicators:kaufman_s_adaptive_moving_average)
        Kaufman's Adaptive Moving Average is designed to account for market noise or volatility.

        It has 2 optional parameter and 2 required parameter

        fast - optional, the parameter for fast EMA smoothing, default to 5
        slow - optional, the parameter for slow EMA smoothing, default to 34
        column - required, the column to calculate
        window - required, rolling window size

        Examples:

        -> "close_10_kama_2_30" retrieves 10 periods KAMA of the close price with fast = 2 and slow = 30
        -> "close_2_kama" retrieves 2 periods KAMA of the close price

    """

    "features_used" : ["adj_close", "rsi"],
    "target_used"   : "adj_close",

    "normalizer"    : ["sigmoid", "minmax"], # same ln as "features_used"; each feature with its normalizer
    
    "optimizer"     : "AdamW",
    "learning_rate" : 0.001,
    
    "criterion"     : "MSELoss",

    "eps_scheduler" : "EpsilonScheduler",
    "epsilon"       : 1,
    "epsilon_final" : 0.01,
    "epsilon_decay" : 0.995,

    "action_space"  : 3,
    "window_size"   : 10,                       # the same thing as state_size
    "batch_size"    : 32,
    "n_episodes"    : 10,

    "replay_size"   : 1000,
    "sync_steps"    : 1000,                     # only for DQN with fixed targets

    "prob_alpha"    : 0.6,                      # only in DQN with prioritized targets
    "beta_start"    : 0.4
}

# This should be updated after the we find the financial metrics
OUTPUTS = {
    "train_profit": "NA",
    "valid_profit": "NA",

    "train_reward": "NA",
    "valid_reward": "NA",
    "observation" : "Use all features from features_used field" # This field should be used as a comment in GloablLogger.csv
}

SCALERS = {

}

def state_creator(data, timestep, window_size):
    """ Generating Features for the estimators """
    starting_id = timestep - window_size + 1

    # # TODO: use all features, not just the first in the list
    # # data = np.squeeze(data)
    # data = np.array(data)[:, 0]

    if starting_id >= 0:
        windowed_data = data[starting_id: timestep + 1]
    else:
        # if there are not data points we padd it by repeating the first element
        # windowed_data = int(np.abs(starting_id)) * [data[0]] + data[0: timestep + 1]
        windowed_data = np.concatenate((int(np.abs(starting_id)) * [data[0]], data[0: timestep + 1]), axis=0)
    
    state = []
    for i in range(len(windowed_data) - 1):
        # state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))
        state.append(normalize_features(windowed_data[i + 1], windowed_data[i], CFG, SCALERS))
    # for i in range(len(windowed_data) - 1):
    #     state.append(windowed_data[i])

    return np.array([state])

def train_fn(trader, train_data, window_size, global_step, batch_size, sync_target):
    train_profit = 0
    trader.portfolio, train_rewards = [], []    

    # Standard "data" iteration
    # for timestep, current_stock_price in enumerate(train_data):
    #     print(current_stock_price)
    features, stock_prices = train_data[0], train_data[1]
    state_creator_param = features
    for timestep, current_stock_price in enumerate(stock_prices):
        # We discard the last sample because we don't have a next state for it
        if timestep == len(state_creator_param) - 1: break

        # Generate the current state to be predicted, 
        # [states] variable represents the input features for DNN to determine the action
        state  = state_creator(state_creator_param, timestep, window_size + 1)

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
        done = timestep == len(state_creator_param) - 2
        next_state = state_creator(state_creator_param, timestep + 1, window_size + 1)
        trader.append_state((state, action, reward, next_state, done))
        train_rewards.append(reward)

        global_step += 1
        # Training the trader based on the specific type
        # This should be updated when the replay size will be usable for all Agents
        if CFG['trader'] == "DQN":
            if len(trader.memory) % batch_size == 0:
                trader.batch_train(batch_size)
        else:
            if global_step % trader.replay_size == 0:
                trader.batch_train(batch_size)
            
            if CFG['trader'] in ["DQNFixedTargets", "DQNPrioritizedTargets", "DQNDouble"] \
                and global_step % sync_target == 0:
                # ISSUE: if replay_size > sync_target
                trader.sync_target()
            

    return trader, np.mean(train_rewards), train_profit, global_step

def valid_fn(trader, valid_data, window_size):
    valid_profit = 0
    trader.portfolio, valid_rewards = [], []    

    # for timestep, current_stock_price in enumerate(valid_data):
    features, stock_prices = valid_data[0], valid_data[1]
    state_creator_param = features
    for timestep, current_stock_price in enumerate(stock_prices):
        state  = state_creator(state_creator_param, timestep, window_size + 1)    
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
    PATH_TO_MODEL = f"models/{USER}/stage-{STAGE}/model-{CFG['id']}"
    if SAVE_TO_LOG:
        if not os.path.isdir(PATH_TO_MODEL): os.makedirs(PATH_TO_MODEL, 0o777)
        logger = Logger(
            path_to_logger = os.path.join(PATH_TO_MODEL, 'model_{}.log'.format(CFG['id'])), 
            distributed    = False
        )
    else:
        logger = Logger(distributed = False)

    logger.print("Config File")
    logger.print(CFG)

    data       = pd.read_csv(PATH_TO_DATA)
    data      = wrap(data)

    train_data = [data[data['split'] == 0][CFG["features_used"]].fillna(0).values.tolist(), data[data['split'] == 0][CFG["target_used"]].fillna(0).tolist()]
    valid_data = [data[data['split'] == 1][CFG["features_used"]].fillna(0).values.tolist(), data[data['split'] == 1][CFG["target_used"]].fillna(0).tolist()]
    test_data = [data[data['split'] == 2][CFG["features_used"]].fillna(0).values.tolist(), data[data['split'] == 2][CFG["target_used"]].fillna(0).tolist()]

    # Train MinMaxScaler on data if it will be used in normalizer
    if 'minmax' in CFG['normalizer']:
        from sklearn.preprocessing import MinMaxScaler

        for idx, elem in enumerate(CFG['normalizer']):
            if elem == 'minmax':
                scaler = MinMaxScaler()
                scaler.fit(np.array(train_data[0])[:, idx].reshape(-1, 1))

                SCALERS[CFG["features_used"][idx]] = scaler
                # print(np.array(train_data[0])[:, idx])


    # valid_data = data[data['split'] == 1][["adj_close"]].values.tolist()
    # test_data  = data[data['split'] == 2][["adj_close"]].values.tolist()

    # train_data = data[data['split'] == 0][CFG["target_used"]].tolist()
    # valid_data = data[data['split'] == 1][CFG["target_used"]].tolist()
    # test_data  = data[data['split'] == 2][CFG["target_used"]].tolist()
    
    # All getter for the config file can be found in getters.py
    model        = get_estimator(CFG)
    target_model = get_estimator(CFG) # the target model should be the same as the estimator
    optimizer    = get_optimizer(model.parameters(), CFG)
    loss_fn      = get_criterion(CFG) 
    scheduler    = get_scheduler(CFG)

    trader = None
    if CFG["trader"] == "DQN":
        trader = DQN(
            model        = model,
            state_size   = CFG['window_size'],
            action_space = CFG['action_space'],
            scheduler    = scheduler,
            optimizer    = optimizer,
            loss_fn      = loss_fn,
        )
    elif CFG["trader"] == "DQNVanilla":
        trader = DQNVanilla(
            model        = model,
            state_size   = CFG['window_size'],
            action_space = CFG['action_space'],
            scheduler    = scheduler,
            optimizer    = optimizer,
            loss_fn      = loss_fn,
            replay_size  = CFG['replay_size']
        )
    elif CFG["trader"] == "DQNFixedTargets":
        trader = DQNFixedTargets(
            model        = model,
            target_model = target_model,
            state_size   = CFG['window_size'],
            action_space = CFG['action_space'],
            scheduler    = scheduler,
            optimizer    = optimizer,
            loss_fn      = loss_fn,
            replay_size  = CFG['replay_size']
        )
    elif CFG["trader"] == "DQNPrioritizedTargets":
        trader = DQNPrioritizedTargets(
            model        = model,
            target_model = target_model,
            state_size   = CFG['window_size'],
            action_space = CFG['action_space'],
            scheduler    = scheduler,
            optimizer    = optimizer,
            loss_fn      = loss_fn,
            replay_size  = CFG['replay_size'],
            prob_alpha   = CFG['prob_alpha'],
            beta_start   = CFG['beta_start'],
            n_episodes   = CFG['n_episodes']
        )
    elif CFG["trader"] == "DQNDouble":
        trader = DQNDouble(
            model        = model,
            target_model = target_model,
            state_size   = CFG['window_size'],
            action_space = CFG['action_space'],
            scheduler    = scheduler,
            optimizer    = optimizer,
            loss_fn      = loss_fn,
            replay_size  = CFG['replay_size']
        )

    global_step = 0
    best_mean_reward = 0
    train_profits, valid_profits = [], [] 
    train_rewards, valid_rewards = [], []

    output_train_profit, output_valid_profit = 0, 0
    output_train_reward, output_valid_reward = 0, 0

    # Standard "epochs" iteration
    for episode in range(CFG['n_episodes']):
        logger.print(40 * "="+ f" Episode: {episode + 1}/{CFG['n_episodes']} " + 40 * "=")

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

            output_train_profit = train_profit
            output_valid_profit = valid_profit
            output_train_reward = train_reward
            output_valid_reward = valid_reward

            if SAVE_TO_LOG:
                # save as before
                trader.save_model(
                    model_path = os.path.join(
                        PATH_TO_MODEL, f"model_{CFG['id']}_episode_{episode}_profit_{RD(valid_profit)}_reward_{RD(valid_reward)}.pth")
                )
                # also save as "best" just for the ease of use in the inference
                trader.save_model(
                    model_path = os.path.join(
                        PATH_TO_MODEL, f"best.pth")
                )

        train_profits.append(train_profit)
        valid_profits.append(valid_profit)
        train_rewards.append(train_reward)
        valid_rewards.append(valid_reward)

        logger.print("=" * 40 + f" Episode: {episode + 1}, Train Profit: {RD(train_profit)}, Train Reward: {RD(train_reward)}")
        logger.print("=" * 40 + f" Episode: {episode + 1}, Valid Profit: {RD(valid_profit)}, Valid Reward: {RD(valid_profit)}")

    plt.figure(figsize = (10, 5))
    plt.plot(train_rewards, label = "Train Rewards")
    plt.plot(valid_rewards, label = "Valid Rewards")
    plt.legend()
    plt.title(f"[Model {CFG['id']}]: Rewards")
    plt.ylabel(f"Rewards")
    if SAVE_TO_LOG: 
        plt.savefig(
            os.path.join(PATH_TO_MODEL, f"rewards_model_{CFG['id']}.png")
        )
    plt.show()

    plt.figure(figsize = (10, 5))
    plt.plot(train_profits, label = "Train Profits")
    plt.plot(valid_profits, label = "Valid Profits")
    plt.legend()
    plt.title(f"[Model {CFG['id']}]: Profits")
    plt.ylabel(f"Profits")
    if SAVE_TO_LOG: 
        plt.savefig(
            os.path.join(PATH_TO_MODEL, f"profits_model_{CFG['id']}.png")
        )
    plt.show()

    OUTPUTS['train_profit'] = RD(output_train_profit)
    OUTPUTS['valid_profit'] = RD(output_valid_profit)
    OUTPUTS['train_reward'] = RD(output_train_reward)
    OUTPUTS['valid_reward'] = RD(output_valid_reward)
    GLOBAL_LOGGER.append(CFG, OUTPUTS)

if __name__ == "__main__":
    main()