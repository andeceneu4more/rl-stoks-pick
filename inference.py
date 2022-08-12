from common  import *
from trader  import valid_fn, state_creator, CFG
from getters import get_estimator 

def test_fn(model, test_data, window_size, scalers):
    total_profit = 0
    portfolio, rewards, account_values = [], [], []
    actions = []

    account_value = 1000

    # for timestep, current_stock_price in enumerate(test_data):
    features, stock_prices = test_data[0], test_data[1]
    state_creator_param = features
    for timestep, current_stock_price in enumerate(stock_prices):
        state  = state_creator(state_creator_param, timestep, window_size + 1, scalers)    
        state  = torch.tensor(state).float().to(DEVICE)
        
        with torch.no_grad(): 
            action = model(state)
        
        action = np.argmax(action[0].cpu().numpy())
        reward = 0

        if action == 1: # Buying
            portfolio.append(current_stock_price)
            
        elif action == 2 and len(portfolio) > 0: # Selling
            buy_price     = portfolio.pop(0)
            profit        = current_stock_price - buy_price
            reward        = max(profit, 0)
            total_profit += profit
            account_value += profit
        
        rewards.append(reward)
        actions.append(action)
        account_values.append(account_value)

    return rewards, total_profit, actions, account_values

def draw_points(series, actions, data_type = "validation", savefig = None):
    plt.figure(figsize = (18, 9))
    plt.plot(series, color = 'green')
    markers = ["", "d", "o"]   
    colors  = ["", "blue", "red"] # blue -> buy, red -> sell
    labels  = ["", "Buy Point", "Sell Point"]

    cnt = set()
    for i, _ in enumerate(actions):
        if actions[i] == 0: continue

        # BUG: if two actions are the same, one after the other, the legend is badly generated: RESOLVED
        if actions[i] not in cnt:
            plt.scatter([i], [series[i]], color = colors[actions[i]], label = labels[actions[i]], marker = markers[actions[i]])
            cnt.add(actions[i])
        else:
            plt.scatter([i], [series[i]], color = colors[actions[i]], marker = markers[actions[i]])

    plt.title(f"Buying and Selling Points on the {data_type} set")
    plt.legend(loc = 'best')
    
    if savefig != None:
        plt.savefig(savefig)

    plt.show()
    
def pyfolio_backtesting(profits, df_original, data_type = "validation"):
    def get_daily_return(df):
        df['daily_return']=df.account_value.pct_change(1)
        #df=df.dropna()
        print('Sharpe: ',(252**0.5)*df['daily_return'].mean()/ df['daily_return'].std())

        return df

    # def backtest_strat(df):
    #     strategy_ret= df.copy()
    #     strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'], errors='coerce')
    #     strategy_ret.set_index('Date', drop = False, inplace = True)
    #     # strategy_ret.index = strategy_ret.index.tz_localize('UTC')
    #     del strategy_ret['Date']
    #     ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)

    #     return ts

    df_account_value = pd.DataFrame(profits, columns=["account_value"])

    PATH_TO_RESULTS = f"results/{USER}/stage-{STAGE}/model-{MODEL}"
    os.makedirs(PATH_TO_RESULTS, exist_ok=True)

    df_stocks = df_original.copy()
    df_stocks = df_stocks.reset_index(drop=True)
    df_stocks['daily_return'] = df_stocks['Adj_Close'].pct_change(1)

    df_account_value = get_daily_return(df_account_value)
    df_account_value['Date'] = df_stocks['Date']
    
    # stocks_strat = backtest_strat(df_stocks)
    # account_value_strat = backtest_strat(df_account_value)

    df_stocks.to_csv(os.path.join(PATH_TO_RESULTS, f'df_stocks_{data_type}.csv'), index=False)
    df_account_value.to_csv(os.path.join(PATH_TO_RESULTS, f'df_account_value_{data_type}.csv'), index=False)

    # try:
    #     # backtest = pyfolio.create_full_tear_sheet(returns=account_value_strat,
    #     #                                 benchmark_rets=stocks_strat, set_context=False)
    #     # print(backtest)

    #     with pyfolio.plotting.plotting_context(font_scale=1.1):
    #         pyfolio.create_full_tear_sheet(returns=account_value_strat,
    #                                         benchmark_rets=stocks_strat, set_context=False)
    # except Exception as err:
    #     #TODO: install pyfolio in a newly created env, with
    #     # pip install git+https://github.com/quantopian/pyfolio
    #     print(err)


if __name__ == "__main__":
    # STAGE = 0
    # USER  = "andreig"
    MODEL   = 2

    model = get_estimator(CFG)

    try:
        PATH_TO_MODEL = f'models/{USER}/stage-{STAGE}/model-{MODEL}/best.pth'
        model.load_state_dict(torch.load(PATH_TO_MODEL)['model'])
    except OSError as e:
        EPISODE = 999
        PROFIT  = 60.379
        REWARD  = 0.503
        PATH_TO_MODEL = f'models/{USER}/stage-{STAGE}/model-{MODEL}/model_{MODEL}_episode_{EPISODE}_profit_{PROFIT}_reward_{REWARD}.pth'
        model.load_state_dict(torch.load(PATH_TO_MODEL)['model'])

    model.eval()

    data       = pd.read_csv(PATH_TO_DATA)

    valid_df = data[data['Split'] == 1]
    test_df = data[data['Split'] == 2]

    data      = wrap(data)

    features = list(CFG["features_used"].keys())
    normalizers = list(CFG["features_used"].values())

    train_data = [data[data['split'] == 0][features].fillna(0).values.tolist(), data[data['split'] == 0][CFG["target_used"]].fillna(0).tolist()]
    valid_data = [data[data['split'] == 1][features].fillna(0).values.tolist(), data[data['split'] == 1][CFG["target_used"]].fillna(0).tolist()]
    test_data = [data[data['split'] == 2][features].fillna(0).values.tolist(), data[data['split'] == 2][CFG["target_used"]].fillna(0).tolist()]

    # valid_data = data[data['Split'] == 1]["Adj_Close"].tolist()
    # test_data  = data[data['Split'] == 2]["Adj_Close"].tolist()

    scalers = None
    # Train MinMaxScaler on data if it will be used in normalizer
    if 'minmax' in normalizers:
        scalers = calculate_scalers(normalizers=normalizers, features=features, train_data=train_data)
    
    valid_rewards, valid_profit, valid_actions, valid_account_values = test_fn(model, valid_data, CFG['window_size'], scalers)
    test_rewards,  test_profit,  test_actions, test_account_values  = test_fn(model, test_data,  CFG['window_size'], scalers)
    print(f"Finished with valid_profit = {valid_profit} and test_profit = {test_profit}")

    pyfolio_backtesting(valid_account_values, valid_df, data_type="validation")
    pyfolio_backtesting(test_account_values,  test_df,   data_type="test")

    # valid_actions = [0] * CFG['window_size'] + valid_actions[: -CFG['window_size']]
    # test_actions  = [0] * CFG['window_size'] + test_actions[: -CFG['window_size']]

    path_to_images = f"images/{USER}/stage-{STAGE}"
    os.makedirs(path_to_images, exist_ok=True)
    
    draw_points(valid_data[1], valid_actions, data_type = "validation", savefig = f"{path_to_images}/valid_actions_model_{MODEL}.png")
    draw_points(test_data[1],  test_actions,  data_type = "test"      , savefig = f"{path_to_images}/test_actions_model_{MODEL}.png")