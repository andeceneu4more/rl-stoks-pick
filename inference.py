from common  import *
from models  import BaseEstimator
from trader  import valid_fn, state_creator, CFG
from getters import get_estimator 

def test_fn(model, test_data, window_size):
    total_profit = 0
    portfolio, rewards = [], []    
    actions = []

    for timestep, current_stock_price in enumerate(test_data):
        state  = state_creator(test_data, timestep, window_size + 1)    
        state  = torch.tensor(state).float().to(DEVICE)
        
        with torch.no_grad(): 
            action = model(state)
        
        action = np.argmax(action[0].cpu().numpy())
        reward = 0

        if action == 1: 
            portfolio.append(current_stock_price)
            
        elif action == 2 and len(portfolio) > 0:
            buy_price     = portfolio.pop(0)
            profit        = current_stock_price - buy_price
            reward        = max(profit, 0)
            total_profit += profit
        
        rewards.append(reward)
        actions.append(action)

    return rewards, total_profit, actions

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
    
if __name__ == "__main__":
    # STAGE = 0
    # USER  = "andreig"
    MODEL   = 6

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
    valid_data = data[data['Split'] == 1]["Adj_Close"].tolist()
    test_data  = data[data['Split'] == 2]["Adj_Close"].tolist()
    
    valid_rewards, valid_profit, valid_actions = test_fn(model, valid_data, CFG['window_size'])
    test_rewards,  test_profit,  test_actions  = test_fn(model, test_data,  CFG['window_size'])

    # valid_actions = [0] * CFG['window_size'] + valid_actions[: -CFG['window_size']]
    # test_actions  = [0] * CFG['window_size'] + test_actions[: -CFG['window_size']]

    path_to_images = f"images/{USER}/stage-{STAGE}"
    os.makedirs(path_to_images, exist_ok=True)

    draw_points(valid_data, valid_actions, data_type = "validation", savefig = f"{path_to_images}/valid_actions_model_{MODEL}.png")
    draw_points(test_data,  test_actions,  data_type = "test"      , savefig = f"{path_to_images}/test_actions_model_{MODEL}.png")