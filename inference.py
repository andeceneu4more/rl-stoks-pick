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

    cnt = 0
    for i, _ in enumerate(actions):
        if actions[i] == 0: continue
        if cnt < 2:
            plt.scatter([i], [series[i]], color = colors[actions[i]], label = labels[actions[i]], marker = markers[actions[i]])
            cnt += 1
        else:
            plt.scatter([i], [series[i]], color = colors[actions[i]], marker = markers[actions[i]])

    plt.title(f"Buying and Selling Points on the {data_type} set")
    plt.legend(loc = 'best')
    
    if savefig != None:
        plt.savefig(savefig)

    plt.show()
    
if __name__ == "__main__":
    STAGE = 0
    MODEL = 0
    PATH_TO_MODEL = f'models/stage-{STAGE}/model-{MODEL}/model_{MODEL}_episode_999_profit_157.025_reward_1.109.pth'

    model = get_estimator(CFG)
    model.load_state_dict(torch.load(PATH_TO_MODEL)['model'])
    model.eval()

    data       = pd.read_csv(PATH_TO_DATA)
    valid_data = data[data['Split'] == 1]["Adj_Close"].tolist()
    test_data  = data[data['Split'] == 2]["Adj_Close"].tolist()
    
    valid_rewards, valid_profit, valid_actions = test_fn(model, valid_data, CFG['window_size'])
    test_rewards,  test_profit,  test_actions  = test_fn(model, test_data,  CFG['window_size'])

    # valid_actions = [0] * CFG['window_size'] + valid_actions[: -CFG['window_size']]
    # test_actions  = [0] * CFG['window_size'] + test_actions[: -CFG['window_size']]

    draw_points(valid_data, valid_actions, data_type = "validation", savefig = f"images/valid_actions_stage_{STAGE}_model_{MODEL}.png")
    draw_points(test_data,  test_actions,  data_type = "test"      , savefig = f"images/test_actions_stage_{STAGE}_model_{MODEL}.png")