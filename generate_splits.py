from common import *

plt.style.use(["ggplot"])
if __name__ == "__main__":
    COMPANY  = "AAPL"       # In the future we might change the company
    SAVE_FIG =  True        # Saving fig for presentation in images/ folder
    COLUMN   = "Adj_Close"  # We will save all features (because some libraries might need the), 
                            # But the main feature will be switched from "Close" to "Adj_Close"

    data = datareader.DataReader('AAPL', data_source = "yahoo")
    data = data.rename(columns = {"Adj Close" : "Adj_Close"})
    first_day = str(data.index.values[0]).split("T")[0]
    last_day  = str(data.index.values[-1]).split("T")[0]

    data["Split"] = -1
    train_idx = int(len(data) * 0.7)
    valid_idx = int(len(data) * 0.9)

    pd.options.mode.chained_assignment        = None
    data["Split"].iloc[0         : train_idx] = 0
    data["Split"].iloc[train_idx : valid_idx] = 1
    data["Split"].iloc[valid_idx :          ] = 2

    data.to_csv(f"data/{COMPANY}_stocks_splits.csv")

    styles = ['', '', '']
    colors = {0: 'green', 1: 'orange', 2: 'red'};

    plt.figure(figsize = (10, 5))
    for idx, grp in data.groupby("Split"):
        grp[COLUMN].plot(style = styles[idx], color = colors[idx])

    plt.title(
        f'[{COMPANY}] {COLUMN} From {first_day} to {last_day}'
    )
    plt.axvline(x = data.index.values[train_idx], color = 'blue')
    plt.axvline(x = data.index.values[valid_idx], color = 'blue')
    plt.xlabel("Timestamp")
    plt.ylabel(f"{COLUMN} Value")
    if SAVE_FIG: 
        plt.savefig(f"images/AAPL_{COLUMN}_Splits.png")
    else:
        plt.show()