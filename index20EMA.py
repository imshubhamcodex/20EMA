import pandas as pd
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt
from tabulate import tabulate
from datetime import datetime, timedelta

# Load and preprocess your data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[:, :5]
    df = df.dropna()
    return df
    
    return fetch_data()


# Get latest data
def fetch_data():
    ticker = "^NSEI"
    time_interval = "5m"
    
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=7)
    data = yf.download(ticker, start=start_date, end=end_date, interval=time_interval)
    data.rename(columns={'Open': 'open', 'Close': 'close','High' : 'high', 'Low' : 'low'}, inplace=True)
    data['date'] = data.index

    return data



# Calculate EMA, Slope, BB, RSI
def calculate_ema_slope_bb_rsi(data):
    data['20_MA'] = ta.ema(data['close'], length=20)
    data['20_MA_Slope'] = (data['20_MA'] - data['20_MA'].shift(3)) / 3
    data['20_U_BB'] = ta.bbands(data['close'], length=20, std=2)['BBU_20_2.0']
    data['20_RSI'] = ta.rsi(data['close'], length=20)
    data = data.dropna(subset=['20_MA'])
    data.reset_index(drop=True, inplace=True)
    return data




# Find candles that is near EMA
def find_candles(data):
    return data[((data['open'] > data['20_MA'] - 5) 
                 & (data['open'] < data['20_MA'] + 10)) & (data['20_MA_Slope'] >= 0) 
                    & (data['20_RSI'] >= 48)  & (data['high'] < data['20_U_BB'] - 1) 
                        & (abs(data['open'] - data['close']) >= 0.2)]



# Backtesting strategy
def backtest_strategy(data, marked_candles):
    positions = []
    profit_points = []
    exit_date = None
    entry_date = None
    total_profit = 0
    overshoot = 0
    drawdown = 0
    
    candles_on_same_date = 0
    prev_date = None
    
    for i in range(len(marked_candles)):
        profit = 0
        
        entry_index = data.index[marked_candles.index[i]]
        entry_date = data['date'][entry_index]
        entry_price = data['close'][entry_index]
        entry_date = datetime.fromisoformat(str(entry_date))
        
        # if exit_date != None and datetime.fromisoformat(exit_date) > datetime.fromisoformat(entry_date):
        #     continue
        
        
        # Allow only 20 signal each day 
        if prev_date == entry_date.date():
            candles_on_same_date += 1
            prev_date = entry_date.date()
        else:
            candles_on_same_date = 0
            prev_date = entry_date.date()
        
        if candles_on_same_date >= 21:
            continue
        
            
        stop_loss_price = entry_price - 50    
        take_profit_price = entry_price + 80
        
        
        sl = round(abs(entry_price - stop_loss_price),2)
        tp = round(abs(entry_price - take_profit_price),2)
        
        
        for j in range(entry_index + 1, len(data)):
            
            if data['low'][j] <= stop_loss_price:
                exit_date = data['date'][j]
                exit_price = stop_loss_price
                status = "loss"
                profit = -abs(entry_price - exit_price)
                overshoot = 0
                drawdown += abs(profit) 
                break
                
            elif data['high'][j] >= take_profit_price:
                exit_date = data['date'][j]
                exit_price = take_profit_price
                status = "profit"
                profit = abs(entry_price - exit_price)
                overshoot += profit
                drawdown = 0
                break
        
        total_profit += profit
        
        if profit == 0:
            overshoot = 0
            drawdown = 0
        
        positions.append({
                "Entry Date": entry_date,
                "Exit Date": exit_date,
                "Entry Price": round(entry_price,2),
                "Exit Price": round(exit_price,2),
                "Status": status,
                "SL Price" : round(stop_loss_price,2),
                "TP Price" : round(take_profit_price,2),
                "SL" : sl,
                "TP" : tp,
                "Pts. Gain": profit,
                "Cum. Profit" : total_profit
            })
        
        profit_points.append((profit, total_profit, overshoot, drawdown))
            
    return positions, profit_points



# Printing data
def print_matrix(positions, profit_points):
    win_trade = 0
    loss_trade = 0

    for position in positions:
        if position["Status"] == "loss":
            loss_trade += 1
        elif position["Status"] == "profit":
            win_trade += 1
            
    win_per = round(win_trade*100/(win_trade + loss_trade),2)
    print("\nAssest Nifty50 @ 5min")
    print("Total Win Trade:", win_trade)
    print("Total Loss Trade:", loss_trade)
    print("Total Trades:", win_trade + loss_trade)
    print("Win %:", win_per)
    print("Take Profit Points:", 80)
    print("Stop Loss Points:", 50)
    print("Total Profit Points:", profit_points[-1][1])
    print("Maximum Signal on same day:", 20)
    
    
    headers = positions[0].keys()
    data_values = [[entry[key] for key in headers] for entry in positions]
    table = tabulate(data_values, headers, tablefmt="grid")

    file_path = "backtest.txt"

    # Save the table to a text file
    with open(file_path, "w") as file:
        file.write(table)




# Plot profit points chart 
def plot_chart(profit_points):
    data = []
    overshoot_data = []
    drawdown_data = []
    
    for profit_point, total_profit, overshoot, drawdown in profit_points:
        data.append(total_profit)
        overshoot_data.append(overshoot)
        drawdown_data.append(drawdown)
        
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(data)), data, label='Profit Points')
    plt.plot(range(len(data)), drawdown_data, label='Drawdown')
    plt.plot(range(len(data)), overshoot_data, label='Overshoot')
    plt.xlabel('Numbers of Trade')
    plt.ylabel('Total gain Points')
    plt.title('Trade Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# Entry
def main():
    pd.set_option('display.max_rows', None)
    pd.options.mode.chained_assignment = None
    
    data = load_and_preprocess_data("NIFTY50-5.csv")
    data = calculate_ema_slope_bb_rsi(data)
    candles_at_20ma = find_candles(data)
    
    candles_at_20ma['close - open'] = candles_at_20ma['close'] - candles_at_20ma['open']
    
    print("\nTotal Event Occured : ",candles_at_20ma['date'].count())
    print("Total Positive close - open : " , candles_at_20ma[candles_at_20ma['close - open'] > 0]['date'].count())
    print("Total Negative close - open : " ,candles_at_20ma[candles_at_20ma['close - open'] < 0]['date'].count())

    positions, profit_points = backtest_strategy(data, candles_at_20ma)
    print_matrix(positions, profit_points)
    plot_chart(profit_points)

if __name__ == "__main__":
    main()