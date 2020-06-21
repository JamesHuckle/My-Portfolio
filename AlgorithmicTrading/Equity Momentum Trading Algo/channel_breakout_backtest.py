import bs4 as bs
from datetime import datetime
import numpy as np
import os
import csv
import sys
import pandas as pd
import requests
import talib
import matplotlib.pyplot as plt
import yfinance as yf 
from IPython.display import display
import seaborn as sns

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 50

from plotly import subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)         # initiate notebook for offline plot
from plotly.graph_objs import *
import plotly.graph_objs as go


def save_tickers(index):
    if index == "sp500":
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        names = []
        for row in table.findAll('tr')[1:]:
            name = row.findAll('td')[1].text.strip('\n')
            names.append(name)
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip('\n')
            tickers.append(ticker)
        return dict(zip(tickers,names))

    elif index == 'ftse100':
        resp = requests.get('https://en.wikipedia.org/wiki/FTSE_100_Index')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        names = []
        for row in table.findAll('tr')[1:]:
            name = row.findAll('td')[0].text.strip('\n')
            names.append(name)
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[1].text.strip('\n')
            tickers.append(ticker)
        final_tickers = []
        for ticker in tickers:
            ticker = ticker.replace('.L',"")
            ticker = ticker.replace('.',"")
            ticker = f"{ticker}.L"
            final_tickers.append(ticker)
        return dict(zip(final_tickers,names))

    elif index == 'russell2000':
        tickers = pd.read_csv('https://www.ishares.com/uk/professional/en/products/239710/' +
                            'ishares-russell-2000-etf/1506575576011.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund',
                            skiprows=2)[['Issuer Ticker','Name']]
        tickers.set_index('Issuer Ticker',drop=True,inplace=True)
        return tickers.to_dict()['Name']
##save_tickers('sp500')

def stock_outlier_filter(all_stock_data):
    outlier_stocks = []
    for stock in all_stock_data['Adj Close'].columns:
        adj_close = all_stock_data['Adj Close'][stock]
        if adj_close.min() <= 0 or adj_close.max() == np.inf or adj_close.pct_change().max() > 300:
            outlier_stocks.append(stock)
        else:
            diff = all_stock_data['Close'][stock] - adj_close
            for col in ['Open','High','Low','Close']:
                all_stock_data[col][stock] = all_stock_data[col][stock] - diff
        print('outlier stock data',stock)
    all_stock_data.drop(outlier_stocks, level=1, axis='columns', inplace=True)
    print('finished all outliers')
    return all_stock_data


def download_data_local_check(name, tickers, start, end):
    data_exist = f"./{name}_all_stock_data_{str(datetime.now().date())}.csv"
    if os.path.exists(data_exist):
        all_stock_data = pd.read_csv(data_exist,header=[0,1])
        all_stock_data = all_stock_data[1:]
        all_stock_data.set_index(all_stock_data['Unnamed: 0_level_0']['Unnamed: 0_level_1'], drop=True, inplace=True)
        all_stock_data.index.name = 'Date'
        all_stock_data.index = pd.to_datetime(all_stock_data.index)
        all_stock_data.drop(('Unnamed: 0_level_0','Unnamed: 0_level_1'), axis='columns', inplace=True)
    else:
        all_stock_data = yf.download(tickers, start, end)
        all_stock_data = all_stock_data.ffill()
        all_stock_data = stock_outlier_filter(all_stock_data)
        all_stock_data.to_csv(data_exist)
    tickers = list(all_stock_data['Adj Close'].columns)
    return all_stock_data, tickers
# start = '2000-01-01'
# end = '2019-08-8'
# sp500_tickers = sorted(list(save_tickers('sp500').keys()))
# download_data_local_check('SP500', sp500_tickers, start, end)

def filter_for_stock(all_stock_data, stock):
    cols = ['Open','High','Low','Close']
    stock_data = pd.DataFrame()
    for col in cols:
        stock_data[col] = all_stock_data[col][stock]
    stock_data.dropna(inplace=True)
    return stock_data
#stock_data = filter_for_stock(all_stock_data, 'AAPL')

def transform_data(stock_data):
    data_points = ['Open','High','Low','Close']
    master = stock_data[data_points].copy()
    master.columns = [col.lower() for col in master.columns]
    master['timestamp'] = master.index
    master.index.names = ['timestamp']
    return master
# master = transform_data(stock_data)
# data = master.copy()


def calc_atr(data, atr_len):
    data["atr"] = talib.ATR(data["high"].values,data["low"].values,data["close"].values,timeperiod=atr_len)
    data = data.dropna(axis="rows")
    data = data.reset_index(drop=True)
    data.head()
# calc_atr(data)
# data.head()


def calc_atr_trail(data):
    atr_size = 10
    data["long_trail"] = data["high"] - (data["atr"]*atr_size)
    data["short_trail"] = data["low"] + (data["atr"]*atr_size) 
    data[["long_trail","short_trail"]] = data[["long_trail","short_trail"]].shift(1)
    data = data.dropna()
    data = data.reset_index(drop=True)
# calc_atr_trail(data)
# data.head()

def calc_sharpe(data, lookback_days):
    returns = np.log10(1 + (data['close'].pct_change()))
    roll_std = returns.rolling(lookback_days, lookback_days).std()
    roll_avg = returns.rolling(lookback_days, lookback_days).mean()
    data['sharpe'] = (roll_avg / roll_std) / 252**0.5
    data['sharpe'] = data['sharpe'] * 100


def calc_high_low_chan(data, lookback_days):
    data["high_chan"] = data["high"].rolling(window=lookback_days,min_periods=lookback_days).max().shift(1)
    data["low_chan"] = data["low"].rolling(window=lookback_days,min_periods=lookback_days).min().shift(1)


def backtest_donchain_breakout(data, lookback_days, sharpe_filter, atr_trailing_stop=False, stop_and_target={}):
    calc_high_low_chan(data, lookback_days)
    calc_sharpe(data, lookback_days)
    trailing_stop = atr_trailing_stop
    
    trades = []
    buy_trade = {}
    sell_trade = {}
    long = False
    short = False
    long_hit_target_or_stop = False
    short_hit_target_or_stop = False
    get = data.at ## just a shortcut to the data.get_value() function
    error = 0
    #data = data[data["timestamp"]>datetime.datetime(2018,11,16)]
    for idx in data.index:   
        try:      
            ## MFE, MAE
            if long == True:
                mfe_long_price = get[idx,"high"]
                mfe_long = mfe_long_price - buy_trade['entry']
                if mfe_long > buy_trade["mfe"]:
                    buy_trade["mfe"] = mfe_long
                    buy_trade["mfe_price"] = mfe_long_price

                mae_long_price = get[idx,"low"]
                mae_long = buy_trade['entry'] - mae_long_price
                if mae_long > buy_trade["mae"]:
                    buy_trade["mae"] = mae_long
                    buy_trade["mae_price"] = mae_long_price

            elif short == True:
                mfe_short_price = get[idx,'low'] 
                mfe_short = sell_trade["entry"] - mfe_short_price
                if mfe_short > sell_trade["mfe"]:
                    sell_trade["mfe"] = mfe_short
                    sell_trade["mfe_price"] = mfe_short_price

                mae_short_price = get[idx,'high'] 
                mae_short = mae_short_price - sell_trade["entry"]
                if mae_short > sell_trade["mae"]:
                    sell_trade["mae"] = mae_short
                    sell_trade["mae_price"] = mae_short_price

            ### channel trading only
            if (get[idx,"high"] > get[idx,"high_chan"]) and (get[idx,"low"] < get[idx,"low_chan"]):
                #print("trade broke both high and low channel, not sure!!!",get[idx,'timestamp'],
                #      'high',get[idx,'high'],'high_chan',get[idx,'high_chan'],
                #      'low',get[idx,'low'],'low_chan',get[idx,'low_chan'])
                continue
            
            elif get[idx,"high_chan"] - get[idx,"low_chan"] == 0:
                #print('channel size is zero, skipping!!')
                continue
            
            ## trade entry?
            elif get[idx,"high"] > get[idx,"high_chan"] and long == False:
                if sharpe_filter is None or get[idx,'sharpe'] >= sharpe_filter:
                    buy_trade["entry_time"] = get[idx,"timestamp"]
                    buy_trade["entry"] = get[idx,'close'] #max(get[idx,"high_chan"], get[idx,"open"])
                    buy_trade["direction"] = "long"
                    buy_trade['sharpe'] = get[idx,'sharpe']
                    buy_trade["mfe"] = 0
                    buy_trade['mae'] = 0
                    buy_trade["mfe_price"] = 0
                    buy_trade["band_size"] = get[idx,"high_chan"] - get[idx,"low_chan"]
                    long = True
                    #print("long")
                if short == True:
                    if short_hit_target_or_stop == False:
                        #print("close short")
                        sell_trade["exit_time"] = get[idx,"timestamp"]
                        sell_trade["exit"] = get[idx,'close'] #max(get[idx,"high_chan"], get[idx,"open"])
                        sell_trade["exit_type"] = "channel"
                    sell_trade["profit"] = sell_trade["entry"] - sell_trade["exit"]
                    trades.append(sell_trade)
                    sell_trade = {}
                    short = False
                    short_hit_target_or_stop = False
                continue

            elif get[idx,"low"] < get[idx,"low_chan"] and short == False:
                if sharpe_filter is None or get[idx,'sharpe'] <= -sharpe_filter:
                    sell_trade["entry_time"] = get[idx,"timestamp"]
                    sell_trade["entry"] = get[idx,'close'] #min(get[idx,"low_chan"], get[idx,"open"])
                    sell_trade["direction"] = "short"
                    sell_trade['sharpe'] = get[idx,'sharpe']
                    sell_trade["mfe"] = 0
                    sell_trade['mae'] = 0
                    sell_trade["mfe_price"] = 0
                    sell_trade["band_size"] = get[idx,"high_chan"] - get[idx,"low_chan"]
                    short = True
                    #print("short")
                if long == True:
                    if long_hit_target_or_stop == False:
                        #print("close_long")
                        buy_trade["exit_time"] = get[idx,"timestamp"]
                        buy_trade["exit"] = get[idx,'close'] #min(get[idx,"low_chan"], get[idx,"open"])
                        buy_trade["exit_type"] = "channel"
                    buy_trade["profit"] = buy_trade["exit"] - buy_trade["entry"]
                    trades.append(buy_trade)
                    buy_trade = {}
                    long = False
                    long_hit_target_or_stop = False
                continue    

            ### trailing stop loss
            if trailing_stop or stop_and_target:
                if trailing_stop == True:
                    long_stop = get[idx,"long_trail"]
                    short_stop = get[idx,"short_trail"]
                if long == True and long_hit_target_or_stop == False:
                    if stop_and_target:
                        long_stop = buy_trade['entry'] - buy_trade['band_size'] * stop_and_target['stop']
                        long_target = buy_trade['entry'] + buy_trade['band_size'] * stop_and_target['target']
                        buy_trade['target'] = long_target
                        buy_trade['stop'] = long_stop
                        if get[idx,"high"] > long_target:
                            buy_trade["exit_time"] = get[idx,"timestamp"]
                            buy_trade["exit"] = get[idx,'close'] #max(get[idx,'open'], long_target)
                            buy_trade["exit_type"] = "target"
                            long_hit_target_or_stop = True
                    if get[idx,"low"] < long_stop:
                        buy_trade["exit_time"] = get[idx,"timestamp"]
                        buy_trade["exit"] = get[idx,'close'] #min(get[idx,'open'], long_stop)
                        buy_trade["exit_type"] = "stop"
                        long_hit_target_or_stop = True

                elif short == True and short_hit_target_or_stop == False:
                    if stop_and_target:
                        short_stop = sell_trade['entry'] + sell_trade['band_size'] * stop_and_target['stop']
                        short_target = sell_trade['entry'] - sell_trade['band_size'] * stop_and_target['target']
                        sell_trade['target'] = short_target
                        sell_trade['stop'] = short_stop
                        if get[idx,"low"] < short_target:
                            sell_trade["exit_time"] = get[idx,"timestamp"]
                            sell_trade["exit"] = get[idx,'close'] #min(get[idx,'open'], short_target)
                            sell_trade["exit_type"] = "target"
                            short_hit_target_or_stop = True  
                    if get[idx,"high"] > short_stop:
                        sell_trade["exit_time"] = get[idx,"timestamp"]
                        sell_trade["exit"] = get[idx,'close'] #max(get[idx,'open'], short_stop)
                        sell_trade["exit_type"] = "stop"
                        short_hit_target_or_stop = True    

        except Exception as e:
            error+=1
            print('===',e)
            print(data.loc[idx])
            print(long,'buy_trade',buy_trade)
            print(short,'sell_trade',sell_trade)
            if error >20:
                raise e
            continue

    if len(buy_trade) > 0:
        buy_trade["exit_time"] = get[idx,"timestamp"]
        buy_trade["exit"] = get[idx,"close"]
        buy_trade["exit_type"] = "channel"
        buy_trade["profit"] = buy_trade["exit"] - buy_trade["entry"]
        trades.append(buy_trade)
        buy_trade = {}
        long = False
    if len(sell_trade) > 0:
        sell_trade["exit_time"] = get[idx,"timestamp"]
        sell_trade["exit"] = get[idx,"close"]
        sell_trade["exit_type"] = "channel"
        sell_trade["profit"] = sell_trade["entry"] - sell_trade["exit"]
        trades.append(sell_trade)
        sell_trade = {}
        short = False
        
    return pd.DataFrame(trades)     
# master_trades = backtest_donchain_breakout(data, lookback_days=9, atr_trailing_stop=False)
# trades = master_trades.copy()


def calculate_stats(stock, trades, risk_per_trade_perc, comms_rt_perc, daily_interest_cost, print_output=True):
    trades["profit%"] = (trades["profit"] / trades["entry"])*100
    trades['days_in_trade'] = (trades['exit_time'] - trades['entry_time']).dt.days
    trades['comms%'] = (trades['days_in_trade'] * daily_interest_cost) + comms_rt_perc
    trades['profit%_of_cap'] = (trades['profit'] / trades['band_size']) * risk_per_trade_perc
    trades["mfe_of_band"] = trades["mfe"] / trades['band_size'] * risk_per_trade_perc
    trades["mae_of_band"] = trades["mae"] / trades['band_size'] * risk_per_trade_perc
    trades['net_profit%_of_cap'] = (trades['profit'] - (trades['entry'] * trades['comms%'] / 100)) / trades['band_size']

    trades["running_profit"] = trades["net_profit%_of_cap"].cumsum()
    trades["min_equity"] = trades[::-1]["running_profit"].cummin()
    trades["dd"] = trades["running_profit"] - trades["min_equity"]

    total_profit = trades["net_profit%_of_cap"].sum().round()
    max_dd = round(trades["dd"].max())
    dd_per = round((max_dd / total_profit),2) if total_profit != 0 else 0
    type_of_trades = trades["exit_type"].value_counts().to_dict()
    num_trades = trades["exit_type"].notnull().sum()
    wins = (trades["net_profit%_of_cap"]>0).sum()
    loses = (trades["net_profit%_of_cap"]<=0).sum()
    win_rate = round(wins/(wins+loses)*100,2)

    if print_output == True:
        print(type_of_trades)
        print(num_trades,"# trades")
        print(wins,':# wins')
        print(loses,':# loses')
        print(win_rate,'% :win rate')
        print()
        print(round(trades["profit%_of_cap"].mean(),2),"% :avg profit per trade before fees and interest")
        print(round(trades["net_profit%_of_cap"].mean(),2),"% :avg profit per trade after fees and interest")
        print(round(trades["net_profit%_of_cap"].max(),2),"% :max profit")
        print(round(trades["net_profit%_of_cap"].min(),2),"% :max loss")
        print()
        print(total_profit,"% :total profit")
        print(max_dd, "% :max_dd")
        print(dd_per, ": dd_ratio")
        trades.plot(x="entry_time",y="running_profit",title="Equity curve(%)")
        trades.plot(x="entry_time",y="net_profit%_of_cap",title="net profit each trade (%)")
        trades.plot(x="entry_time",y="band_size")
        plt.axhline(0,color="k")
        
    target_type = type_of_trades.get('target',np.nan)
    stop_type = type_of_trades.get('stop',np.nan)
    channel_type = type_of_trades.get('channel',np.nan)
    return {'stock':stock, 'target':target_type, 'channel':channel_type, 'stop':stop_type, 'num_trades':num_trades, 'win_rate':win_rate,
            'total_profit':total_profit, 'max_dd':max_dd, 'dd_per':dd_per}
# daily_interest_cost = 2 / 365
# comms_rt_perc = 0.16
# risk_per_trade_perc = 1
# print_output = False
# stats = calculate_stats(stock, trades, risk_per_trade_perc, comms_rt_perc, daily_interest_cost, print_output=False)


def band_size_vs_profitability(trades, print_output=True):
    trades["band_size_buckets"] = pd.cut(trades["band_size"],bins=20)
    band_sizes = trades[["band_size_buckets","net_profit%_of_cap","mfe_of_band"]].groupby("band_size_buckets").agg({"net_profit%_of_cap":"sum","mfe_of_band":"mean"})
    if print_output == True:
        display(band_sizes)
        trades.plot.scatter("net_profit%_of_cap","band_size")
        plt.axvline(0,color="k")
        band_sizes.plot.barh(y="net_profit%_of_cap")
        band_sizes.plot.barh(y="mfe_of_band")

    return band_sizes
# band_sizes = band_size_vs_profitability(trades, print_output=False)
# band_sizes

def run_optimization(trades, print_output, non_fixed_profit, column="mfe_of_band"):
    target_max = int(trades[column].max())
    results = {}
    for target in [x / 4 for x in range(1, target_max * 4 + 6)]:
        stop_or_target = 1 if 'mfe' in column else -1
        hit_target = (trades[column] >= target).sum() * stop_or_target * target ## total profit from hitting targets
        missed_target = trades[trades[column] < target]["net_profit%_of_cap"].sum() # total profit from trades that missed target
        #print(target,'=== #hit_target profit:',hit_target,'| missed_target profit:',missed_target,'| total profit',hit_target + missed_target)
        results[target] = hit_target + missed_target
    results = pd.Series(results)
    if results.empty:
        return np.nan
    best_target = results.idxmax()
    if print_output:
        print(best_target,"time band size is the best fixed target -->",column)
        results.plot()
        plt.ylabel("Total %")
        plt.axvline(best_target,color="k")
        plt.axhline(non_fixed_profit,color="g")
        plt.show()
    return best_target

def target_optimization(stock, trades, non_fixed_profit, modify_trades_in_place, print_output=True, optimize_column='target'):  
    ## modify_trades_in_place: if target_optimization yields better results than no target, then modify the trades
    if print_output:
        #print(trades)
        trades.plot.scatter(x="net_profit%_of_cap",y="mfe_of_band")
        plt.show()
        ax = trades["net_profit%_of_cap"].plot.hist(22,figsize=(15,8),alpha=0.6)
        trades["mfe_of_band"].plot.hist(22,alpha=0.5,ax=ax)
        plt.legend()
        plt.show()
    
    best_target = run_optimization(trades, print_output, non_fixed_profit, column="mfe_of_band")
    best_stop = run_optimization(trades, print_output, non_fixed_profit, column="mae_of_band")  
    ## equity curve of best fixed target 
    if modify_trades_in_place:
        fixed_target_trades = trades
    else:
        fixed_target_trades = trades.copy()
    hit_target = []
    if optimize_column == 'target' and best_target is not None:
        hit_target = trades["mfe_of_band"] >= best_target
        fixed_target_trades.loc[hit_target,"net_profit%_of_cap"] = best_target
    elif optimize_column == 'stop' and best_stop is not None:
        hit_target = trades["mae_of_band"] >= best_stop
        fixed_target_trades.loc[hit_target,"net_profit%_of_cap"] = -best_stop

    if len(hit_target) > 0:
        ## standardised
        # trades["contracts"] = 10000/trades["entry"]
        # trades["net_profit_standard"] = trades["net_profit"]*trades["contracts"]
        # trades["running_profit"] = trades["net_profit_standard"].cumsum()
        ## nothing
        fixed_target_trades["running_profit"] = fixed_target_trades["net_profit%_of_cap"].cumsum()
        #####
        fixed_target_trades["min_equity"] = fixed_target_trades[::-1]["running_profit"].cummin()
        fixed_target_trades["dd"] = fixed_target_trades["running_profit"] - fixed_target_trades["min_equity"]
        fixed_target_max_dd = round(fixed_target_trades["dd"].max())
        fixed_target_profit = round(fixed_target_trades["running_profit"].values[-1])
        fixed_target_dd_per = round((fixed_target_max_dd / fixed_target_profit),2) if fixed_target_profit != 0 else 0
        fixed_target_wins = (fixed_target_trades["net_profit%_of_cap"]>0).sum()
        fixed_target_loses = (fixed_target_trades["net_profit%_of_cap"]<=0).sum()
        fixed_target_win_rate = round(fixed_target_wins / (fixed_target_wins + fixed_target_loses)*100,2)
        
        if print_output:
            fixed_target_trades.plot(x="entry_time",y="running_profit")
            plt.show()
            print(fixed_target_wins,':# wins')
            print(fixed_target_loses,':# loses')
            print(fixed_target_win_rate,'% :win rate')
            print("fixed_target_max_dd(%):",fixed_target_max_dd)
            print("fixed_target_dd ratio:",fixed_target_dd_per)
            print("fixed_target_profit(%):",fixed_target_profit)
    else:
        return {'stock':stock, 'win_rate':np.nan, 'total_profit':np.nan,
            'max_dd':np.nan, 'dd_per':np.nan, 'best_target':np.nan, 'best_stop':np.nan}
    return {'stock':stock, 'win_rate':fixed_target_win_rate, 'total_profit':fixed_target_profit,
            'max_dd':fixed_target_max_dd, 'dd_per':fixed_target_dd_per, 'best_target':best_target, 'best_stop':best_stop}
# # if set to true, then the trades dataframe will be modified if a target is better
# modify_trades_in_place = False
# fixed_target_stats = target_optimization(trades, modify_trades_in_place, print_output=True)
# fixed_target_stats


def run_backtest(stock, stock_data, lookback_days, daily_interest_cost, comms_rt_perc, risk_per_trade_perc, modify_trades_in_place,
                 file_path, atr_trailing_stop, atr_len, reverse, delete_files, save_files, plot_chart, print_output, optimize_column,
                 stop_and_target, sharpe_filter):

    #check stock data
    if len(stock_data) == 0:
        print('no data for stock')
        return []

    # configure data
    master = transform_data(stock_data)
    data = master.copy()
    calc_atr(data, atr_len)
    calc_atr_trail(data)
    # run backtest
    master_trades = backtest_donchain_breakout(data, lookback_days, sharpe_filter, atr_trailing_stop, stop_and_target)
    trades = master_trades.copy()
    if reverse and len(trades) > 0:
        trades['direction'] = trades['direction'].map({'long':'short','short':'long'})
        trades.rename({'mfe':'mae','mae':'mfe','mfe_price':'mae_price','mae_price':'mfe_price'},axis='columns', inplace=True)
        trades['profit'] = -trades['profit']
    # calc stats
    if trades.empty:
        return []
    stats = calculate_stats(stock, trades, risk_per_trade_perc, comms_rt_perc, daily_interest_cost, print_output)
    non_fixed_profit = stats['total_profit']
    if optimize_column:
        fixed_target_stats = target_optimization(stock, trades, non_fixed_profit, modify_trades_in_place, print_output, optimize_column)
    # write to files
    trades['stock'] = stock
    if plot_chart:
        plot_charts(data, trades)
    if save_files:
        stats_file = f"./{file_path}/lookback_{lookback_days}_trailing_{atr_trailing_stop}_stats.csv"
        fixed_target_file = f"./{file_path}/lookback_{lookback_days}_trailing_{atr_trailing_stop}_fixed_target_stats.csv"
        if optimize_column:
            file_looper = {stats_file:stats, fixed_target_file:fixed_target_stats}
        else:
            file_looper = {stats_file:stats}
        for file_name, stats in file_looper.items():
            if delete_files:
                open(file_name, 'w', newline="").close()
            with open(file_name, 'a+', newline="") as f:
                writer = csv.DictWriter(f, list(stats.keys()))
                if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
                    print('print headers')
                    writer.writeheader()
                writer.writerow(stats)
    ## return trades
    return trades

def plot_charts(data, trades):
    ### plotly plot ####
    trace0 = go.Scattergl(
        x = data["timestamp"], 
        y = data["high_chan"],
        line = dict(
            width = 1,
            color = "green",
            dash = "dash"),
        name = "chan_high")

    trace1 = go.Scattergl(
        x = data["timestamp"], 
        y = data["low_chan"],
        line = dict(
            width = 1,
            color = "red",
            dash = "dash"),
        name = "chan_low")

    trace2 = go.Scattergl(
        x = data["timestamp"], 
        y = data["high"],
        line = dict(
            width = 1,
            color = "grey"),
        name = "high")

    trace3 = go.Scattergl(
        x = data["timestamp"], 
        y = data["low"],
        line = dict(
            width = 1,
            color = "grey"),
        name = "low")

    trace4 = go.Scattergl(
        x = data["timestamp"], 
        y = data["long_trail"],
        line = dict(
            width = 2,
            color = "blue",
            dash = "dash"),
        name = "long_trail")

    trace5 = go.Scattergl(
        x = data["timestamp"], 
        y = data["short_trail"],
        line = dict(
            width = 2,
            color = "blue",
            dash = "dash"),
        name = "short_trail")


    trace8 = go.Scattergl(
        x = trades["entry_time"], 
        y = trades["entry"],
        name = "entry",
        mode = "markers",
        marker = dict(
            size = 8,
            color = "purple"),
    )
    trace9 = go.Scattergl(
        x = trades["exit_time"], 
        y = trades["exit"],
        name = "exit",
        mode = "markers",
        marker = dict(
            size = 8,
            color = "black"),
    )
    my_plot = [trace0,trace1,trace2,trace3,trace8,trace9] #trace4,trace5
    #py.iplot(plot)

    # trace10 = go.Scattergl(
    #     x = trades["exit_time"], 
    #     y = trades["profit%_of_cap"], 
    #     line = dict(
    #         width = 1,
    #         color = "green",
    #         dash = "dash"),
    #     name = "profit%_of_cap") 
    # my_plot1 = [trace10]

    trace10 = go.Scattergl(
        x = data["timestamp"], 
        y = data["sharpe"], 
        line = dict(
            width = 1,
            color = "green",
            dash = "dash"),
        name = "sharpe") 
    my_plot1 = [trace10]
    #py.iplot(plot1)

    fig = subplots.make_subplots(rows=2, cols=1, specs=[[{}], [{}]],
                            shared_xaxes=True, shared_yaxes=False,vertical_spacing=0.001)
    for trace in my_plot1:
        fig.append_trace(trace, 2, 1)
    for trace in my_plot:
        fig.append_trace(trace, 1, 1)

    fig['layout'].update(title='Stacked Subplots with Shared X-Axes',height=600)#height=600, width=600
    plot(fig)


if __name__ == "__main__":
    # cli args ##
    args = sys.argv[1:]
    lookback = int(args[0]) if len(args) > 0 else None
    stop = float(args[1]) if len(args) > 1 else None
    target = float(args[2]) if len(args) > 2 else None

    ## vars ##
    start = '2000-01-01'
    end = '2019-08-8'
    index = 'russell2000'
    #lookback = 10
    daily_interest_cost = 2 / 365
    comms_rt_perc = 0.25 # 0.1 spread 0.1 comms 0.5 slippage
    risk_per_trade_perc = 1
    sharpe_filter = None # only takes trades bigger than given sharpe, set to 'None' if you don't want to filter
    atr_len = 70
    atr_trailing_stop = False  
    reverse = False # do you want to reverse the who trading strat to go long when you went short
    optimize_column = None #'target' # you can either optimize of 'target' or 'stop' , you cannot do both, set to 'None' if you don't want to optimize
    modify_trades_in_place = False ## if target_optimization yields better results than no target, then modify the trades

    save = False #<< main, alters the below
    save_files = save # set to False if you dont want to save files and just run through for charts
    delete_files = save # set to True to stop the stats being appended to the same old file
    plot_chart = not save
    print_output = not save

    string_end = 'reverse' if reverse else 'breakout'
    stop_and_target = {}
    if stop and target:
        string_end = f"{string_end}_stop_{stop}_target_{target}"
        stop_and_target = {'stop':stop,'target':target} #{}
        if reverse:
            stop_and_target = {'stop':target,'target':stop}

    folder_name_trades = f"backtest_results_trades_{optimize_column}_{index}_{string_end}" ##############
    folder_name_stats = f"backtest_results_stats_{optimize_column}_{index}_{string_end}" #################

    os.makedirs(f"./{folder_name_trades}",exist_ok=True)
    os.makedirs(f"./{folder_name_stats}",exist_ok=True)

    ## download data ##
    tickers = sorted(list(save_tickers(index).keys()))
    all_stock_data, tickers = download_data_local_check(index, tickers, start, end)
    tickers = ['AAOI'] #['AAPL']
    
    ## backtest ##
    for lookback_days in [lookback]: #[5,10,20,30,40,50,60,70,80,90,120,150,180,210,250,280,310,350]:
        print('lookback_days', lookback_days)
        all_trades = []
        for stock in tickers:
            print(stock)
            stock_data = filter_for_stock(all_stock_data, stock)
            trades = run_backtest(stock, stock_data, lookback_days, daily_interest_cost, comms_rt_perc, risk_per_trade_perc, 
                                  modify_trades_in_place, folder_name_stats, atr_trailing_stop, atr_len, reverse, delete_files, 
                                  save_files, plot_chart, print_output, optimize_column, stop_and_target, sharpe_filter)
            delete_files = False
            if len(trades) > 0:
                all_trades.append(trades)
        ## save all trades to file
        if len(all_trades) > 0:
            all_trades = pd.concat(all_trades)
            if save_files:
                all_trades.to_csv(f"./{folder_name_trades}/lookback_{lookback_days}_trailing_{atr_trailing_stop}_trades.csv")
            #stock = 'AMCR'
            #run_backtest(stock)
        
    print('finished!')