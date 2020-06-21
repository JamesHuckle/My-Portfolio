from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf 
import pandas as pd
import numpy as np
import bs4 as bs
import requests
import os


def save_sp500_tickers():
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


def save_ftse100_tickers():
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

def save_russell2000_tickers():
    tickers = pd.read_csv('https://www.ishares.com/uk/professional/en/products/239710/' +
                          'ishares-russell-2000-etf/1506575576011.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund',
                           skiprows=2)[['Issuer Ticker','Name']]
    tickers.set_index('Issuer Ticker',drop=True,inplace=True)
    return tickers.to_dict()['Name']


def download_data_local_check(name, start, end):
    folder = r'C:\Users\Jameshuckle\Dropbox\My-Portfolio\AlgorithmicTrading\data'
    data_exist = f'{folder}\{name}_all_stock_data_{start}-{end}.csv'
    print(data_exist)
    if os.path.exists(data_exist):
        all_stock_data = pd.read_csv(data_exist,header=[0,1])
        all_stock_data = all_stock_data[1:]
        all_stock_data.set_index(all_stock_data['Unnamed: 0_level_0']['Unnamed: 0_level_1'], drop=True, inplace=True)
        all_stock_data.index.name = 'Date'
        all_stock_data.index = pd.to_datetime(all_stock_data.index)
        all_stock_data.drop(('Unnamed: 0_level_0','Unnamed: 0_level_1'), axis='columns', inplace=True)
    else:
        if name == 'SP500':
            tickers = sorted(list(save_sp500_tickers().keys()))
        elif name == 'RUSSELL2000':
            tickers = sorted(list(save_russell2000_tickers().keys()))
        elif name == 'FTSE100':
            tickers = sorted(list(save_ftse100_tickers().keys()))
        else:
            raise Exception(f'name: {name} has no built function to download data from tickers yet')
        all_stock_data = yf.download(tickers, start, end)
        all_stock_data = all_stock_data.ffill()
        all_stock_data.to_csv(data_exist)
    return all_stock_data
    
    
def process_fx_file(file_name):
    folder = r'C:\Users\Jameshuckle\Dropbox\My-Portfolio\AlgorithmicTrading\data'
    pip_decimal_places = {'EURUSD':4, 'USDJPY':2, 'NZDUSD':4, 'USDCAD':4, 'AUDUSD':4}
    raw = pd.read_csv(f"{folder}/{file_name}")
    #raw[['Open','High','Low','Close']] = raw[['Open','High','Low','Close']] * 10**pip_decimal_places[file_name[:6]]
    time_col = list(raw.columns)[0]
    raw.rename(columns={time_col:'Gmt time'}, inplace=True)
    return raw
    
    
def prep_stock_data(all_stock_data, filter_start_date_tuple=None):
    stock_tickers = list(set(all_stock_data.columns.get_level_values(1)))
    loaded_files = {}
    print('num stocks:',len(stock_tickers))
    for stock in tqdm(stock_tickers):
        columns = [(price, stock) for price in ['Open','High','Low','Close']]
        my_stock = all_stock_data[columns]
        my_stock = my_stock.droplevel(level=1, axis='columns')   
        my_stock.dropna(inplace=True)
        my_stock.index.rename('Gmt time', inplace=True)
        pct_change = my_stock['Close'].pct_change()
        if (pct_change > 1).sum() > 0 or (pct_change < - 0.5).sum() > 0:
            print(stock, 'has moves that are too large')
            continue
        if len(my_stock) < 100:
            print(stock, 'does not have enough data')
            continue
        if filter_start_date_tuple:
            my_stock = my_stock[my_stock.index > datetime(*filter_start_date_tuple)]
        loaded_files[stock] = my_stock
    return loaded_files


def prep_fx_data(fx_files, filter_start_date_tuple=None):
    loaded_files = {}
    for fx_file in fx_files:
        data = process_fx_file(fx_file)
        data['Gmt time'] = pd.to_datetime(data['Gmt time'].str[:16], dayfirst=True)
        data = data.set_index('Gmt time')
        if filter_start_date_tuple:
            data = data[data.index > datetime(*filter_start_date_tuple)]
        loaded_files[fx_file] = data
        print(fx_file)
    return loaded_files

    
def calc_sharpe(daily_pct_change):
    daily_pct_change = pd.Series(daily_pct_change)
    yearly_returns = daily_pct_change.resample('Y').sum()
    sharpe = (yearly_returns.mean() / (yearly_returns.std() + 1e-12))
    return round(sharpe, 2)

    
def calc_romad(daily_pct_change, filter_large_trades=None, yearly_agg=np.median, compound=False, plot=False, extra_title=''):
    if filter_large_trades:
        daily_pct_change = daily_pct_change[daily_pct_change < filter_large_trades]
    if compound:
        yearly_returns = np.log(daily_pct_change + 1).resample('A').sum().values
        blank_years = yearly_returns != 0
        yearly_returns = yearly_returns[blank_years]
        avg_yearly_return = np.exp(yearly_agg(yearly_returns)) -1
        equity = (daily_pct_change + 1).cumprod()
        max_equity = equity.cummax()
        max_dd = ((max_equity - equity) / max_equity).max()  
    else:
        yearly_returns = daily_pct_change.resample('A').sum().values
        blank_years = yearly_returns != 0
        yearly_returns = yearly_returns[blank_years]
        avg_yearly_return = yearly_agg(yearly_returns)
        equity = daily_pct_change.cumsum()
        max_equity = equity.cummax()
        max_dd = (max_equity - equity).max()
        
    romad = round(avg_yearly_return / (max_dd + 1e-12), 2)
    if plot:
        print(yearly_returns)
        title = (f'{extra_title}\n'
                f'romad:{romad}\n'
                f'tot profit:{round(daily_pct_change.sum(),2)} | avg_yearly_return:{round(avg_yearly_return,2)} \n'
                f'max_dd:{round(max_dd,2)}')
        equity.plot(title=title)
        max_equity.plot()
        plt.show()
    return romad

