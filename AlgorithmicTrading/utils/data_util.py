from tqdm import tqdm_notebook as tqdm
from win10toast import ToastNotifier
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf 
import pandas as pd
import numpy as np
import bs4 as bs
import requests
import pickle
import random
import time
import copy
import sys
import os
import gc

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import interpolate

sys.path.append(r'C:\Users\Jameshuckle\Dropbox\My-Portfolio\AlgorithmicTrading\utils')


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

def download_data_local_check(name, start, end, individual_tickers=None):
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
        if individual_tickers:
            pass
        elif name == 'SP500':
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
      
def process_file(file_name):
    folder = r'C:\Users\Jameshuckle\Dropbox\My-Portfolio\AlgorithmicTrading\data'
    raw = pd.read_csv(f"{folder}/{file_name}")
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
        data = process_file(fx_file)
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
   
def calc_romad(daily_pct_change, filter_large_trades=None, yearly_agg=np.median, compound=False, plot=False,
    extra_title=''):
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
        print('yearly_returns:', yearly_returns)
        title = (f'{extra_title}\n'
                f'romad:{romad}\n'
                f'tot profit:{round(daily_pct_change.sum(),2)} | avg_yearly_return:{round(avg_yearly_return,2)} \n'
                f'max_dd:{round(max_dd,2)}')
        equity.plot(title=title)
        max_equity.plot()
        plt.show()
    return romad

def increase_wave_len(full_wave, wave_len_increase):
    full_wave[-2] = full_wave[0] # match the start and end
    x = np.arange(len(full_wave))
    f = interpolate.interp1d(x, full_wave)
    x_new = np.arange(0,len(full_wave)-1, 1/wave_len_increase)
    y_new = f(x_new)
    return y_new
# new_wave = increase_wave_len(full_wave, wave_len_increase=1)
# plt.plot(new_wave)

def create_waves(starting_price=100, overall_wave_len=0.1, overall_wave_amp=1, 
                     secondary_wave_amp=0.5, noise_amp=0, trend=0):
    data = []
    slow_cosine_wave = np.cos(np.arange(0, 12.7, 0.1))
    fast_cosine_wave = np.cos(np.arange(0, 12.7 * 6, 0.1 * 6)) * secondary_wave_amp
    full_wave = slow_cosine_wave + fast_cosine_wave
    full_wave = increase_wave_len(full_wave, overall_wave_len)
    noise = np.random.randn(len(full_wave)) * noise_amp
    full_wave = (full_wave  + noise) * overall_wave_amp
    calc_start_diff = starting_price - full_wave[0] 
    full_wave += calc_start_diff
    full_wave += ((np.array(range(len(full_wave)+1))**trend) - 1)[1:]  # [1:] stops **trend giving neg values
    return full_wave

def wave_data(wave_cycles=100, starting_price=100):
    data = np.array([])
    for w in range(wave_cycles):
        full_wave = create_waves(starting_price=starting_price, overall_wave_len=1, overall_wave_amp=1, 
                                     secondary_wave_amp=0.5, noise_amp=0, trend=0)
        starting_price = full_wave[-1]
        data = np.concatenate([data, full_wave])
    return data

def any_data(var, file_name=''):
    if var.dataset_type == 'wave':
        raw_data = wave_data(wave_cycles=100, starting_price=100)
        raw_data = np.expand_dims(raw_data, axis=1)
    elif var.dataset_type == 'random':
        raw_data = np.random.rand(12600) + 100
        raw_data = np.expand_dims(raw_data, axis=1)
    elif var.dataset_type == 'monte_carlo':
        raw_data = pd.read_csv('monte_carlo_audusd.csv').to_numpy()
    elif var.dataset_type == 'stock':
        if var.read_single_file:
            #print('Reading in single file:',file_name)
            raw_data = process_file(file_name)
        else:
            #print('Indexing file from var.loaded_files:',file_name)
            raw_data = var.loaded_files[file_name]
        if var.resample:
            raw_data = raw_data.resample(var.resample).agg({'Open':'first','High':'max','Low':'min','Close':'last'})
        raw_data.dropna(inplace=True)
        raw_data = raw_data[var.cols].to_numpy()
    else:
        raise Exception(f'datset of type {dataset} not recognised')
       
    return raw_data

def categorical_classification(y, data, std_thresh):
    y = y.copy()
    pop_mean = data.mean()
    pop_std = data.std() * std_thresh
    large_neg = y < (pop_mean - pop_std)
    large_pos = y > (pop_mean + pop_std)
    small_move = ~(large_neg | large_pos)
    y[large_neg] = 0
    y[small_move] = 1
    y[large_pos] = 2
    return y    

def sliding_window(data, window=4, step=2):
    shape = (data.size - window + 1, window)
    strides = data.strides * 2
    window_data = np.lib.stride_tricks.as_strided(data, strides=strides, shape=shape)[0::step]
    return window_data

def create_window_data(raw_data, data, var): 
    data_windows = sliding_window(data=data.flatten(), window=var.window * len(var.cols), step=len(var.cols)) 
    data_windows = data_windows.reshape(-1, var.window, len(var.cols))
    
    y_close_prices = raw_data[:, -1][var.window + (var.num_bars - 1):]
    x = data_windows[:len(y_close_prices)]

    x_close_prices = raw_data[:, -1][var.window - 1: len(y_close_prices) + var.window - 1]
    y_pct_diff = (y_close_prices - x_close_prices) / x_close_prices

    if var.problem_type == 'regression':
        y = y_close_prices
        if var.data_percentage_diff_y:
            y = y_pct_diff
    elif var.problem_type == 'binary':
        y = np.where(y_pct_diff < 0, 0, 1) # classification task
    elif var.problem_type == 'category':
        y = categorical_classification(y_pct_diff, data, std_thresh=0.29)
    return x, y, y_pct_diff
    
def create_window_data_multi_y(raw_data, data, var): 
    data_windows = sliding_window(data=data.flatten(), window=var.window * len(var.cols), step=len(var.cols)) 
    data_windows = data_windows.reshape(-1, var.window, len(var.cols))
    
    all_y_close_prices = []
    all_y_pct_diff = []
    seq_len = []
    for num_bars in var.num_bars:
        y_close_prices = raw_data[:, -1][var.window + (num_bars - 1):]
        x_close_prices = raw_data[:, -1][var.window - 1: len(y_close_prices) + var.window - 1]
        y_pct_diff = (y_close_prices - x_close_prices) / x_close_prices
        all_y_close_prices.append(y_close_prices)
        all_y_pct_diff.append(y_pct_diff)
        seq_len.append(y_close_prices.shape[0])
    
    # remove end of sequences to match the shortest
    all_y_close_prices = [seq[:min(seq_len)] for seq in all_y_close_prices]
    all_y_pct_diff = [seq[:min(seq_len)] for seq in all_y_pct_diff]
    all_y_close_prices = np.vstack(all_y_close_prices)
    all_y_pct_diff = np.vstack(all_y_pct_diff)
    x = data_windows[:min(seq_len)]
    
    if var.problem_type == 'regression':
        y = all_y_close_prices
        if var.data_percentage_diff_y:
            y = all_y_pct_diff
    else:
        raise Exception('only implemented for regression, sorry')
    return x, y, all_y_pct_diff

def bool_argmax(bool_array):
    if bool_array.sum() == 0:
        return -1
    else:
        return np.argmax(bool_array)

def rp(price):
    return round(price * 10**tick_size_decimals) / 10**tick_size_decimals

def calc_binary_stop_target(raw_data, bar_horizon=100, stop_size_pct=0.004, target_size_r_r=1, tick_size_decimals=4):
    hit_neither_target_or_stop = 0
    hit_target_and_stop = 0
    
    random_numbers = np.random.rand(raw_data.shape[0])
    random_win_thresh = 1/(target_size_r_r+1)

    binary_classes = []
    for row_idx in range(raw_data.shape[0]):
        if row_idx == raw_data.shape[0] -1:
            binary_classes.append(0)
            continue
            
        close_price = raw_data[row_idx, 3]
        child_order_distance = rp(close_price * stop_size_pct)
        target_price = rp(close_price + child_order_distance)
        stop_price = rp(close_price - child_order_distance)

        highs_price_range = raw_data[row_idx+1:row_idx+bar_horizon, 1]
        lows_price_range = raw_data[row_idx+1:row_idx+bar_horizon, 2]

        bool_target = highs_price_range >= target_price
        target_bar_idx = bool_argmax(bool_target)

        bool_stop = lows_price_range <= stop_price
        stop_bar_idx = bool_argmax(bool_stop)

        if target_bar_idx == -1 and stop_bar_idx == -1:
            hit_neither_target_or_stop += 1
            last_bar_close_price = raw_data[row_idx+1:row_idx+bar_horizon, 3][-1]
            if last_bar_close_price <= close_price:
                binary_class = 0
            else:
                binary_clas = 1
        elif target_bar_idx == -1:
            binary_class = 0
        elif stop_bar_idx == -1:
            binary_class = 1
        elif stop_bar_idx < target_bar_idx:
            binary_class = 0
        elif target_bar_idx < stop_bar_idx:
            binary_class = 1
        elif target_bar_idx == stop_bar_idx:
            hit_target_and_stop +=1
            random_win = random_numbers[row_idx] <= random_win_thresh
            binary_class = int(random_win)
        binary_classes.append(binary_class)
    print('hit_target_and_stop:', hit_target_and_stop, 'hit_neither_target_or_stop:', hit_neither_target_or_stop)
    return np.array(binary_classes[window-1:-1])

def display_stop_target(x, y, start=0, end=100, vline=33, hline=0.638):
    plot_data = pd.DataFrame(x[:,-1])
    plot_data.columns = 'open','high','low','close'
    plot_data['y'] = y
    child_order_dist = plot_data['close'] * stop_size_pct
    plot_data['target'] = plot_data['close'] + child_order_dist
    plot_data['stop'] = plot_data['close'] - child_order_dist

    window_plot_data = plot_data[start:end] ## change me
    
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios':[3,1]}, figsize=(20,12))
    plt.subplots_adjust(wspace=0, hspace=0)
    window_plot_data['high'].plot(c='grey', ax=ax[0])
    window_plot_data['low'].plot(c='grey', ax=ax[0])
    window_plot_data['close'].plot(c='k', ax=ax[0])
    window_plot_data['target'].plot(c='g', ax=ax[0])
    window_plot_data['stop'].plot(c='r', ax=ax[0])
    window_plot_data['y'].plot(c='k', ax=ax[1])
    
    ax[0].axvline(vline, c='k', linestyle='dashed') ## change me
    ax[0].axhline(hline, c='k', linestyle='dashed') ## change me
    plt.show()
    
def open_diff(numpy_ohlc):
    diff_data_numpy = np.zeros(numpy_ohlc.shape)
    # open vs close
    diff_data_numpy[1:,0] = (numpy_ohlc[1:,0] - numpy_ohlc[:-1,3]) / numpy_ohlc[:-1,3]
    # high, low, close vs open
    diff_data_numpy[:,[1,2,3]] = ((numpy_ohlc[:,[1,2,3]].T - numpy_ohlc[:,0]) / numpy_ohlc[:,0]).T 
    return diff_data_numpy

def dataset_diff(train_data_raw, test_data_raw, data_percentage_diff):
    if data_percentage_diff in ['ohlc_diff', 'uni_diff']:
        train_data = np.diff(train_data_raw, axis=0) / train_data_raw[:-1] # pct_change of data
        train_data = np.insert(train_data, 0, 0, axis=0)
        test_data = np.diff(test_data_raw, axis=0) / test_data_raw[:-1] # pct_change of data
        test_data = np.insert(test_data, 0, 0, axis=0)
    elif data_percentage_diff == 'close_diff':
        train_data = ((train_data_raw[1:].T - train_data_raw[:-1,-1]) / train_data_raw[:-1,-1]).T
        train_data = np.insert(train_data, 0, 0, axis=0)
        test_data = ((test_data_raw[1:].T - test_data_raw[:-1,-1]) / test_data_raw[:-1,-1]).T
        test_data = np.insert(test_data, 0, 0, axis=0)
    elif data_percentage_diff == 'open_diff':
        train_data = open_diff(train_data_raw)
        test_data = open_diff(test_data_raw)
    else:
        train_data = train_data_raw
        test_data = test_data_raw
    return train_data, test_data

def dataset_target_stop():
    y = calc_binary_stop_target(train_data_raw, bar_horizon, stop_size_pct, target_size_r_r, tick_size_decimals)
    y_test = calc_binary_stop_target(test_data_raw, bar_horizon, stop_size_pct, target_size_r_r, tick_size_decimals)
    y_pct_diff = np.where(y == 0, -1, 1) * stop_size_pct
    y_test_pct_diff = np.where(y_test == 0, -1, 1) * stop_size_pct
    return y, y_test, y_pct_diff, y_test_pct_diff

def fast_fourier_transform(X, n_coefs=10):
    n_samples, window = X.shape
    dft = DiscreteFourierTransform(n_coefs=n_coefs, norm_mean=False,
                                   norm_std=False)
    X_dft = dft.fit_transform(X)

    # Compute the inverse transformation
    if n_coefs % 2 == 0:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * np.c_[X_dft[:, imag_idx],
                                            np.zeros((n_samples, ))]
        ]
    else:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs + 1, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * X_dft[:, imag_idx]
        ]
    X_irfft = np.fft.irfft(X_dft_new, window)  
    return X_irfft
   
   
def create_dataset(file_name, var):
    raw_data = any_data(var, file_name=file_name)
            
    train_end_idx = int(len(raw_data) * var.train_split)
    train_data_raw = raw_data[:train_end_idx]
    test_data_raw = raw_data[train_end_idx:]
    
    # returns data untouched if data_percentage_diff is False
    train_data, test_data = dataset_diff(train_data_raw, test_data_raw, var.data_percentage_diff)
            
    create_window_func = create_window_data_multi_y if var.multi_y == True else create_window_data
    x, y, y_pct_diff = create_window_func(train_data_raw, train_data, var)   
    x_test, y_test, y_test_pct_diff = create_window_func(test_data_raw, test_data, var)
    
    if var.target_stop:
        y = calc_binary_stop_target(train_data_raw, var.bar_horizon, var.stop_size_pct, var.target_size_r_r,
                                    var.tick_size_decimals)
        y_test = calc_binary_stop_target(test_data_raw, var.bar_horizon, var.stop_size_pct, 
                                         var.target_size_r_r, var.tick_size_decimals)
        y_pct_diff = np.where(y == 0, -1, 1) * var.stop_size_pct
        y_test_pct_diff = np.where(y_test == 0, -1, 1) * var.stop_size_pct        
    
    x_shape = x.shape
    x_test_shape = x_test.shape
    
    if var.standardize:
        global scaler
        if var.standardize == 'std':
            scaler = StandardScaler()
        elif var.standardize == 'min_max':
            scaler = MinMaxScaler((-1,1))
        x = scaler.fit_transform(x.reshape(x_shape[0], -1))
        x = x.reshape(x_shape)
        x_test = scaler.transform(x_test.reshape(x_test_shape[0], -1))
        x_test = x_test.reshape(x_test_shape)
        if var.problem_type == 'regression':
            y_scaler = StandardScaler()
            y = y_scaler.fit_transform(np.expand_dims(y, axis=1))
            y_test = y_scaler.transform(np.expand_dims(y_test, axis=1))
            
    if var.pca_features:
        pca = PCA(n_components=pca_features*len(var.cols))
        x = pca.fit_transform(x.reshape(x_shape[0],-1))
        x = x.reshape(x_shape[0], -1, x_shape[2])
        x_test = pca.transform(x_test.reshape(x_test_shape[0], -1))
        x_test = x_test.reshape(x_test_shape[0], -1, x_test_shape[2])
        
    if var.embeddings:
        assert not var.standardize and not var.pca_features
        train_data_labels = create_candlestick_corpus(train_data_raw, train=True, pandas_with_dates=False)[:,-1]
        test_data_labels = create_candlestick_corpus(train_data_raw, train=False, pandas_with_dates=False)[:,-1]
        
        if var.embedding_type == 'light':
            train_data_embedding = train_data_labels.astype(int)
            test_data_embedding = test_data_labels.astype(int)
        else:
            with open(f'candlestick_embeddings_{vector_size}.pkl','rb') as f:
                embeddings_dict = pickle.load(f)
            data_labels = set(np.concatenate([train_data_labels, test_data_labels]))
            unknown_labels = [label for label in data_labels if label not in list(embeddings_dict.keys())]
            unknown_embeddings = {label: np.random.rand(vector_size) for label in unknown_labels}
            embeddings_dict = {**embeddings_dict, **unknown_embeddings}    
            train_data_embedding = np.array([embeddings_dict[label] for label in train_data_labels])
            test_data_embedding = np.array([embeddings_dict[label] for label in test_data_labels])   
            
        x_embed = sliding_window(data=train_data_embedding.flatten(), window=var.window * var.vector_size,
                                 step=var.vector_size) 
        x_embed = x_embed.reshape(-1, var.window, var.vector_size)
        x = x_embed[:x.shape[0]]
        
        x_test_embed = sliding_window(data=test_data_embedding.flatten(), window=var.window * var.vector_size, 
                                      step=var.vector_size) 
        x_test_embed = x_test_embed.reshape(-1, var.window, var.vector_size)
        x_test = x_test_embed[:x_test.shape[0]]
        
    if var.problem_type != 'regression':
        elem, count = np.unique(y, return_counts=True)
        
    return x, y, x_test, y_test, y_pct_diff, y_test_pct_diff, train_data_raw, test_data_raw
 
 
class algo_variables():
    pass


def add_dates(x, y, dates, target_cols):
    x = x.reshape(x.shape[0],-1)
    x, y = pd.DataFrame(x), pd.DataFrame(y.T)
    y.columns = target_cols 
    x.index = dates
    y.index = dates
    return x, y
    
    
def normalize_bar_volatility(x, y, raw_data, ma=100):
    bar_high_low = raw_data[1] - raw_data[2]
    bar_high_low_pct_diff = bar_high_low / raw_data[1]
    bar_high_low_pct_diff.fillna(0, inplace=True)
    roll_avg_diff = bar_high_low_pct_diff.rolling(window=ma, min_periods=0).mean().reset_index(drop=True)
    x = x.div(roll_avg_diff, axis=0)
    y = y.div(roll_avg_diff, axis=0)
    return x, y


def final_dataset(var):
    all_x = []
    all_y = []
    all_x_test = []
    all_y_test = []
    all_train_raw = []
    all_test_raw = []
    
    if var.multi_y:
        target_cols = [f'target_bars_{n}' for n in var.num_bars]
    else:
        target_cols = [f'target_bars_{var.num_bars}']
        
    for file_name in tqdm(list(var.loaded_files.keys())):
        (x, y, x_test, y_test, y_pct_diff, y_test_pct_diff,
         train_data_raw, test_data_raw) = create_dataset(file_name=file_name, var=var)

        loaded_data = var.loaded_files[file_name]
        if var.resample:
            loaded_data = loaded_data.resample(var.resample).agg({'Open':'first','High':'max',
                                                                  'Low':'min','Close':'last'})   
        dates = loaded_data.dropna().index
        train_dates = dates[:len(x)]
        test_dates = dates[len(x):len(x)+len(x_test)]
        
        x, y = add_dates(x, y, train_dates, target_cols)
        all_x.append(x)
        all_y.append(y)
        
        x_test, y_test = add_dates(x_test, y_test, test_dates, target_cols)
        all_x_test.append(x_test)
        all_y_test.append(y_test)
        
        train_data_raw, test_data_raw = pd.DataFrame(train_data_raw), pd.DataFrame(test_data_raw)
        train_raw_dates = dates[:len(train_data_raw)]
        test_raw_dates = dates[len(train_data_raw):]
        train_data_raw.index = train_raw_dates
        test_data_raw.index = test_raw_dates
        all_train_raw.append(train_data_raw)
        all_test_raw.append(test_data_raw)
        
    # Sort data chronologically so the same dates of different instruments are together.
    x = pd.concat(all_x, axis=0).sort_index().reset_index(drop=True)
    y = pd.concat(all_y, axis=0).sort_index().reset_index(drop=True)
    x_test = pd.concat(all_x_test, axis=0).sort_index().reset_index(drop=True)
    y_test = pd.concat(all_y_test, axis=0).sort_index().reset_index(drop=True)

    test_data_raw = pd.concat(all_test_raw, axis=0).sort_index()  
    train_data_raw = pd.concat(all_train_raw, axis=0).sort_index()
    
    train_data = pd.concat([x,y], axis=1)
    test_data = pd.concat([x_test,y_test], axis=1)
    all_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    
    if var.norm_by_vol:
        x, y = normalize_bar_volatility(x, y, train_data_raw[:len(train_data)], ma=1000)
        x_test, y_test = normalize_bar_volatility(x_test, y_test, test_data_raw[:len(test_data)], ma=1000)
    
    return x, y, x_test, y_test, train_data_raw, test_data_raw, all_data   
    
    
def fit_data_for_knn(loaded_files, window, n_candles, percentage_diff='close_diff', close_only=False,
                     norm_by_vol=False, resample=None, train_split=0.8):
    var = algo_variables()
    var.loaded_files = loaded_files
    var.window = window
    var.num_bars = n_candles
    var.multi_y = True # works with a list of var.num_bars
    var.data_percentage_diff = percentage_diff # False, 'close_diff', 'ohlc_diff', 'open_diff'
    var.data_percentage_diff_y = True
    var.train_split = train_split
    var.resample = resample # None '1D', '4H', '1W'
    var.norm_by_vol = norm_by_vol
    var.problem_type = 'regression' #'regression' 'binary' 'category'
    var.dataset_type = 'stock' #'wave', 'random', 'stock', 'monte_carlo'
    var.cols = ['Open', 'High', 'Low', 'Close'] if close_only == False else ['Close']
    var.read_single_file = None
    var.target_stop = False
    var.embeddings = False
    var.pca_features = False
    var.standardize = False
    
    x, y, x_test, y_test, train_data_raw, test_data_raw, all_data = final_dataset(var)
    return x, y, x_test, y_test, train_data_raw, test_data_raw, all_data    
    
def deep_learning_dataset(var, train_validation=0):
    var.multi_y = False
    x, y, x_test, y_test, train_data_raw, test_data_raw, all_data = final_dataset(var)
    if train_validation:
        print('train_validation:', train_validation)
        # only uses train data to make validation (test) data.
        train_idx = int(len(x) * train_validation)
        x_train = x[:train_idx]
        y_train = y[:train_idx]
        x_valid = x[train_idx:]
        y_valid = y[train_idx:]
        return x_train, y_train, x_valid, y_valid, train_data_raw, test_data_raw, all_data  
    else:    
        return x, y, x_test, y_test, train_data_raw, test_data_raw, all_data    
    
    
# ### FX data #######
# fx_files = [
#              'EURUSD_1h_2003-2010.csv', #'EURUSD_1h_2010-2020.csv',
#              # 'USDJPY_1h_2003-2010.csv', 'USDJPY_1h_2010-2020.csv',
#              # 'NZDUSD_1h_2003-2020.csv',
#              # 'AUDUSD_1h_2003-2020.csv',
#              # 'USDCAD_1h_2003-2020.csv',
#              ]

# loaded_files = prep_fx_data(fx_files)
# print('data_util.py')

# ###############################################
# window = 10
# pca_features = False # False, 10
# standardize = None #'std', 'minmax'
# data_percentage_diff = 'uni_diff' # False, 'close_diff', 'ohlc_diff', 'open_diff', 'uni_diff'
# train_split = 0.7
# resample = None # None '1D', '4H', '1W'
# read_single_file = None #all_files[3] #None
# loaded_data = loaded_files

# num_bars = 10
# problem_type = 'binary' #'regression' 'binary' 'category'
# dataset_type = 'wave' #'wave', 'random', 'stock', 'monte_carlo'
# #cols = ['Open', 'High', 'Low', 'Close'] if dataset_type == 'stock' else ['univariate']
# cols = ['Close'] if dataset_type in ['stock','monte_carlo'] else ['univariate']

# ###
# input_len = pca_features if pca_features else window
# ###

# ## target/stop binary outcomes ##
# target_stop = False 
# if target_stop:
#     num_bars = 1 # must be equal to 1!
#     problem_type = 'binary'
#     dataset_type = 'stock'
#     cols = ['Open', 'High', 'Low', 'Close']
#     bar_horizon = 10000 # how long to wait for stop or target hit, otherwise assign 1 if in profit or 0 if not
#     stop_size_pct = 0.0050 # size of stop in pct
#     target_size_r_r = 1 #at the moment it only makes sense to keep it at 1
#     tick_size_decimals = 10 # used for rounding (no that important)

# embeddings = False
# embedding_type = None #None 'light'
# if embeddings:
#     standardize = False 
#     pca_features = False
#     vector_size = 200 # 200, 4
#     if embedding_type == 'light':
#         vector_size = 1
    
# generator = False
# if generator: 
#     ## save all stocks to csv and tfrecords, then load tfrecords as dataset
#     save_numpy_to_csv_all_files()
#     batch_size = 1000
#     train_dataset = create_tfrecord_dataset('all_data_train')
#     test_dataset = create_tfrecord_dataset('all_data_test')
# else:
#     ### load single stock into numpy
#     x, y, x_test, y_test, y_pct_diff, y_test_pct_diff, train_data_raw, test_data_raw = create_dataset(
#         file_name=list(loaded_files.keys())[0])
# #####################################################################