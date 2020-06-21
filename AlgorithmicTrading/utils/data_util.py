from tqdm import tqdm_notebook as tqdm
from win10toast import ToastNotifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import random
import time
import copy
import os
import gc

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import interpolate

import sys
sys.path.append(r'C:\Users\Jameshuckle\Dropbox\My-Portfolio\AlgorithmicTrading\utils')
from trading_util import (download_data_local_check, prep_stock_data, prep_fx_data, calc_sharpe, calc_romad)


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

def any_data(dataset_type, cols, file_name=''):
    if dataset_type == 'wave':
        raw_data = wave_data(wave_cycles=100, starting_price=100)
        raw_data = np.expand_dims(raw_data, axis=1)
    elif dataset_type == 'random':
        raw_data = np.random.rand(12600) + 100
        raw_data = np.expand_dims(raw_data, axis=1)
    elif dataset_type == 'monte_carlo':
        raw_data = pd.read_csv('monte_carlo_audusd.csv').to_numpy()
    elif dataset_type == 'stock':
        if read_single_file:
            #print('Reading in single file:',file_name)
            raw_data = process_file(file_name)
        else:
            #print('Indexing file from loaded_files:',file_name)
            raw_data = loaded_files[file_name]
        if resample:
            raw_data = raw_data.resample(resample).agg({'Open':'first','High':'max','Low':'min','Close':'last'})
        raw_data.dropna(inplace=True)
        raw_data = raw_data[cols].to_numpy()
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

def sliding_window(data, window = 4, step = 2):
    shape = (data.size - window + 1, window)
    strides = data.strides * 2
    window_data = np.lib.stride_tricks.as_strided(data, strides=strides, shape=shape)[0::step]
    return window_data

def create_window_data(raw_data, data, window, cols, problem_type, num_bars): 
    data_windows = sliding_window(data=data.flatten(), window=window * len(cols), step=len(cols)) 
    data_windows = data_windows.reshape(-1, window, len(cols))
    
    y_close_prices = raw_data[:, -1][window + (num_bars - 1):]
    x = data_windows[:len(y_close_prices)]

    x_close_prices = raw_data[:, -1][window - 1: len(y_close_prices) + window - 1]
    y_pct_diff = (y_close_prices - x_close_prices) / x_close_prices

    if problem_type == 'regression':
        y = y_close_prices
        if data_percentage_diff:
            y = y_pct_diff
    elif problem_type == 'binary':
        y = np.where(y_pct_diff < 0, 0, 1) # classification task
    elif problem_type == 'category':
        y = categorical_classification(y_pct_diff, data, std_thresh=0.29)
    return x, y, y_pct_diff

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

def dataset_diff(train_data_raw, test_data_raw, data_percentage_diff, cols):
    if data_percentage_diff in ['ohlc_diff', 'uni_diff']:
        train_data = np.diff(train_data_raw, axis=0) / train_data_raw[:-1] # pct_change of data
        train_data = np.insert(train_data, 0, [0]*len(cols), axis=0)
        test_data = np.diff(test_data_raw, axis=0) / test_data_raw[:-1] # pct_change of data
        test_data = np.insert(test_data, 0, [0]*len(cols), axis=0)
    elif data_percentage_diff == 'close_diff':
        train_data = ((train_data_raw[1:].T - train_data_raw[:-1,-1]) / train_data_raw[:-1,-1]).T
        train_data = np.insert(train_data, 0, [0]*len(cols), axis=0)
        test_data = ((test_data_raw[1:].T - test_data_raw[:-1,-1]) / test_data_raw[:-1,-1]).T
        test_data = np.insert(test_data, 0, [0]*len(cols), axis=0)
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
    
def create_dataset(file_name, **kwargs):
    raw_data = any_data(dataset_type, cols, file_name=file_name)
            
    train_end_idx = int(len(raw_data) * train_split)
    train_data_raw = raw_data[:train_end_idx]
    test_data_raw = raw_data[train_end_idx:]
    
    if data_percentage_diff:
        train_data, test_data = dataset_diff(train_data_raw, test_data_raw, data_percentage_diff, cols)
        
    x, y, y_pct_diff = create_window_data(train_data_raw, train_data, window, cols, problem_type, num_bars)   
    x_test, y_test, y_test_pct_diff = create_window_data(test_data_raw, test_data, window, cols, problem_type, num_bars)
    
    if target_stop:
        y = calc_binary_stop_target(train_data_raw, bar_horizon, stop_size_pct, target_size_r_r, tick_size_decimals)
        y_test = calc_binary_stop_target(test_data_raw, bar_horizon, stop_size_pct, target_size_r_r, tick_size_decimals)
        y_pct_diff = np.where(y == 0, -1, 1) * stop_size_pct
        y_test_pct_diff = np.where(y_test == 0, -1, 1) * stop_size_pct        
    
    x_shape = x.shape
    x_test_shape = x_test.shape
    
    if standardize:
        global scaler
        if standardize == 'std':
            scaler = StandardScaler()
        elif standardize == 'min_max':
            scaler = MinMaxScaler((-1,1))
        x = scaler.fit_transform(x.reshape(x_shape[0], -1))
        x = x.reshape(x_shape)
        x_test = scaler.transform(x_test.reshape(x_test_shape[0], -1))
        x_test = x_test.reshape(x_test_shape)
        if problem_type == 'regression':
            y_scaler = StandardScaler()
            y = y_scaler.fit_transform(np.expand_dims(y, axis=1))
            y_test = y_scaler.transform(np.expand_dims(y_test, axis=1))
            
    if pca_features:
        pca = PCA(n_components=pca_features*len(cols))
        x = pca.fit_transform(x.reshape(x_shape[0],-1))
        x = x.reshape(x_shape[0], -1, x_shape[2])
        x_test = pca.transform(x_test.reshape(x_test_shape[0], -1))
        x_test = x_test.reshape(x_test_shape[0], -1, x_test_shape[2])
        
    if embeddings:
        assert not standardize and not pca_features
        train_data_labels = create_candlestick_corpus(train_data_raw, train=True, pandas_with_dates=False)[:,-1]
        test_data_labels = create_candlestick_corpus(train_data_raw, train=False, pandas_with_dates=False)[:,-1]
        
        if embedding_type == 'light':
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
            
        x_embed = sliding_window(data=train_data_embedding.flatten(), window=window * vector_size, step=vector_size) 
        x_embed = x_embed.reshape(-1, window, vector_size)
        x = x_embed[:x.shape[0]]
        
        x_test_embed = sliding_window(data=test_data_embedding.flatten(), window=window * vector_size, step=vector_size) 
        x_test_embed = x_test_embed.reshape(-1, window, vector_size)
        x_test = x_test_embed[:x_test.shape[0]]
        
    if problem_type != 'regression':
        elem , count = np.unique(y, return_counts=True)
        
    return x, y, x_test, y_test, y_pct_diff, y_test_pct_diff, train_data_raw, test_data_raw
    
    
    
### FX data #######
fx_files = [
             'EURUSD_1h_2003-2010.csv', #'EURUSD_1h_2010-2020.csv',
             # 'USDJPY_1h_2003-2010.csv', 'USDJPY_1h_2010-2020.csv',
             # 'NZDUSD_1h_2003-2020.csv',
             # 'AUDUSD_1h_2003-2020.csv',
             # 'USDCAD_1h_2003-2020.csv',
             ]

loaded_files = prep_fx_data(fx_files)

###############################################
window = 10
pca_features = False # False, 10
standardize = None #'std', 'minmax'
data_percentage_diff = 'uni_diff' # False, 'close_diff', 'ohlc_diff', 'open_diff', 'uni_diff'
train_split = 0.7
resample = None # None '1D', '4H', '1W'
read_single_file = None #all_files[3] #None
loaded_data = loaded_files

num_bars = 10
problem_type = 'binary' #'regression' 'binary' 'category'
dataset_type = 'wave' #'wave', 'random', 'stock', 'monte_carlo'
#cols = ['Open', 'High', 'Low', 'Close'] if dataset_type == 'stock' else ['univariate']
cols = ['Close'] if dataset_type in ['stock','monte_carlo'] else ['univariate']

###
input_len = pca_features if pca_features else window
###

## target/stop binary outcomes ##
target_stop = False 
if target_stop:
    num_bars = 1 # must be equal to 1!
    problem_type = 'binary'
    dataset_type = 'stock'
    cols = ['Open', 'High', 'Low', 'Close']
    bar_horizon = 10000 # how long to wait for stop or target hit, otherwise assign 1 if in profit or 0 if not
    stop_size_pct = 0.0050 # size of stop in pct
    target_size_r_r = 1 #at the moment it only makes sense to keep it at 1
    tick_size_decimals = 10 # used for rounding (no that important)

embeddings = False
embedding_type = None #None 'light'
if embeddings:
    standardize = False 
    pca_features = False
    vector_size = 200 # 200, 4
    if embedding_type == 'light':
        vector_size = 1
    
generator = False
if generator: 
    ## save all stocks to csv and tfrecords, then load tfrecords as dataset
    save_numpy_to_csv_all_files()
    batch_size = 1000
    train_dataset = create_tfrecord_dataset('all_data_train')
    test_dataset = create_tfrecord_dataset('all_data_test')
else:
    ### load single stock into numpy
    x, y, x_test, y_test, y_pct_diff, y_test_pct_diff, train_data_raw, test_data_raw = create_dataset(
        file_name=list(loaded_files.keys())[0])
#####################################################################