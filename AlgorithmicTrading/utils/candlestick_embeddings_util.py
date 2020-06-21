#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[3]:


def open_diff(numpy_ohlc):
    diff_data_numpy = np.zeros(numpy_ohlc.shape)
    # open vs close
    diff_data_numpy[1:,0] = (numpy_ohlc[1:,0] - numpy_ohlc[:-1,3]) / numpy_ohlc[:-1,3]
    # high, low, close vs open
    diff_data_numpy[:,[1,2,3]] = ((numpy_ohlc[:,[1,2,3]].T - numpy_ohlc[:,0]) / numpy_ohlc[:,0]).T 
    return diff_data_numpy

def remove_dates(raw_data):
    dates = raw_data.index
    raw_data = raw_data.reset_index(drop=True)
    return raw_data, dates

def diff(raw_data):
    diff_data = open_diff(raw_data)
    diff_data = pd.DataFrame(diff_data, columns=['open_diff','high_diff','low_diff','close_diff'])
    return diff_data

def scale(diff_data, train=True):
    if train:
        global data_scaler
        data_scaler = StandardScaler()
        scale_data = data_scaler.fit_transform(diff_data)
    else:
        scale_data = data_scaler.transform(diff_data)
    scale_data = pd.DataFrame(scale_data, columns=['open_scale','high_scale','low_scale','close_scale'])
    return scale_data

def scale_bins(scale_data, num_bins=5):
    cols = ['open_scale','high_scale','low_scale','close_scale']
    for col in cols:
        scale_data[f'{col}_bins'] = pd.cut(scale_data[col], num_bins, labels=False)
    for col in cols:
        scale_data[f'{col}_bins_label'] = pd.cut(scale_data[col], num_bins)
        
    bin_cols = [f'{col}_bins' for col in cols]
    scale_data[bin_cols] = scale_data[bin_cols].astype(int).astype(str)
    scale_data['label'] = scale_data[bin_cols].agg(''.join, axis=1)
    return scale_data

def all_data_steps(raw_data, train=True):
    raw_data, dates = remove_dates(raw_data)
    diff_data= diff(raw_data)
    scale_data = scale(diff_data, train=train)
    scale_data_bins = scale_bins(scale_data, num_bins=[-np.inf, -1.5, -1, -0.6, -0.1, 0.1, 0.6, 1, 1.5, np.inf])
    all_data = pd.concat([raw_data, diff_data, scale_data_bins],axis=1)
    all_data.index = dates
    return all_data

def create_candlestick_corpus(raw_data, train=True, pandas_with_dates=True):
    if pandas_with_dates:
        raw_data, dates = remove_dates(raw_data)
        diff_data = diff(raw_data.to_numpy())
    else:
        diff_data = diff(raw_data)
        raw_data = pd.DataFrame(raw_data)
        raw_data.columns = ['Open','High','Low','Close']
    scale_data = scale(diff_data, train=train)
    scale_data_bins = scale_bins(scale_data, num_bins=[-np.inf, -1.5, -1, -0.6, -0.1, 0.1, 0.6, 1, 1.5, np.inf])
    data = pd.concat([raw_data, scale_data_bins['label']], axis=1)
    if pandas_with_dates:
        data.index = dates
    else:
        data = data.to_numpy()
    return data

def plot_candlestick_types(candle_one_str, candle_two_str, num_candles):
    s  = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size':20})
    candle_one_filter = all_data.query('label == @candle_one_str').head(num_candles)
    candle_two_filter = all_data.query('label == @candle_two_str').head(num_candles)
    print('there are',len(candle_one_filter), 'candle_one')
    print('there are',len(candle_two_filter), 'candle_two')
    filtered = pd.concat([candle_one_filter, candle_two_filter], axis= 0)[['open_diff','high_diff','low_diff','close_diff']]
    filtered.columns = ['Open','High','Low','Close']
    filtered[['High','Low','Close']] = (filtered[['High','Low','Close']].T + filtered['Open']).T
    mpf.plot(filtered, type='candle', figscale=2)

#corpus_data = create_candlestick_corpus(raw_data, pandas_with_dates=False)

