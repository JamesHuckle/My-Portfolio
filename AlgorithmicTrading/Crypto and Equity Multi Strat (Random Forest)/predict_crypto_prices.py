from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import (train_test_split,  
                                     KFold, 
                                     cross_val_score,
                                     cross_val_predict,
                                    )
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
import sklearn.metrics as metrics
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import scikitplot as skplt
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
import asyncio 
import aiohttp  # pip install aiohttp aiodns 
import pickle
import talib
import time
import json
import sys
import ast
import os

import warnings
warnings.filterwarnings('ignore')

with open('env.txt', 'r') as f:
    global api_key
    api_key = ast.literal_eval(f.read())['cc_api_key']


def clean_get_data(data):
    data.drop(['conversionType','conversionSymbol'],axis='columns',inplace=True)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data['daily_return'] = (data['close'].pct_change()*100)
    data['target'] = (data['daily_return']).shift(-1)
    no_data = data[data[['volumefrom','volumeto']].sum(axis='columns') == 0].index
    data.drop(no_data, axis='rows', inplace=True)
    inf = data[data['daily_return'] > 100].index
    data.drop(inf, axis='rows', inplace=True)
    data.reset_index(inplace=True, drop=True)
    return data


def get_data(pair, exchange='Coinbase'):
    fsym, tsym = pair.split('-')
    try:
        response = requests.get(f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={fsym}&tsym={tsym}&'
                                f'limit=2000&e={exchange}&tryConversion=false&api_key={api_key}').json()
        data = pd.DataFrame(response['Data']['Data'])
        data = clean_get_data(data)
    except:
        print('No data found for', pair, exchange)
        return []
    return data


async def async_get(session, pair, exchange): 
    fsym, tsym = pair.split('-')
    try:
        url = (f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={fsym}&tsym={tsym}&'
               f'limit=2000&e={exchange}&tryConversion=false&api_key={api_key}')
        resp = await session.request('GET', url=url, ssl=False) 
        data = await resp.json()
        data = pd.DataFrame(data['Data']['Data'])
        data = clean_get_data(data)
        data = feature_pipeline(data)
    except Exception as e:
       print('Async, no data found for', pair, exchange, e)
       return pd.DataFrame()
    return data


async def async_get_data(pairs_and_exchanges): 
    async with aiohttp.ClientSession() as session: 
        tasks = [] 
        for pair, exchange in pairs_and_exchanges.items(): 
            tasks.append(async_get(session=session, pair=pair, exchange=exchange)) 
        result = await asyncio.gather(*tasks, return_exceptions=True) 
        all_pairs = []
        for res in result:
            all_pairs.append(res)
        data = pd.concat(all_pairs)
        return data 


def get_best_pairs(e='Coinbase'):
    pair_info = requests.get(f'https://min-api.cryptocompare.com/data/pair/mapping/exchange?e={e}&api_key={api_key}').json()['Data']
    pairs = []
    for row in pair_info:
        fsym = row['exchange_fsym']
        tsym = row['exchange_tsym']
        pairs.append(f'{fsym}-{tsym}')
    return pairs


def add_roll_and_lag(data, feature, roll_range, lag_range):
    for roll in roll_range:
        data[f"{feature}_roll_mean_{roll}"] = data[feature].rolling(roll,roll).mean()
        data[f"{feature}_roll_std_{roll}"] = data[feature].rolling(roll,roll).std()
        data[f"{feature}_roll_min_{roll}"] = data[feature].rolling(roll,roll).min()
        data[f"{feature}_roll_max_{roll}"] = data[feature].rolling(roll,roll).max()

    for lag in lag_range:
        data[f"{feature}_lag_{lag}"] = data[feature].shift(lag)
        
    data[f"{feature}_pct_change"] = data[feature].pct_change().replace([np.inf, -np.inf], 0)
    return data


def dist_from_high_low(data, win_range, plot=False):
    for win in win_range:
        for col in ['high','low']:
            if col == 'high':
                data[f"extreme_{col}_{win}"] = data[col].rolling(win, win).max()
                data[f"dist_from_{col}_window_{win}"] = 1 - (data[col] / data[f"extreme_{col}_{win}"])
            elif col == 'low':
                data[f"extreme_{col}_{win}"] = data[col].rolling(win, win).min()
                data[f"dist_from_{col}_window_{win}"] = 1 - (data[f"extreme_{col}_{win}"] / data[col]) 
            
        dist_high = data[f"dist_from_high_window_{win}"]
        dist_low = data[f"dist_from_low_window_{win}"]
        both_high_and_low = ((dist_high + dist_low) == 0)
        data.loc[both_high_and_low,
                 f"distance_from_high_low_{win}"] = 0.5
        data.loc[~both_high_and_low,
                 f"distance_from_high_low_{win}"] = dist_low / (dist_high + dist_low)

        
    ## plotting results
    if plot:
        for plot_col in ['low','high']:
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15,10))
            plt.subplots_adjust(wspace=0, hspace=0)
            data[0:200].plot(x='time',y=f"extreme_{plot_col}_10", ax=ax[0])
            data[0:200].plot(x='time', y=plot_col, ax=ax[0])
            data[0:200].plot(x='time', y=f"dist_from_{plot_col}_window_10", color='k', ax=ax[1])

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15,10))
        plt.subplots_adjust(wspace=0, hspace=0)
        data[0:200].plot(x='time',y=f"extreme_low_10", ax=ax[0])
        data[0:200].plot(x='time',y=f"extreme_high_10", ax=ax[0])
        data[0:200].plot(x='time',y='low', ax=ax[0])
        data[0:200].plot(x='time',y='high', ax=ax[0])
        data[0:200].plot(x='time', y="distance_from_high_low_10", color='k', ax=ax[1])
    return data


def ema(data, roll_range):
    points = ['high','low','close']
    for price_point in points:
        for roll in roll_range:
            ema = talib.EMA(data[price_point], timeperiod=roll)
            data[f'ema_{roll}_{price_point}'] = ema
            for price_point_diff in points:
                data[f"ema_{roll}_{price_point}_diff_{price_point_diff}"] = ((data[price_point_diff] / ema) -1) * 100
    return data


def stoch(data, period_range, speed_range, plot=False):
    for period in range(2,15,2):
        for speed in range(1,15,2):
            (data[f'slowk_{period}_{speed}'],
             data[f'slowd_{period}_{speed}']) = talib.STOCH(data['high'], data['low'],
                                                data['close'], period, speed, 0, speed, 0)

    ## plot data
    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15,10))
        plt.subplots_adjust(wspace=0, hspace=0)
        data[:200].plot(x='time', y='close', ax=ax[0])
        data[:200].plot(x='time', y=f'slowk_5_3', ax=ax[1])
        data[:200].plot(x='time', y=f'slowd_5_3', ax=ax[1])
    return data


def stoch_turn(data, thresh_range, plot=False):
    period = 5
    speed = 3
    (data[f'slowk_{period}_{speed}'],
    data[f'slowd_{period}_{speed}']) = talib.STOCH(data['high'], data['low'],
                                                   data['close'], period, speed, 0, speed, 0)
    
    for thresh in thresh_range:
        turn_up = (data[f'slowk_{period}_{speed}'].shift(1) < data[f'slowd_{period}_{speed}'].shift(1)) & \
                  (data[f'slowk_{period}_{speed}'] > data[f'slowd_{period}_{speed}']) & \
                  (data[f'slowd_{period}_{speed}'] <= thresh)
        data[f'stoch_turn_up_{thresh}'] = 0
        data.loc[turn_up,f'stoch_turn_up_{thresh}'] = 1
        
    for thresh in thresh_range:
        turn_down = (data[f'slowk_{period}_{speed}'].shift(1) > data[f'slowd_{period}_{speed}'].shift(1)) & \
                    (data[f'slowk_{period}_{speed}'] < data[f'slowd_{period}_{speed}']) & \
                    (data[f'slowd_{period}_{speed}'] >= thresh)
        data[f'stoch_turn_down_{thresh}'] = 0
        data.loc[turn_down,f'stoch_turn_down_{thresh}'] = 1
    
    ## plot data
    if plot:
        fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(15,10))
        plt.subplots_adjust(wspace=0, hspace=0)
        data[:200].plot(x='time', y='close', ax=ax[0])
        data[:200].plot(x='time', y=f'slowk_5_3', ax=ax[1])
        data[:200].plot(x='time', y=f'slowd_5_3', ax=ax[1])  
        data[:200].plot(x='time', y=f'stoch_turn_up_20', ax=ax[2])
        data[:200].plot(x='time', y=f'stoch_turn_down_80', ax=ax[3])
    return data


def candlestick_patters(data, meth_filter=None):
    if meth_filter:
        candle_meth = {meth:getattr(talib,meth) for meth in dir(talib) 
                       if 'CDL' in meth and meth in meth_filter}
    else:
        candle_meth = {meth:getattr(talib,meth) for meth in dir(talib) 
                       if 'CDL' in meth}
    for name, meth in candle_meth.items():
        data[f"{name}_0_0"] = meth(data['open'], data['high'], data['low'], data['close'])
        data.loc[data[f"{name}_0_0"] <= 0, f"{name}_0_0"] = 0
        data.loc[data[f"{name}_0_0"] > 0, f"{name}_0_0"] = 1
    return data


def adx(data, period_range, upper_range, padding_range, plot=False):
    for period in period_range:
        adx = talib.ADX(data['high'], data['low'], data['close'], period)
        data[f'adx_{period}'] = adx
        for upper in upper_range:
            for padding in padding_range:
                name = f'adx_{period}_{upper}_{padding}'
                roll = data['close'].rolling(period).mean()
                down = (adx > upper) & (adx < upper+padding) & (data['close'] <= roll)
                up = (adx > upper) & (adx < upper+padding) & (data['close'] > roll)
                data.loc[down, f"{name}_down"] = 1
                data[f"{name}_down"].fillna(0, inplace=True)
                data.loc[up, f"{name}_up"] = 1
                data[f"{name}_up"].fillna(0, inplace=True)
        
    ## plot data
    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15,10))
        plt.subplots_adjust(wspace=0, hspace=0)
        data[:200].plot(x='time', y='close', ax=ax[0])
        data[:200].plot(x='time', y=f'adx_10_60_20', ax=ax[1])
    return data


def day_of_week(data):
    day = data['time'].dt.dayofweek
    days = pd.get_dummies(day)
    days_list = ['mon','tue','wed','thu','fri','sat','sun']
    days.columns = days_list[:len(days.columns)]
    data = pd.concat([days ,data], axis='columns', sort=False)
    return data


def rsi(data, period_range, plot=False):
    for period in range(4,50,5):
        data[f'rsi_{period}'] = talib.RSI(data['close'], period)

    ## plot data
    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15,10))
        plt.subplots_adjust(wspace=0, hspace=0)
        data[:200].plot(x='time', y='close', ax=ax[0])
        data[:200].plot(x='time', y=f'rsi_10', ax=ax[1])
    return data


def print_candlestick_chart(data):
    fig = go.Figure(data=[go.Candlestick(x=data['time'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'])])

    # fig.add_trace(go.Scatter(x=data['time'], y=data['doji']))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()


def corr_to_target():
    corrs = data.corr()['target'].abs().sort_values(ascending=False).head(30)
    print(corrs)
    return corrs


def binary_var(data, feature_name, range1=None, range2=None, range3=None, range4=None):
    results = {}
    target_mean = data['target'].mean()
    if range4:
        for feat1 in range1:
            for feat2 in range2:
                for feat3 in range3:
                    for feat4 in range4:
                        binary_feature = f'{feature_name}_{feat1}_{feat2}_{feat3}_{feat4}'
                        feature = data[data[binary_feature] == 1]
                        non_feature = data[data[binary_feature] == 0]

                        if len(feature) == 0:
                            continue
                        t, p = ks_2samp(feature['target'], data['target'])
                        results[binary_feature] = {
                            'p_value':p, '#_instances':len(feature),
                            'target_mean_diff':feature['target'].mean() - target_mean
                        }
    elif range3:
        for feat1 in range1:
            for feat2 in range2:
                for feat3 in range3:
                    binary_feature = f'{feature_name}_{feat1}_{feat2}_{feat3}'
                    feature = data[data[binary_feature] == 1]
                    non_feature = data[data[binary_feature] == 0]

                    if len(feature) == 0:
                        continue
                    t, p = ks_2samp(feature['target'], data['target'])
                    results[binary_feature] = {
                        'p_value':p, '#_instances':len(feature),
                        'target_mean_diff':feature['target'].mean() - target_mean
                    }
    elif range2:
        for feat1 in range1:
            for feat2 in range2:
                binary_feature = f'{feature_name}_{feat1}_{feat2}'
                feature = data[data[binary_feature] == 1]
                non_feature = data[data[binary_feature] == 0]

                if len(feature) == 0:
                    continue
                t, p = ks_2samp(feature['target'], data['target'])
                results[binary_feature] = {
                    'p_value':p, '#_instances':len(feature),
                    'target_mean_diff':feature['target'].mean() - target_mean
                }  
    elif range1:
        for feat1 in range1:
            binary_feature = f'{feature_name}_{feat1}'
            feature = data[data[binary_feature] == 1]
            non_feature = data[data[binary_feature] == 0]

            if len(feature) == 0:
                continue
            t, p = ks_2samp(feature['target'], data['target'])
            results[binary_feature] = {'p_value':p, '#_instances':len(feature),
                                       'target_mean_diff':feature['target'].mean() - target_mean}        
    else:
        binary_feature = f'{feature_name}'
        feature = data[data[binary_feature] == 1]
        non_feature = data[data[binary_feature] == 0]
        t, p = ks_2samp(feature['target'], data['target'])
        results[f"{feature_name}"] = {'p_value':p, '#_instances':len(feature),
                                      'target_mean_diff':feature['target'].mean() - target_mean}
    
    data.fillna(0,inplace=True)
    return pd.DataFrame(results).transpose()


def plot_binary_var(data, feature_name):
    feature = data[data[feature_name] == 1]
    non_feature = data[data[feature_name] == 0]
    ax = feature['target'].plot.kde(figsize=(12,8), legend=True)
    non_feature['target'].plot.kde(ax=ax, legend=True)
    data['target'].plot.kde(ax=ax, legend=True)
    ax.legend(["feature", "no_feature", "all_data"])

    dp = 2
    print('all_data',round(data['target'].mean(),dp),'|',round(data['target'].std(),dp))
    print('feature',round(feature['target'].mean(),dp),'|',round(feature['target'].std(),dp))
    print('non_feature',round(non_feature['target'].mean(),dp),'|',round(non_feature['target'].std(),dp))


def candlestick_test(data, candle_meth):
    results = {}
    for name, meth in candle_meth.items():
        results1 = binary_var(data, name, [0], [0])
        if results1:
            results[list(results1.keys())[0]] = list(results1.values())[0]
    return pd.DataFrame(results).transpose().sort_values('p_value')


def feature_pipeline(data, volume_col='volumefrom'):
    data[volume_col].replace(0, np.nan, inplace=True)
    data[volume_col] = data[volume_col].ffill()
    data = add_roll_and_lag(data, volume_col, range(2,101,5), range(1,101,5))
    print('roll 1 done')
    data = add_roll_and_lag(data, f'{volume_col}_pct_change', range(2,101,5), range(1,101,5))
    print('roll 2 done')
    data = add_roll_and_lag(data, 'daily_return', range(2,101,5), range(1,101,5))
    print('roll 3 done')
    data = dist_from_high_low(data, range(5,101,5))
    print('high_low done')
    data = ema(data, range(2,201,10))
    print('ema done')
    data = stoch(data, range(2,15,2), range(1,15,2))
    data = stoch_turn(data, range(1,101,5))
    print('stoch_turn done')
    data = candlestick_patters(data)
    print('candlestick_patters done')
    data = adx(data, range(10,21,10), range(5,76,10), range(5,21,5))
    print('adx done')
    data = day_of_week(data)
    data = rsi(data, range(4,51,5))
    print('rsi done')
    print('all done')
    
    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data

# time_col = data['time'].copy()
# data.drop(columns=['time'], inplace=True)
# data = feature_pipeline(data)


def best_features(data, features):
    data = data[features+['target']]
    return data


def normalize(data):
    #data['target'] = pd.cut(data['target'],bins=[-30,-10,-5,-3,-1,1,3,5,10,30])
    # data['target'] = pd.cut(data['target'],bins=[-30,0,30])
    # data['target'] = data['target'].astype(str)
    # data['target'].value_counts(normalize=True)*100
    
    # max_filter = 5
    # large = data['target'] > max_filter
    # small = data['target'] < -max_filter
    # data.loc[large,'target'] = max_filter
    # data.loc[small,'target'] = -max_filter
    
    pos = data['target'] > 0
    neg = data['target'] <= 0
    # pos = (data['target'] > -0.5) & (data['target'] < 1)
    # neg = (data['target'] <= -0.5) | (data['target'] >= 1)
    data.loc[pos,'target'] = 1
    data.loc[neg,'target'] = 0
    return data


def master_dataset(get_existing_data, exchange, max_pairs, problem):
    file_name = f'xgboost_data_{exchange}_{max_pairs}.csv'
    exists = os.path.exists(file_name)
    if get_existing_data and exists:
        data = pd.read_csv(file_name)
    else:
        pairs = get_best_pairs(e=exchange)
        all_pairs = []
        for pair in pairs[0: max_pairs]:
            print(pair)
            response = get_data(pair, exchange)
            if len(response) == 0: 
                continue
            response = feature_pipeline(response)
            all_pairs.append(response)
        data = pd.concat(all_pairs)
        data.reset_index(inplace=True, drop=True)
        data.to_csv(file_name, index=False)
    
    time_col = data['time'].copy()
    real_target = data['target'].copy()
    data.drop(columns=['time'], inplace=True)

    if problem == 'classification':
        data = normalize(data)
        binary_perc = data['target'].value_counts(normalize=True).values
        print('binary class proportions', binary_perc)

    return data, time_col, real_target, binary_perc


def tune_hyperparams(model, problem, X_train, y_train):
    print('tuning model')
    params = {
        'learning_rate'         : [ 0.01, 0.02, 0.03, 0.04, 0.05 ],
        'max_depth'             : [ 1, 2, 3, 4, 5 ],
        'min_child_weight'      : [ 5, 7, 9, 12, 15, 20, 25 ],
        'gamma'                 : [ 0.0, 0.01, 0.02, 0.03, 0.05, 0.1 ],
        'colsample_bytree'      : [ 0.6, 0.7, 0.8, 0.9 ],
        'early_stopping_rounds' : [ 5, 10, 30 ],
    }
    if problem == 'classification':
        params['objective'] = ['binary:logistic']
    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=20,
                                       n_jobs=-1, cv=5, verbose=3)
    random_search.fit(X_train, y_train)

    print(random_search.best_params_)
    return random_search.best_estimator_


def get_huge_out_of_sample(final_pairs, num):
    file_name = 'huge_out_of_sample.csv'
    if os.path.exists(file_name):
        data = pd.read_csv(file_name)
    else:
        config = {}
        for pair, exchanges in final_pairs.items():
            wanted_exchanges = ['cccagg', 'coinbase', 'binance', 'kraken', 'okex',
                                'huobi', 'bitfinex', 'gemini', 'poloniex', 'itbit', 'bitstamp']
            for want in wanted_exchanges:
                exchanges = [e.lower() for e in exchanges]
                if want in exchanges:
                    config[pair] = want
            if len(config) == num: break
        data = asyncio.run(async_get_data(config))  # Python 3.7+
        data.to_csv('huge_out_of_sample.csv', index=False)
    data.dropna(inplace=True)
    data['time'] = pd.to_datetime(data['time'])
    print(data.columns[data.columns.duplicated()])
    return data


def get_out_of_sample(exchange, pair, top_pair_idx=None, generate_huge_out_sample=False):
   
    out_sample_pairs, final_pairs = get_out_sample_pair_exchange(train_exchange, train_max_pairs)
    
    if generate_huge_out_sample:
        data = get_huge_out_of_sample(final_pairs, num=400)
    elif top_pair_idx:
        pair = get_best_pairs(exchange)[top_pair_idx]
        data = get_data(pair, exchange)
        data = feature_pipeline(data)
    elif not exchange or not pair:
        while True:
            random_pair = str(np.random.choice(out_sample_pairs))
            random_exchange = str(np.random.choice(final_pairs[random_pair]))
            print('No valid exchange, pair or pair idx to run backtest... choosing one at random', random_pair, random_exchange)
            print('getting out of sample data', random_pair, random_exchange)
            data = get_data(random_pair, random_exchange)
            if len(data) < 100:
                print(random_pair, random_exchange, 'does not have enough data! Trying again ...')
                continue
            else:
                break
        data = feature_pipeline(data)
    else:
        data = get_data(pair, exchange)

    time_col = data['time'].copy()
    data.drop(columns=['time'], inplace=True)
    real_target = data['target'].copy()
    if problem == 'classification':
        data = normalize(data)
        binary_perc = data['target'].value_counts(normalize=True).values
        print('binary class proportions', binary_perc)
    # load data
    features = [col for col in data.columns.to_list() if col != 'target']
    # split data
    X_test = data[features]
    y_test = data['target']
    return X_test, y_test, time_col, real_target, binary_perc


def get_out_sample_pair_exchange(train_exchange, train_max_pairs):
    data = requests.get(f'https://min-api.cryptocompare.com/data/v4/all/exchanges?api_key={api_key}').json()
    final_pairs = {}
    try:
        exchange_pairs = data['Data']['exchanges']
    except Exception as e:
        message = data.setdefault('Message','')
        print('Problem getting all exchanges and pairs', message, e)
    for exchange, pair_data in exchange_pairs.items():
        pairs = pair_data['pairs']
        for fsym, tsyms in pairs.items():
            for tysm in tsyms['tsyms']:
                pair = f'{fsym}-{tysm}'
                final_pairs.setdefault(pair,[]).append(exchange)
    in_sample_pairs = get_best_pairs(e=train_exchange)[0: train_max_pairs]
    out_sample_pairs = [pair for pair in final_pairs if pair not in in_sample_pairs]
    return out_sample_pairs, final_pairs


def is_pair_truely_out_of_sample(exchange, pair, top_pair_idx, train_exchange, train_max_pairs):
    if top_pair_idx:
        pair = get_best_pairs(e=exchange)[top_pair_idx]
    in_sample_pairs = get_best_pairs(e=train_exchange)[0: train_max_pairs]
    if pair in in_sample_pairs:
        print('Warning!',pair,'was used to train model, so may give misleading historical backtest results.',
              'It is fine for predicting purposes')


def output_accuracy(output_roc_chart, problem, pred_dataset_y, y_predict_proba, predictions, binary_perc):
    if problem == 'classification':
        if output_roc_chart:
            skplt.metrics.plot_roc(pred_dataset_y, y_predict_proba)
        acc = metrics.accuracy_score(pred_dataset_y, predictions)
        text = f"Accuracy: {round(acc,3)} | Class percentage: {round(binary_perc[0],3)} | Improvement (>1?): {round(acc/binary_perc[0],3)}"
        print(text)
        return text


def backtest(positions, pred_dataset_x, predictions, problem, pair, exchange, text):
    all_results = {}
    for position in positions:
        
        x_target = real_target[pred_dataset_x.index].values
        x_time = time_col[pred_dataset_x.index].values
        results = pd.DataFrame([x_time, x_target, predictions]).transpose()
        results.columns = ['time', 'target', 'predict']
        results.sort_values('time', inplace=True)

        if problem == 'classification':
            up = results[results['predict'] == 1].index
            down = results[results['predict'] == 0].index
            if position == 'short':
                results.loc[down, 'predict'] = -1
                results.loc[up, 'predict'] = 0
            elif position == 'long':
                None
            elif position == 'both':
                results.loc[down, 'predict'] = -1
            results['final'] = results['target'] * results['predict']

            market_up = results[results['target'] > 0].index
            market_down = results[results['target'] <= 0].index
            print('average up', round(results.loc[up, 'final'].sum(),2), 'market up', round(results.loc[market_up, 'target'].sum(),2))
            print('average down', round(results.loc[down, 'final'].sum(),2), 'market down', round(results.loc[market_down, 'target'].sum(),2))

        elif problem == 'regression':
            up = results[results['predict'] >= 0].index
            down = results[results['predict'] <= 0].index
            if position == 'short':
                results.loc[down, 'final'] = -results['target']
            elif position == 'long':
                results.loc[up, 'final'] = results['target']
            elif position == 'both':
                results.loc[up, 'final'] = results['target']
                results.loc[down, 'final'] = -results['target']

        results['final'] = results['final'].fillna(0)
        results['cumsum_predict'] = results['final'].cumsum()
        results['cumsum_target'] = results['target'].cumsum()
        all_results[position] = results
        return_per_pred = results['final'].sum().round(1)
        return_per_act = results['target'].sum().round(1)
        outperformance = return_per_pred - return_per_act
        pair = pair if pair else 'Multi-pair'
        exchange = exchange if exchange else 'Model data'
        ax = results.plot(x='time', y='cumsum_predict', figsize=(15,10), 
        title=f'| {pair} | {exchange} | {position} directional trades |\n'
              f'| {return_per_pred} % model return | {return_per_act} % underlying return | {outperformance} outperformance |\n' 
              f'| {text} |')
        results.plot(x='time', y='cumsum_target', ax=ax)
        plt.show()
        
    return all_results


def fit_graphs(problem, predictions, y_test, results, direction='both'):
    if problem == 'regression':
        final = pd.DataFrame([predictions, y_test]).transpose()
        final.columns = ['predictions','y_test']
        final['mean'] = 0.27154374825584504
        final['zero'] = 0
        mse_random_guess = metrics.mean_absolute_error(final['y_test'],final['mean'])
        rmse_random_guess = mse_random_guess ** 0.5
        mse_predict = metrics.mean_absolute_error(final['y_test'],final['predictions'])
        rmse_predict = mse_predict ** 0.5
        print('error random guess',round(rmse_random_guess,4))
        print('error prediction  ', round(rmse_predict,4))

    plt.scatter(x=results[direction]['target'], y=results[direction]['predict'])
    plt.show()


def show_feature_importance(features, num):
    importances = dict(zip(features , model.feature_importances_))
    importances = pd.Series(importances)                                                                                                                                     
    importances = importances.sort_values(ascending=False)
    imp_features = importances.head(num)
    print(imp_features)
    return imp_features


if __name__ == '__main__':
    # python predict_crypto_prices.py backtest_only=True exchange=Coinbase pair=BTC-USD
    # python predict_crypto_prices.py backtest_only=True  # will pick and exchange pair at random

    args = sys.argv[1:]
    args = dict([arg.split('=') for arg in args])
    ## Input Variables ##
    # data
    train_exchange = 'Coinbase'
    train_max_pairs = 50
    get_existing_data = True
    # model
    seed = 11
    problem = 'classification' #'regression', 'classification'
    load_existing_model = 'crypto_xgb.pkl' #False, 'crypto_xbg.model'
    save_model = False #'crypto_xgb_tuned.pkl'  #False
    tune_model = False
    cv_score = False
    # test
    backtest_only = bool(args.get('backtest_only',True))
    generate_huge_out_sample = True
    exchange = args.get('exchange',None)
    pair = args.get('pair',None)
    top_pair_idx = None 
    # output
    output_roc_chart = True
    output_graphs_on = 'test' #'test','train'
    output_backtest_chart = True
    backtest_trade_directions = 'both' #'long,short,both'
    backtest_trade_directions = backtest_trade_directions.split(',')
    output_fit_chart = False
    output_best_features = None  #10, 20, 40

    with open('env.txt', 'r') as f:
        global api_key
        api_key = ast.literal_eval(f.read())['cc_api_key']

    if save_model and load_existing_model:
        raise Exception('You cannot both save a model and load another model, choose one!')
    if backtest_only:
        output_graphs_on = 'test'
        if cv_score:
            raise Exception('Cannot run cv_score on backtest_only')
        if output_roc_chart:
            print('Warning! Cannot output roc chart on backtest only run')
            output_roc_chart = False
        if pair:
            assert type(pair) == str
        if top_pair_idx:
            assert type(top_pair_idx) == int
        is_pair_truely_out_of_sample(exchange, pair, top_pair_idx, train_exchange, train_max_pairs)

    else:
        if not train_exchange or not train_max_pairs:
            raise Exception('Must enter a train_exchange name and train_max_pairs number to run machine learning model')
        assert type(train_max_pairs) == int
        assert type(train_exchange) == str
        if not problem or problem not in ['classification', 'regression']:
            raise Exception('Must enter a valid ML probem, either classification or regression')
    if output_backtest_chart:
        if not any([direction in ['both','long','short'] for direction in backtest_trade_directions]):
            raise Exception('Must enter string or comma seperated list of backtest_trade_directions e.g both or both,long,short')


    ## Get Data and Features ##
    if not backtest_only:
        data, time_col, real_target, binary_perc = master_dataset(get_existing_data, train_exchange, train_max_pairs, problem)

        features = [col for col in data.columns.to_list() if col != 'target']
        X_train, X_test, y_train, y_test = train_test_split(data[features], data['target'], test_size=0.2, random_state=seed)

        if problem == 'regression':
            model = xgb.XGBRegressor()
        elif problem == 'classification':
            model = xgb.XGBClassifier(objective='binary:logistic')

        ## Train Model ##
        if tune_model:
            model = tune_hyperparams(model, problem, X_train, y_train)

        if not load_existing_model:
            print('fitting model')
            model.fit(X_train, y_train)
            if save_model:
                print('saving model')
                pickle.dump(model, open(save_model, "wb"))

    if load_existing_model:
        print('loading model',load_existing_model)
        model = pickle.load(open(load_existing_model, "rb"))

    ## Test Model ##
    if cv_score and not backtest_only:
        print('conducting cross fold test...')
        kfold = KFold(n_splits=3, random_state=seed)
        score = cross_val_score(model, X_train, y_train, cv=kfold)
        print('cross fold score:',score)

    if backtest_only:
        X_test, y_test, time_col, real_target, binary_perc = get_out_of_sample(exchange, pair, top_pair_idx, generate_huge_out_sample)

    pred_dataset_x = X_train if output_graphs_on == 'train' else X_test
    pred_dataset_y = y_train if output_graphs_on == 'train' else y_test
    y_predict_proba = model.predict_proba(pred_dataset_x) if problem == 'classification' else None
    predictions = model.predict(pred_dataset_x)

    output_accuracy_text = output_accuracy(output_roc_chart, problem, pred_dataset_y, y_predict_proba, predictions, binary_perc)
    if output_backtest_chart:
        all_results = backtest(backtest_trade_directions, pred_dataset_x, predictions, problem, pair, exchange, output_accuracy_text)
    if output_fit_chart:
        fit_graphs(problem, predictions, y_test, all_results, direction='both')

    ## Predict Today ##
    if backtest_only:
        today_prediction = model.predict_proba(X_test.tail(1))
        print('today_prediction:',dict(zip(['probability of upwards move','probablity of downwards move'],today_prediction[0])))

    ## Feature Importances ## 
    if output_best_features:
        imp_features = show_feature_importance(features, num=output_best_features)
