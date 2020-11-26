# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: 'Python 3.7.4 64-bit (''Jameshuckle'': virtualenv)'
#     language: python
#     name: python37464bitjameshucklevirtualenv1ec6556667214fdb836c8fa1e3a8bcb5
# ---

import wandb
from wandb.keras import WandbCallback
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from time_series_deep_learning import *

sweep_config = {
    'name': 'keras-1',
    'program': 'time_series_hyperparam_sweep.py',
    'method': 'random',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize',
    },
    'parameters': {
        'units': {'values': [5, 10, 20, 50, 100, 200]},
        'layers': {'values': [3, 5, 7, 10, 20, 40, 60]},
        'epochs': {'values': [150]},
        'window': {'values': [5, 7, 10, 15, 20, 30, 50, 100]},
        'num_bars': {'values': [5, 7, 10, 15, 20, 30, 50, 100]},
        'norm_by_vol': {'values': [False, False, False, False, True]},
        'resample': {'values': ['1D']},
        'close_only': {'values': [False, False, False, False, True]},
        'target_stop': {'values': [False, False, False, False, True]},
        'model_arch': {'values': ['lstm','lstm','dnn']},
        'l1_reg': {'values': [0, 0, 0, 0, 1e-8, 1e-7, 1e-6]},
        'l2_reg': {'values': [0, 0, 0, 0, 0, 1e-7, 1e-6, 1e-5]},
        'drop_rate': {'values': [0, 0, 0, 0, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4]},
        'lr': {'values': [1e-3, 1e-4, 1e-5]},
        'problem_type': {'values': ['binary', 'binary', 'binary', 'category', 'regression']},
        'standardize': {'values': ['min_max', 'min_max','std', 'std', False]},
        'pca_features': {'values': [True, False, False, False, False]},
####  
#         'epochs': {'values': [150]},
#         'close_only': {'values': [False]},       
#         'drop_rate': {'values': [0.2]},
#         'l1_reg': {'values': [1e-8]},
#         'l2_reg': {'values': [0]},
#         'layers': {'values': [3]},
#         'lr': {'values': [1e-3]},
#         'model_arch': {'values': ['lstm']}, #set lstm_layers size in code if 'lstm'
#         'norm_by_vol': {'values': [False]},
#         'num_bars': {'values': [100]},  
#         'pca_features': {'values': [False]}, #set pca_fraction size in code if TRUE
#         'problem_type': {'values': ['category']}, #set std_thresh size in code if 'category'
#         'resample': {'values': ['1H']},
#         'standardize': {'values': ['min_max']},
#         'target_stop': {'values': [False]},  #set stop_target size in code if True
#         'units': {'values': [10]},
#         'window': {'values': [5]},
    }
}

# + tags=["active-ipynb"]
# sweep_id = wandb.sweep(sweep_config, project='timeseries-4')

# +
data_source = 'stock' # 'fx', 'stock'

if data_source == 'fx':
    ### FX data #######
    fx_files = [
                 'EURUSD_1h_2003-2020.csv',
                 'GBPUSD_1h_2003-2020.csv',
                 'USDJPY_1h_2003-2020.csv',
                 'NZDUSD_1h_2003-2020.csv',
                 'AUDUSD_1h_2003-2020.csv',
                 'USDCAD_1h_2003-2020.csv',
                 'USDCHF_1h_2003-2020.csv',
                 ]

    loaded_files = prep_fx_data(fx_files)
        
if data_source == 'stock':
    ### stock data ######
    start = '2000-01-01'
    end = '2020-04-28'
    ## download data
    all_stock_data = download_data_local_check('SP500', start, end)
    loaded_files = prep_stock_data(all_stock_data, filter_start_date_tuple=None) #(2015,1,1)


# + tags=["active-ipynb"]
# # this config is only used to test the code in the notebook
# test_config = {
#     'units': 100,
#     'layers': 10,
#     'epochs': 10,
#     'window': 50,
#     'num_bars': 50,
#     'norm_by_vol': False,
#     'resample': '1D',
#     'close_only': True,
#     'target_stop': True,
#     'model_arch': 'lstm',
#     'l1_reg': 1e-7,
#     'l2_reg': 1e-7,
#     'drop_rate': 0.1,
#     'lr': 1e-35,
#     'problem_type': 'binary',
#     'standardize': 'min_max',
#     'pca_features': False, 
# }

# +
def train(test=False):
    if test:
        wb = test_config
    else:
        config = {k:v['values'][0] for k,v in sweep_config['parameters'].items()}
        wandb.init(config=config, magic=False)
        wb = wandb.config
    sweep = True # works with my code to make modifications for a sweep
    
    class algo_variables():
        pass

    var = algo_variables()
    var.window = wb['window'] # number of training bars
    var.pca_features = wb['pca_features'] # False, 10
    if var.pca_features:
        fraction_of_features = np.random.choice([0.25, 0.5])
        var.pca_features = int(wb['window'] * fraction_of_features)
        wb.update({'pca_features':var.pca_features, 'pct_fraction':fraction_of_features},
                  allow_val_change=True)
    var.standardize = wb['standardize'] #'std', 'min_max'
    var.norm_by_vol = wb['norm_by_vol'] #True
    var.data_percentage_diff = 'close_diff' # False, 'close_diff', 'ohlc_diff', 'open_diff'
    var.data_percentage_diff_y = True
    var.train_split = datetime(2019,1,1) #0.9, datetime(2018,1,1)
    var.resample = wb['resample'] # None '1D', '4H', '1W'
    var.read_single_file = None #all_files[3] #None
    var.loaded_files = loaded_files

    var.num_bars = wb['num_bars'] # prediction horizon
    var.problem_type = wb['problem_type'] #'regression' 'binary' 'category'  
    if var.problem_type == 'category':
        var.std_thresh = 1 # to determine a positive, negative or flat trade
        if sweep:
            var.std_thresh = 0.25 #np.random.choice([0.25,0.5])
            wb.update({'std_thresh':var.std_thresh}, allow_val_change=True)
    var.dataset_type = 'stock' #'wave', 'random', 'stock', 'monte_carlo'
    var.close_only = wb['close_only']
    if var.close_only:
        var.cols = ['Close'] if var.dataset_type in ['stock','monte_carlo'] else ['univariate']
    else:
        var.cols = ['Open', 'High', 'Low', 'Close'] if var.dataset_type == 'stock' else ['univariate']
    var.multi_y = False

    var.input_len = var.pca_features if var.pca_features else var.window

    ## target/stop binary outcomes (1 R/R) ##
    var.target_stop = wb['target_stop']
    if var.target_stop:
        var.num_bars = 1 # must be equal to 1!
        var.problem_type = 'binary'
        var.dataset_type = 'stock'
        var.close_only = False
        var.cols = ['Open', 'High', 'Low', 'Close']
        var.bar_horizon = 10000 # how long to wait for stop or target hit, otherwise assign current profit
        var.bar_size_ma = 100 # how long is moving average for bar size (used to calc stop and target)
        var.stop_target_size = 3 # size of stop and target relative to averge bar size
        if sweep:
            var.stop_target_size = np.random.choice([2,3,4])
            wb.update({'stop_target_size':var.stop_target_size}, allow_val_change=True)

    var.embeddings = False
    var.embedding_type = None #None 'light'
    if var.embeddings:
        var.standardize = False 
        var.pca_features = False
        var.vector_size = 200 # 200, 4
        if var.embedding_type == 'light':
            var.vector_size = 1

    generator = True
    if generator: 
        ## save all stocks to csv and tfrecords, then load tfrecords as dataset
        var.train_validation = 0.8 #False # Uses traning data to create test set (for validation)
        var.batch_size = 500
        base_path = f'C:/Users/Jameshuckle/Documents/Algo_Trading/data'
        save_numpy_to_csv_all_files(base_path, var)
        train_dataset = create_tfrecord_dataset(f'{base_path}/all_data_train', var)
        test_dataset = create_tfrecord_dataset(f'{base_path}/all_data_test', var)
    else:
        ### load single stock into numpy
        (x, y, x_test, y_test, y_pct_diff, y_test_pct_diff, train_data_raw,
         test_data_raw) = create_dataset(file_name=list(loaded_files.keys())[1], var=var)
        train_dataset, test_dataset = [], []
        
    tf.keras.backend.clear_session()
    ###
    var.model_arch = wb['model_arch'] # 'dnn','lstm','conv1d','incept1d'
    var.l1_reg = wb['l1_reg'] #1e-6
    var.l2_reg = wb['l2_reg'] #1e-5
    var.drop_rate = wb['drop_rate'] #0.1 #0.2
    ###
        
    if sweep:
        var.layers = wb['layers']
        var.units = wb['units']
        if var.model_arch == 'lstm':
            var.lstm_layers = 2 #np.random.randint(1,4)
            wb.update({'lstm_layers':var.lstm_layers}, allow_val_change=True)

    model = get_model_arch(var.model_arch, var, sweep)
    (total_epochs, all_history, checkpoint_path_base,
     checkpoint_path_model) = reset_model_checkpoint(
        path='B:Algo_Trading/model_checkpoints')

    ################
    metric = 'root_mean_squared_error' if var.problem_type == 'regression' else 'accuracy'
    
    # Do extra epochs if model is learning well
    while total_epochs <= 1500:
        # load model to keep continuity of epochs. To create new model run cell above.
        if os.path.exists(checkpoint_path_model):
            print('loading model')
            model = tf.keras.models.load_model(checkpoint_path_model)

        plot_lr_rate = False
        decrease_lr_rate = False
        validation = True
        var.epochs = wb['epochs']
        var.lr = wb['lr']

        model = compile_model(model=model, lr=var.lr, var=var)
        gc.collect()

        checkpoint_path_cb = checkpoint_path_base+'/model_epoch-{epoch}.ckpt'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_cb,
                                                         save_best_only=False,
                                                         save_weights_only=True,
                                                         monitor='val_accuracy',
                                                         mode='max', verbose=0)


        kwargs = {'verbose':2, 'epochs':var.epochs, 'initial_epoch':total_epochs, 'shuffle':False}
        if test:
            kwargs['callbacks'] = [cp_callback]
        else:
            kwargs['callbacks'] = [cp_callback, WandbCallback()]
        kwargs = set_model_hyperparams(var.epochs, kwargs, plot_lr_rate, decrease_lr_rate,
                                       validation, test_dataset, generator)

        if generator:
            ### Parellelize loading
            history = model.fit(x=train_dataset, **kwargs)
        else:
            batch_size = 100
            history = model.fit(x, y, batch_size=batch_size, **kwargs)

        model.save(checkpoint_path_model)
        print('\n---------------')
        total_epochs = var.epochs
        print('\ntotal_epochs',total_epochs)
        gc.collect()

        for key, value in history.history.items():
            all_history.setdefault(key, [])
            all_history[key] += value
        
        # if the val_loss or val metric imporve throughout epoch, continue training
        epoch_size = 150
        one_third = total_epochs - int(epoch_size * (2/3))
        two_third = total_epochs - int(epoch_size * (1/3))
        half = total_epochs - int(epoch_size * (1/2))
        
        mid_epoch_loss = np.mean(all_history['val_loss'][one_third:two_third])
        end_epoch_loss = np.mean(all_history['val_loss'][two_third:])
        best_loss_end_of_epoch = np.argmin(all_history['val_loss']) > half
        
        mid_epoch_metric = np.mean(all_history[f'val_{metric}'][one_third:two_third])
        end_epoch_metric = np.mean(all_history[f'val_{metric}'][two_third:])
        if metric == 'accuracy':
            best_metric_end_of_epoch = np.argmax(all_history[f'val_{metric}']) > half
            metric_better = ((end_epoch_metric - 0.0001) > mid_epoch_metric)
        else:
            best_metric_end_of_epoch = np.argmin(all_history[f'val_{metric}']) > half
            metric_better = ((end_epoch_metric + 0.0001) < mid_epoch_metric)
        
        if (
            (((end_epoch_loss + 0.0001) < mid_epoch_loss) and best_loss_end_of_epoch)
                                              or
            (((end_epoch_metric - 0.0001) > mid_epoch_metric) and best_metric_end_of_epoch)
        ):
            print(f'continue training for another {epoch_size} epochs')
            wb.update({'epochs':total_epochs + epoch_size}, allow_val_change=True)
            del_unneeded_checkpoints(checkpoint_path_base, all_history, metric)
        else:
            break     
    ##################
       
    all_results = {}
    for (epoch_name, ep, man_val_metric) in [('best_epoch_metric', 0, False),
                                             ('best_epoch_loss', 0, 'val_loss'),
                                             ('last_epoch', var.epochs, False)]:      
        man_epoch_idx = ep #Set to 0 or False to choose best accuracy, otherwise choose epoch to load

        model = explore_epoch(metric, man_epoch_idx, man_val_metric, checkpoint_path_base,
                              all_history, model)

        ###
        pip_fees = 1
        review_set = 'test' #'test' 'train' ' all'

        all_returns, all_raw = out_of_sample_results(loaded_files, pip_fees, review_set, model, var)

        ###
        all_returns_final = pd.concat(all_returns, axis=1)
        all_returns_final, suspect_stocks = drop_outliers(all_returns_final)

        print(f'averge profit (after {pip_fees} pip fees):',np.nanmean(all_returns_final))
        all_returns_final['profit'] = all_returns_final.sum(axis=1)
        all_returns_final['returns'] = all_returns_final['profit'].cumsum()
        # all_returns_final['returns'].plot(title='all returns (time scaled)')
        # plt.show()
        # all_returns_final['returns'].reset_index(drop=True).plot(title='all returns (no time)')
        # plt.show()
        
        daily_pct_change = all_returns_final['profit'].resample('1D').sum()
        romad = calc_romad(daily_pct_change, filter_large_trades=False, yearly_agg=np.median,
                           plot=False)
        
        equity_curve = all_returns_final['returns'].reset_index(drop=True).values.tolist()
        all_results[epoch_name] = {
            'romad':romad,
            'returns':equity_curve
        }    
        
    all_raw_final = pd.concat(all_raw, axis=1)
    all_raw_final.drop(suspect_stocks, axis='columns', inplace=True) 
    raw_daily_pct_change = all_raw_final.sum(axis=1).resample('1D').sum()
    romad_raw = calc_romad(raw_daily_pct_change, filter_large_trades=False, yearly_agg=np.median,
                           plot=False)
    raw_equity = raw_daily_pct_change.cumsum().reset_index(drop=True).values.tolist()
        

    if sweep:
        b_e = all_results['best_epoch_metric']['returns']
        l_e = all_results['last_epoch']['returns']
        table1 = wandb.Table(data=zip(list(range(len(b_e))),b_e), columns = ["x", "y"])  
        table2 = wandb.Table(data=zip(list(range(len(l_e))),l_e), columns = ["x", "y"])  
        table3 = wandb.Table(data=zip(list(range(len(raw_equity))), raw_equity), columns = ["x", "y"])  
        print('all_history', all_history.keys())
        metrics = {
            f'max_{metric}': max(all_history[metric]),
            f'max_val_{metric}': max(all_history[f'val_{metric}']),
            'min_val_loss': min(all_history['val_loss']),
            'min_loss': min(all_history['loss']),
            'ROMAD_best_metric': all_results['best_epoch_metric']['romad'],
            'ROMAD_best_loss': all_results['best_epoch_loss']['romad'],
            'ROMAD_last': all_results['last_epoch']['romad'],
            'ROMAD_raw': romad_raw,
        }
        if test:
            print(metrics)
        else:
            metrics.update({
                'custom_plot_best_epoch': wandb.plot.line(table1, "x", "y",
                                                          title="Profit plot best epoch"),
                'custom_plot_last_epoch': wandb.plot.line(table2, "x", "y",
                                                          title="Profit plot last epoch"),
                'custom_plot_raw_benchmark': wandb.plot.line(table3, "x", "y",
                                                             title="Profit plot raw benchmark"),
            })
            wandb.log(metrics)
        
        
if __name__ == '__main__':
    train(test=False)
