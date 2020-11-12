# Time Series Models
* **Deep learning**:
  * Deep neural networks (DNN)
  * Convolutional neural networks (CNN)
  * Recurrent neural networks (RNN, GRU, LSTM)
* **Machine learning**:
  * K-nearest neighbors (KNN)
  * Random forest
* **Reinforcement learning**:
  * Deep Q learning (DQN, DDQN)
  * Proximal policy optimization (PPO)
* **Signal processing**:
  * Fast Fourier transform (FFT)  
* **Image processing**:
  * Gramian angular field (GAF)
* **Unsupervised embeddings**:
  * Global vectors for word representation (GloVe). Altered for candlesticks
* **Technical analysis**:
  * Donchian channel (e.g n week high and low) and moving averages

# Deep learning 
* Results
* Full details
## Results
[Click here to **view the interactive report** (weights and biases)](https://wandb.ai/jameshuckle/timeseries-3/reports/Deep-learning-for-time-series--VmlldzozMTY4NDE?accessToken=dcnujazmnsw0ouuas0s6fu8pxywes82rhaed8e64xwytsx2wvs86cilu8a3bydt9)
### Parameter importance
![image](https://www.dropbox.com/s/ms2beoqy0ccjrvf/1.PNG?raw=1)
![image](https://www.dropbox.com/s/3vqbue03adkqqzn/2.PNG?raw=1)
![image](https://www.dropbox.com/s/1edaiq0ozp5dtpf/3.PNG?raw=1)
![image](https://www.dropbox.com/s/0fo28vs8bc8k9we/4.PNG?raw=1)
![image](https://www.dropbox.com/s/4ag7m7ta6hi8a15/5.PNG?raw=1)
![image](https://www.dropbox.com/s/fjh4r2vvuil0x06/6.PNG?raw=1)
### Learning curves
Training data is from 2003-2016 <br>
Validation data is from 2016-2018  <br>
The ideal learning curve has both training and validation loss trending lower whilst training and validation classification accuracy are increasing. Early stopping is utilized to stop training if the rate of change of the validation loss turns positive above a given threshold, or if the rate of change of the validation accuracy turns negative below a given threshold. The best epoch will be selected retrospectively and the model's weights reverted back to that point in time. 
![image](https://www.dropbox.com/s/q528ym8icevph4b/7.PNG?raw=1)
### Out of sample test results
Out of sample test data is from 2019-2020 (time period not used during the training or validation proccess) <br>
Is used to estimate out of sample variance with the results being represented as cumulative percentage returns (percentage equity curve, based at 0).
#### Best 7 models
![image](https://www.dropbox.com/s/7w7jl2719yiexpt/8.PNG?raw=1)
![image](https://www.dropbox.com/s/foyoyhsh7e7q3dk/9.PNG?raw=1)
#### Next best 5 models
![image](https://www.dropbox.com/s/f05uhyi7j5gayso/10.PNG?raw=1)
![image](https://www.dropbox.com/s/e95l51unki0u429/11.PNG?raw=1)

## Full details
### Goals
Succesfully predicting the future price movements of equities, FX, and crypto currency time series. <br>
Time horizon for prediction ranges from one hour to multiple weeks and is broadly a circumstance of the granularity of the data used to train the model. <br>
Correct inference of both the direction and magnitude of a future price movement are of vital importance in building a successful trading strategy. <br>
Volatility is a latent variable which can be inferred from the duration of a price move, its direction, and its magnitude.

### Data
Intial training/testing was conducted on a basket of major FX markets from 2003-2020. <br>
The data represents daily open, high, low, close of the given instruments. <br>
#### Features
Flattened OHLC into a single row. <br>
Sliding window of one day. <br>
![image](https://www.dropbox.com/s/o8layjx21mafj3i/12.PNG?raw=1)
Each OHLC value is normalized against the previous bar's close, creating a percentage difference between the two.
![image](https://www.dropbox.com/s/v97p7qkdl8qsie5/13.PNG?raw=1)
The data is then scaled using standardization or min max normalization which become a model hyperparameter.
![image](https://www.dropbox.com/s/uw60hqopw7talu9/14.PNG?raw=1)
#### Labels
The percentage change from the last input bar’s close to the close of a set number of bars in the future. <br>
This places the importance on the relative magnitude of each price move rather than absolute values.
#### Train / validation / test sets
Careful consideration has gone into arranging the training, validation and test sets as to avoid look ahead bias.
The temperal aspect of time series data is respected by splitting each instrument separately into the following: <br>
Training data is from 2003-2016 <br>
Validation data is from 2016-2018 <br>
Out of sample test data is from 2019-2020 <br>
The instruments are then merged in chronological order.
### Data hyperparameters
**Dimensionality reduction** - PCA with varying degrees of features kept. <br>

### Evaluation
**Trading costs** - A naive figure of one basis point has been added to account for the average transaction fee plus slippage experienced in the institutional market. Whilst transaction costs vary wildly depending on the traded instrument and the amount of negative slippage experienced varies for each trade, one basis point for liquid FX major pairs should be an acceptable benchmark for the first pass. <br>
**Reward function** - ROMAD (return over maximum drawdown). Higher is better. <br>
**Cross fold validation** - Make use of scikit-learn module TimeSeriesSplit which respects or temporal / ordered nature of time series data: <br>
![image](https://www.dropbox.com/s/10r4ca8zh0bpuh3/20.png?raw=1) <br>
Use a custom built function TimeSeriesShift which keeps the training and validation sets the same size, thus making it fairer for evaluations across different folds / time periods, reducing bias in the validation.  <br>
![image](https://www.dropbox.com/s/ntxsq4g0wftaf8i/21.png?raw=1) <br>
The method used for aggregating the performance across the folds includes the following: <br>
* Mean, median, max, min, sum, last
* 1 / standard deviation
* Weighted by size of training fold (TimeSeriesSplit)
* Binary evaluation of whether each successive fold improves on the last, assuming each fold gets larger (TimeSeriesSplit)


# Machine learning 
## Results
### Parameter importance (using cross-fold validation)
![image](https://www.dropbox.com/s/8q23od4vk003knx/15.png?raw=1)
![image](https://www.dropbox.com/s/0xxrux1lxlw5n71/16.jpg?raw=1)
![image](https://www.dropbox.com/s/mjyernm1f5x7t64/17.PNG?raw=1)
### Out of sample test results
Out of sample test data is from 2019-2020 (time period not used during the training or validation proccess) <br>
Is used to estimate out of sample variance with the results being represented as cumulative percentage returns (percentage equity curve, based at 0).

![image](https://www.dropbox.com/s/t5164tbz405l0mz/18.png?raw=1)
![image](https://www.dropbox.com/s/i6u36wc4gnhfhgv/19.png?raw=1)

## Full details
### Goals
See deep learning goals

### Data
See deep learning data

#### Features
See deep learning features

#### Labels
See deep learning labels <br>
Generating a range of label time horizons at once leverages KNN's ability to perform a similarity / distance lookup between the features once and then use the indices of the found neighbors to greatly speed up the testing of the other time horizons asynchronously without the need for further lookup.

#### Train / validation / test sets
See deep learning Train / validation / test sets <br>

#### Hyperparameters


