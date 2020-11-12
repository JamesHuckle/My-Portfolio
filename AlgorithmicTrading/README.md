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

## Results (DNN, LSTM)
[Click here to **view the interactive report** (weights and biases)](https://wandb.ai/jameshuckle/timeseries-3/reports/Deep-learning-for-time-series--VmlldzozMTY4NDE?accessToken=dcnujazmnsw0ouuas0s6fu8pxywes82rhaed8e64xwytsx2wvs86cilu8a3bydt9)

### Parameter importance
![image](https://www.dropbox.com/s/ms2beoqy0ccjrvf/1.PNG?raw=1)
![image](https://www.dropbox.com/s/3vqbue03adkqqzn/2.PNG?raw=1)
![image](https://www.dropbox.com/s/1edaiq0ozp5dtpf/3.PNG?raw=1)
![image](https://www.dropbox.com/s/0fo28vs8bc8k9we/4.PNG?raw=1)
![image](https://www.dropbox.com/s/4ag7m7ta6hi8a15/5.PNG?raw=1)
![image](https://www.dropbox.com/s/fjh4r2vvuil0x06/6.PNG?raw=1)

### Learning curves
Training data is from 2003-2016. <br>
Validation data is from 2016-2018.  <br>
The ideal learning curve has both training and validation loss trending lower whilst training and validation classification accuracy are increasing. Early stopping is utilized to stop training if the rate of change of the validation loss turns positive above a given threshold, or if the rate of change of the validation accuracy turns negative below a given threshold. The best epoch is selected retrospectively and the model's weights reverted back to that point in time to run out of sample tests on. 
![image](https://www.dropbox.com/s/q528ym8icevph4b/7.PNG?raw=1)

### Out of sample test results
Out of sample test data is from 2019-2020 (a time period not used during the training or validation proccess) <br>
Test data is used to estimate the out of sample variance, with the results being represented as cumulative percentage returns (percentage equity curve, based at 0).
#### Best 7 models
![image](https://www.dropbox.com/s/7w7jl2719yiexpt/8.PNG?raw=1)
![image](https://www.dropbox.com/s/foyoyhsh7e7q3dk/9.PNG?raw=1)
#### Next best 5 models
![image](https://www.dropbox.com/s/f05uhyi7j5gayso/10.PNG?raw=1)
![image](https://www.dropbox.com/s/e95l51unki0u429/11.PNG?raw=1)

## Full details

### Goals
Succesfully predicting the future price movements of equities, FX, and crypto currency time series data. <br>
The time horizon for prediction ranges from one hour to multiple weeks and is broadly a circumstance of the granularity of the data used to train the model. <br>
Correct inference of both the direction and magnitude of a future price movement is of vital importance in building a successful trading strategy. <br>
Volatility is a latent variable which can be inferred from the duration of a price move, its direction, and its magnitude.

### Intuition behind DNN and LSTM for time series

### Data
Intial training/testing was conducted on a basket of major FX markets from 2003-2020. <br>
The data represents daily open, high, low, close of the given instruments. <br>
#### Features
Flattened OHLC into a single row. <br>
Each row is shited forward one time period (e.g sliding window of one day). <br>
![image](https://www.dropbox.com/s/o8layjx21mafj3i/12.PNG?raw=1)
Each OHLC value is normalized against the previous bar's close, creating a percentage difference between the two.
![image](https://www.dropbox.com/s/v97p7qkdl8qsie5/13.PNG?raw=1)
The data is then scaled using standardization or min max normalization. The choice of which becomes a model hyperparameter. <br>
![image](https://www.dropbox.com/s/uw60hqopw7talu9/14.PNG?raw=1)
#### Labels
Are the percentage change from the last input bar’s close to the close of a set number of bars in the future. <br>
This places the importance on the relative magnitude of each price move rather than absolute values. <br>
#### Train, validation and test sets
Careful consideration has gone into arranging the training, validation and test sets as to avoid look ahead bias.
The temperal aspect of time series data is respected by splitting each instrument separately into the following: <br>
 Training data is from 2003-2016. <br>
 Validation data is from 2016-2018. <br>
 Out of sample test data is from 2019-2020. <br>
The instruments are then merged in chronological order.

### Data hyperparameters
**Dimensionality reduction** - PCA with varying degrees of components. <br>

### Evaluation
#### Trading costs
A naive figure of one basis point has been added to account for the average transaction fee plus slippage experienced in the institutional market. Whilst transaction costs vary wildly depending on the traded instrument and the amount of negative slippage experienced varies for each trade, one basis point for liquid FX major pairs should be an acceptable benchmark of the round trip cost. <br>
#### Reward function
ROMAD (return over maximum drawdown). Higher is better. <br>

### Model

# Machine learning

## Results (KNN)

### Parameter importance (using cross-fold validation)
![image](https://www.dropbox.com/s/8q23od4vk003knx/15.png?raw=1)
![image](https://www.dropbox.com/s/0xxrux1lxlw5n71/16.jpg?raw=1)
![image](https://www.dropbox.com/s/mjyernm1f5x7t64/17.PNG?raw=1)
### Out of sample test results
Out of sample test data is from 2019-2020 (a time period not used during the training or validation proccess) <br>
Test data is used to estimate the out of sample variance, with the results being represented as cumulative percentage returns (percentage equity curve, based at 0).

![image](https://www.dropbox.com/s/t5164tbz405l0mz/18.png?raw=1)
![image](https://www.dropbox.com/s/i6u36wc4gnhfhgv/19.png?raw=1)

## Full details
Many of the details are the same for both the deep learning models and for the KNN model, so I will only talk about the notable differences.

### Intuition behind KNN for time series
Given a window of ten OHLC candlesticks representing a test input, we can find the closest five neighbors that match it’s pattern and then take the average future movements of the next bar of those five neighbors to make a prediction on the future movement of our test input. <br>

Test input window of 10 candlesticks:  <br>
![image](https://www.dropbox.com/s/ndasy1rlknvl8xj/24.png?raw=1)
**Test label = 0.001131 % = 1.1e-3 %** <br>

Five nearest neighbors: <br>
![image](https://www.dropbox.com/s/jv8ifga315mlldh/23.png?raw=1)
**Neighbors labels (mean) = 0.00419 % = 4.2e-3 %** <br>
In this example he five nearest neighbors correctly predicted a positive next bar for our test example! KNN gives us the ability to find similar patterns that have occurred in the past to the one we face in the present. By averaging what has happened in the past we can attempt to predict the future movement.


### Data
#### Labels
See deep learning labels <br>
Generating a range of label time horizons at once leverages KNN's ability to perform a similarity/distance lookup between the features once and then use the indices of the found neighbors to greatly speed up the testing of the other time horizons asynchronously without the need for further lookup.

### Data hyperparameters
**Dimensionality reduction** - PCA with varying degrees of features kept. <br>

### Evaluation
#### Cross fold validation
Make use of the scikit-learn module TimeSeriesSplit which respects the temporal/ordered nature of time series data: <br>
![image](https://www.dropbox.com/s/10r4ca8zh0bpuh3/20.png?raw=1) <br>
Another option is to use a custom built function TimeSeriesShift which keeps the training and validation sets the same size, thus making it fairer for evaluations across different folds/time periods, reducing bias in the validation. <br>
![image](https://www.dropbox.com/s/ntxsq4g0wftaf8i/21.png?raw=1) <br>
The method used for aggregating the performance across the folds includes the following: <br>
* Mean, median, max, min, sum, last
* 1 / standard deviation
* Weighted by the size of training fold (TimeSeriesSplit)
* Binary evaluation of whether each successive fold improves on the last, assuming each fold gets larger (TimeSeriesSplit)

### Model
#### Ensemble
Combining the predictions made with a combination of the number of sets of model hyperparameters. This should lead to a more robust overall prediction. There are a number of possible methods to implement an simple ensemble:
A.	Combine the predictions and threshold them, for example, there needs to be x% of buys before considering it a buy, otherwise there is no trade. <br>
B.	Take the simple median or mean of the set of predictions. <br>
C.	Give a direction only when both mean and median are in the same direction (vote), otherwise don’t trade. <br>

