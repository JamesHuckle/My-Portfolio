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

# Deep learning results
[Click here to **view the interactive report** (weights and biases)](https://wandb.ai/jameshuckle/timeseries-3/reports/Deep-learning-for-time-series--VmlldzozMTY4NDE?accessToken=dcnujazmnsw0ouuas0s6fu8pxywes82rhaed8e64xwytsx2wvs86cilu8a3bydt9)
## Parameter importance
![image](https://www.dropbox.com/s/ms2beoqy0ccjrvf/1.PNG?raw=1)
![image](https://www.dropbox.com/s/3vqbue03adkqqzn/2.PNG?raw=1)
![image](https://www.dropbox.com/s/1edaiq0ozp5dtpf/3.PNG?raw=1)
![image](https://www.dropbox.com/s/0fo28vs8bc8k9we/4.PNG?raw=1)
![image](https://www.dropbox.com/s/4ag7m7ta6hi8a15/5.PNG?raw=1)
![image](https://www.dropbox.com/s/fjh4r2vvuil0x06/6.PNG?raw=1)
## Learning curves
Training data is from 2003-2016 <br>
Validation data is from 2016-2018  <br>
The ideal learning curve has both training and validation loss trending lower whilst training and validation classification accuracy are increasing. Early stopping is utilized to stop training if the rate of change of the validation loss turns positive above a given threshold, or if the rate of change of the validation accuracy turns negative below a given threshold. The best epoch will be selected retrospectively and the model's weights reverted back to that point in time. 
![image](https://www.dropbox.com/s/q528ym8icevph4b/7.PNG?raw=1)
## Results
Out of sample test data is from 2019-2020 (time period not used during the training or validation proccess) <br>
Is used to estimate out of sample variancew with the results being represented as cumulative percentage returns (percentage equity curve, based at 0).
### Best 7 models
![image](https://www.dropbox.com/s/7w7jl2719yiexpt/8.PNG?raw=1)
![image](https://www.dropbox.com/s/foyoyhsh7e7q3dk/9.PNG?raw=1)
### Next best 5 models
![image](https://www.dropbox.com/s/f05uhyi7j5gayso/10.PNG?raw=1)
![image](https://www.dropbox.com/s/e95l51unki0u429/11.PNG?raw=1)

# Machine learning results


