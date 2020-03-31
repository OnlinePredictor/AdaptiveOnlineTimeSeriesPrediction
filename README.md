# Adaptive Online Time Series Prediction

# Requirements

TigerForecast is required, as there are many benchmarks and algorithms already implemented.
- clone [TigerForecast](https://github.com/MinRegret/TigerForecast) 

    ```git clone https://github.com/MinRegret/TigerForecast.git```
- checkout commit ae18b169d96dd81db88ab27a8b055036845d3a8f 

    ```git checkout ae18b16```
- follow the instructions in the readme file and install TigerForecast 

- clone this project 

    ```git clone https://github.com/OnlinePredictor/AdaptiveOnlineTimeSeriesPrediction.git```
- to run the experiments later you need to install joblib 

    ```pip install joblib```

# Overview

This repository contains different python files. 

These files define our problems we used to test our algorithm.
1. [ExperimentSetting2.py](code/problems/ExperimentSetting2.py)
2. [ExperimentSetting3.py](code/problems/ExperimentSetting3.py)
3. [ExperimentSetting4.py](code/problems/ExperimentSetting4.py)
4. [ExperimentSetting5.py](code/problems/ExperimentSetting5.py)
5. [ExperimentSetting6.py](code/problems/ExperimentSetting6.py)
6. [ExperimentSetting8.py](code/problems/ExperimentSetting8.py)

The following files contain the predictors.
1. [ArimaAutoregressor.py](code/predictors/ArimaAutoregressor.py)
2. [HedgeAR.py](code/predictors/HedgeAR.py)
3. [HedgeARSE.py](code/predictors/HedgeARSE.py)
4. [RealAutoRegressor.py](code/predictors/RealAutoRegressor.py)

Optimizers:
1. [FTRL_fast.py](code/optimizers/FTRL_fast.py)
2. [RealOGD.py](code/optimizers/RealOGD.py)
3. [RealONS.py](code/optimizers/RealONS.py)
4. [SFftrl.py](code/optimizers/SFftrl.py)

To optimize the hyperparameters for the experiments we used the [Tuning.py](code/Tuning.py). [AE.py](code/losses/AE.py) defines the AE loss. [RealCore.py](code/environment/RealCore.py) is an extended experiment executor which enables us to run our experiments. Furthermore, [RealExperiment.py](code/environment/RealExperiment.py) extends the experiment definition.
Most of these python files are extended versions out of the [TigerForecast project](https://github.com/MinRegret/TigerForecast).



# Run experiments

To reproduce the results in our paper run [PaperExperiments.py](code/PaperExperiments.py).
```python PaperExperiments.py```
If you do not want to run all experiments adjust the tasklist array in the main method.

# Hyperparameter selection

For all of these experiments we used Hedge-FTRL and Hedge-ONS both with <img src="https://render.githubusercontent.com/render/math?math=M = 16"> and ARMA or ARIMA with ONS or OGD as optimizers. To select the best hyperparameters for the baselines (ARMA and ARIMA) a grid search is done for the first 100 iterations to find the lowest average squared or average absolute error. 
For <img src="https://render.githubusercontent.com/render/math?math=m"> we tried values in a set <img src="https://render.githubusercontent.com/render/math?math=%5C%7B8%2C%2016%2C%2032%2C%2064%5C%7D">. In the case of OGD we chose the learning rate among the values in the set <img src="https://render.githubusercontent.com/render/math?math=a = \{10^i | i \in \{-4, -3,\dots, 1\}\}"> and selected the best combination with <img src="https://render.githubusercontent.com/render/math?math=m">. In the case of ONS we tested all combinations for <img src="https://render.githubusercontent.com/render/math?math=\eta"> and <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> where both value were in the set <img src="https://render.githubusercontent.com/render/math?math=a"> and selected the best combination with a corresponding <img src="https://render.githubusercontent.com/render/math?math=m"> again.

This hyperparameter selection is implemented in [Tuning.py](code/Tuning.py). To start the tuning process run ```python Tuning.py```. The results are already transferred into the [PaperExperiments.py](code/PaperExperiments.py) though small differences could occur.

# Experiments

### Setting 1
In the first setting the coefficient vector <img src="https://render.githubusercontent.com/render/math?math=%5Calpha%20%3D%20%5Cbegin%7Bbmatrix%7D0.6%20%26%20-0.5%20%26%200.4%20%26%20-0.4%20%26%200.3%5Cend%7Bbmatrix%7D"> was used. The noise depends on the two last noise terms following <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bbmatrix%7D0.3%20%26%20-0.2%5Cend%7Bbmatrix%7D"> and is uncorrelated and normally distributed as <img src="https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BN%7D(0%2C0.3%5E2)">.

### Setting 2
In contrast to the previous setting, in the second one <img src="https://render.githubusercontent.com/render/math?math=\alpha"> depends on <img src="https://render.githubusercontent.com/render/math?math=t">. With that, the vector slightly changes over time. Furthermore, the error term is distributed uniformly on the interval <img src="https://render.githubusercontent.com/render/math?math=\[-0.5, 0.5]">. The coefficient vectors are defined as <img src="https://render.githubusercontent.com/render/math?math=%5Calpha(t)%20%3D%20%5Cbegin%7Bbmatrix%7D-0.4%20%26%20-0.5%20%26%200.4%20%26%200.4%20%26%200.1%5Cend%7Bbmatrix%7D*(%5Cfrac%7Bt%7D%7B10%5E4%7D)%2B%5Cbegin%7Bbmatrix%7D0.6%20%26%20-0.4%20%26%200.4%20%26%20-0.5%20%26%200.4%5Cend%7Bbmatrix%7D*(1-%5Cfrac%7Bt%7D%7B10%5E4%7D)"> and <img src="https://render.githubusercontent.com/render/math?math=\beta = [0.32, -0.2]">.

### Setting 3
In the third setting we use a non-stationary ARMA-process where the the coefficient vectors change abruptly after some time. During the first half the coefficient vectors from setting 1 <img src="https://render.githubusercontent.com/render/math?math=%5Calpha%20%3D%20%5Cbegin%7Bbmatrix%7D0.6%20%26%20-0.5%20%26%200.4%20%26%20-0.4%20%26%200.3%5Cend%7Bbmatrix%7D"> and <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta%20%3D%20%5Cbegin%7Bbmatrix%7D0.3%20%26%20-0.2%5Cend%7Bbmatrix%7D"> are used. After the first half of the iterations this is changed to <img src="https://render.githubusercontent.com/render/math?math=%5Calpha%20%3D%20%5Cbegin%7Bbmatrix%7D-0.4%20%26%20-0.5%20%26%200.4%20%26%200.4%20%26%200.1%5Cend%7Bbmatrix%7D">. In this experiment the noise is uncorrelated and distributed uniformly on the interval <img src="https://render.githubusercontent.com/render/math?math=\[-0.5, 0.5]">.

### Setting 4

The data in this setting is generated by a stationary ARIMA with <img src="https://render.githubusercontent.com/render/math?math=%5Calpha%20%3D%20%5Cbegin%7Bbmatrix%7D0.6%20%26%20-0.5%20%26%200.4%20%26%20-0.4%20%26%200.3%5Cend%7Bbmatrix%7D">, <img src="https://render.githubusercontent.com/render/math?math=%5Cbeta%20%3D%20%5Cbegin%7Bbmatrix%7D0.3%20%26%20-0.2%5Cend%7Bbmatrix%7D"> and <img src="https://render.githubusercontent.com/render/math?math=d%3D2">. The noise terms are drawn from <img src="https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BN%7D(0%2C%200.3)">.

 ### Setting 5

In this setting, we consider a mixture of ARMA and ARIMA models. The first half of the time series is generated by the ARIMA model of **Setting 4**, while the second half is generated by the ARMA model of **Setting 1**.

 ### Setting 6

 We used the [S&P 500 dataset](https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) which contains time series data describing the daily returns of the index. This is a stock index consisting of the 500 largest US-companies listed on the stock market.

 ### Setting 7

 As a second real-world time series the [unemployment dataset](https://fred.stlouisfed.org/series/UNRATE) was used. It contains the percentual unemployment rate in the US. The dataset contains monthly data between 01.01.1948 and 01.06.2019 (858 samples).

 ### Setting 8

 As a third real-world time series dataset we used the [number of passenger car registrations in the US](https://fred.stlouisfed.org/series/USASACRMISMEI). The dataset contains monthly data between 1960 and 2019.