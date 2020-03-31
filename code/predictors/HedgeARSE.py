import random
import tigerforecast
import jax
import jax.numpy as np
import jax.experimental.stax as stax
from tigerforecast.utils.random import generate_key
from tigerforecast.methods import Method
from tigerforecast.utils.optimizers import * 
from tigerforecast.utils.optimizers.losses import mse
from optimizers.SFftrl import SFftrl
from optimizers.FTRL_fast import FTRL_fast
from tigerforecast.methods.autoregressor import AutoRegressor

class HedgeARSE(Method):
    """
    Description: Implements the equivalent of an AR(p) method - predicts a linear
    combination of the previous p observed values in a time-series
    """
    
    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = False
        self.experts=[]
        self.p=100

    def initialize(self, n = 1, m = None, p = 100, c=5, loss = mse):
        self.initialized = True
        for i in range(p):
            ar= AutoRegressor()
            if c>0:
                optimizer = FTRL_fast(loss = loss,hyperparameters={'c':c})
            else:
                optimizer = FTRL_fast(loss = loss,hyperparameters={})
            ar.initialize(n=n,m=m,p=i+1,optimizer=optimizer)        
            self.experts.append(ar)    
        self.p=p
        self.g=np.zeros(p)
        self.theta=np.zeros(p)
        self.theta_max=0.0
        self.eta=0.0
        self.y=np.zeros(p)
        self.w=np.ones(p)/p
        
    def predict(self, x):
        """
        Description: Predict next value given observation
        Args:
            x (int/numpy.ndarray): Observation
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized, "ERROR: Method not initialized!"
        for i in range(self.p):
            self.y=jax.ops.index_update(self.y, i,self.experts[i].predict(x)) 
        return np.dot(self.y,self.w)

    def forecast(self, x, timeline = 1):
        """
        Description: Forecast values 'timeline' timesteps in the future
        Args:
            x (int/numpy.ndarray):  Value at current time-step
            timeline (int): timeline for forecast
        Returns:
            Forecasted values 'timeline' timesteps in the future
        """
        assert self.initialized, "ERROR: Method not initialized!"
        _k = random.randint(0, self.p-1)
        return self.experts[_k].forecast(x,timeline)

    def update(self, y):
        """
        Description: Updates parameters using the specified optimizer
        Args:
            y (int/numpy.ndarray): True value at current time-step
        Returns:
            None
        """
        assert self.initialized, "ERROR: Method not initialized!"
        for i in range(self.p):
            self.g=jax.ops.index_update(self.g, i,self.experts[i].optimizer.loss(y,self.y[i])) 
            self.experts[i].update(y)
        self.theta= self.theta-self.g
        #print('gradient {0}'.format(self.g))
        #print('theta {0}'.format(self.theta))
        g_norm= np.linalg.norm(x=self.g,ord=np.inf)
        theta_norm= np.linalg.norm(x=self.theta,ord=np.inf)
        self.theta_max = np.maximum(theta_norm,self.theta_max) 
        self.eta = self.eta+g_norm**2
        #lr= np.sqrt(np.maximum(self.eta, self.theta_max))
        lr= np.sqrt(self.eta)
        
        if lr==0:
            self.w= self.theta
        else:
            self.w= self.theta/lr
            
        #print('lr {0}'.format(lr))
        #print('w {0}'.format(self.w))
        w_max= np.maximum(np.max(self.w),1.0)
        self.w=np.exp(self.w-w_max)
        w_sum= np.sum(self.w)
        self.w= self.w/w_sum