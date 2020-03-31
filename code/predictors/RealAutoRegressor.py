import tigerforecast
import jax
import jax.numpy as np
import jax.experimental.stax as stax
from tigerforecast.utils.random import generate_key
from tigerforecast.methods import Method
from tigerforecast.utils.optimizers import * 
from tigerforecast.utils.optimizers.losses import mse

class RealAutoRegressor(Method):
    """
    Description: Implements the equivalent of an AR(p) method - predicts a linear
    combination of the previous p observed values in a time-series
    """
    
    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = False

    def initialize(self, n = 1, m = None, p = 3, optimizer = OGD):
        """
        Description: Initializes autoregressive method parameters

        Args:
            p (int): Length of history used for prediction
            optimizer (class): optimizer choice
            loss (class): loss choice
            lr (float): learning rate for update
        """
        self.initialized = True
        self.n = n
        self.p = p

        self.past = np.zeros((p, self.n))

        glorot_init = stax.glorot() # returns a function that initializes weights

        # self.params = glorot_init(generate_key(), (p+1,1))
        self.params = {'phi' : glorot_init(generate_key(), (p,1))}

        def _update_past(self_past, x):
            new_past = np.roll(self_past, self.n)
            new_past = jax.ops.index_update(new_past, 0, x)
            return new_past
        self._update_past = jax.jit(_update_past)

        def _predict(params, x):
            phi = list(params.values())[0]
            return np.dot(x.T, phi).squeeze()
        self._predict = jax.jit(_predict)

        self._store_optimizer(optimizer, self._predict)

    def predict(self, x):
        """
        Description: Predict next value given observation
        Args:
            x (int/numpy.ndarray): Observation
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized, "ERROR: Method not initialized!"

        self.past = self._update_past(self.past, x) # squeeze to remove extra dimensions
        return self._predict(self.params, self.past)

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

        self.past = self._update_past(self.past, x)
        past = self.past.copy()
        pred = []

        for t in range(timeline):
            x = self._predict(self.params, past)
            pred.append(x)
            past = self._update_past(past, x) 

        return np.array(pred)

    def update(self, y):
        """
        Description: Updates parameters using the specified optimizer
        Args:
            y (int/numpy.ndarray): True value at current time-step
        Returns:
            None
        """
        assert self.initialized, "ERROR: Method not initialized!"

        self.params = self.optimizer.update(self.params, self.past, y)
