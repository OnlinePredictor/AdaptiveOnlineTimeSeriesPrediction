'''
OGD optimizer
'''
import jax.numpy as np
from tigerforecast.utils.optimizers.core import Optimizer
from tigerforecast.utils.optimizers.losses import mse
from tigerforecast import error

class RealOGD(Optimizer):
    """
    Description: Ordinary Gradient Descent optimizer.
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate
    Returns:
        None
    """
    def __init__(self, pred=None, loss=mse, hyperparameters={}):
        self.initialized = False
        self.hps = {"T":10000, "D":1,"G":1,"c":1}
        self.hps.update(hyperparameters)
        for key, value in self.hps.items():
            if hasattr(self, key):
                raise error.InvalidInput("key {} is already an attribute in {}".format(key, self))
            setattr(self, key, value) # store all hyperparameters
        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

    def norm_project(self, y, c):
        """ Project y using norm A on the convex set bounded by c. """
        if np.any(np.isnan(y)) or np.all(np.absolute(y) <= c):
            return y
        y_norm= np.max(np.absolute(y))
        #print(y_norm)
        solution = y/y_norm*c
        return solution

    def reset(self): # reset internal parameters
        self.T = self.T
       

    def update(self, params, x, y, loss=None):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            params (list/numpy.ndarray): Parameters of method pred method
            x (float): input to method
            y (float): true label
            loss (function): loss function. defaults to input value.
        Returns:
            Updated parameters in same shape as input
        """
        assert self.initialized
        assert type(params) == dict, "optimizers can only take params in dictionary format"

        grad = self.gradient(params, x, y, loss=self.loss) # defined in optimizers core class
        if hasattr(self, 'lr'):
            lr=self.lr
        else: 
            lr =self.D/self.G
        new_params = {k:w - lr/np.sqrt(self.T) * dw for (k, w), dw in zip(params.items(), grad.values())}
        norm = self.c
        new_params = {k:self.norm_project(p, norm) for (k,p) in new_params.items()}
        return new_params


    def __str__(self):
        return "<OGD Optimizer, lr={}>".format(self.lr)
