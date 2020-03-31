'''
SF-FTRL optimizer
'''
import jax.numpy as np
from tigerforecast.utils.optimizers.core import Optimizer
from tigerforecast.utils.optimizers.losses import mse
from tigerforecast import error
import jax

class SFftrl(Optimizer):
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
        self.hyperparameters = {}
        self.hyperparameters.update(hyperparameters)
        for key, value in self.hyperparameters.items():
            if hasattr(self, key):
                raise error.InvalidInput("key {} is already an attribute in {}".format(key, self))
            setattr(self, key, value) # store all hyperparameters
        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)
        self.theta = None
        self.eta = None
        self.theta_max = None
        
    def norm_project(self, a,x,y,w):
        
        """ Project y using norm A on the convex set bounded by c. """
        
        if hasattr(self, 'c'):
            if abs(y)>self.c:
                if y>0:
                    cof= (abs(y)-self.c)/np.sum(x*a*x)
                else:
                    cof= (self.c-abs(y))/np.sum(x*a*x)   
                w_p=np.where(np.equal(0.0, a), 0.0, cof*(a*x))
                w=w-w_p
        return w

    def reset(self): # reset internal parameters
        self.theta = None
        self.eta = None
        self.theta_max = None

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

        
        grad = self.gradient(params, x, y, loss=loss) # defined in optimizers core class
        
        if self.theta is None:
            self.theta =  {k: -dw for (k, w), dw in zip(params.items(), grad.values())}
        else:
            self.theta =  {k: v-dw for (k, v), dw in zip(self.theta.items(), grad.values())}
        
        if self.eta is None:
            self.eta =  {k: dw*dw for (k, w), dw in zip(params.items(), grad.values())}
        else:
            self.eta =  {k: v+dw*dw for (k, v), dw in zip(self.eta.items(), grad.values())}
            
        if self.theta_max is None:
            self.theta_max =  {k: np.absolute(v) for (k,v) in self.theta.items()}
        else:
            self.theta_max =  {k: np.where(np.greater(np.absolute(v), v_max),np.absolute(v),v_max) for (k, v), v_max in zip(self.theta.items(), self.theta_max.values())}
            
        new_params = {k: np.where(np.equal(0.0, np.maximum(theta_max,eta)), theta, theta/np.sqrt(np.maximum(theta_max,eta))) for (k, w), theta,theta_max, eta in zip(params.items(),self.theta.values(),self.theta_max.values(),self.eta.values())}

        
        x_new = np.roll(x, 1)
        x_new = jax.ops.index_update(x_new, 0, y)
        y_t = self.pred(params=new_params, x=x_new)
        
#        print('y before {0}'.format(y_t))
        x_plus_bias_new = np.vstack((np.ones((1, 1)), x_new))
        new_mapped_params = {k:self.norm_project( np.where(np.equal(0.0, np.maximum(theta_max,eta)), 0.0, 1.0/np.sqrt(np.maximum(theta_max,eta))) ,x_plus_bias_new,y_t,p) for (k,p),theta_max,eta in  zip(new_params.items(),  self.theta_max.values(), self.eta.values())}
        
#        y_t = self.pred(params=new_mapped_params, x=x_new)
#        print('y after {0}'.format(y_t))
        return new_mapped_params


    def __str__(self):
        return "<SFftrl Optimizer, lr={}>".format(self.lr)



