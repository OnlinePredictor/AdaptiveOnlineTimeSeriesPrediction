import jax.numpy as np 

def ae(y_pred, y_true):
    ''' Description: mean-absolut-error loss
        Args:
            y_pred : value predicted by method
            y_true : ground truth value
            eps: some scalar
    '''
    a = np.absolute(np.array([y_pred]) - np.array([y_true]))[0]
    if(a.shape == (1,)):	#y_pred is sometimes not a scalar but a (1,) vector which causes problems. This does fix the problem.
    	return a[0]
    return a