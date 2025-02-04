import numpy as np


class Gaussian_sampler(object):
    def __init__(self, mean, sd=1, N=20000):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean)))
        self.X = self.X.astype('float32')

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean))).astype('float32')

    def load_all(self):
        return self.X

        
def get_ADRF(x_values=None, x_min=None, x_max=None, nb_intervals=None, dataset='Imbens'):
    """
    Compute the values of the Average Dose-Response Function (ADRF).
    
    Parameters
    ----------
    x_values : list or np.ndarray, optional
        A list or array of values at which to evaluate the ADRF.
        If provided, overrides x_min, x_max, and nb_intervals.
    x_min : float, optional
        The minimum value of the range (used when x_values is not provided).
    x_max : float, optional
        The maximum value of the range (used when x_values is not provided).
    nb_intervals : int, optional
        The number of intervals in the range (used when x_values is not provided).
    dataset : str, optional
        The dataset name (default: 'Imbens'). Must be one of {'Imbens', 'Sun', 'Lee'}.
    
    Returns
    -------
    true_values : np.ndarray
        The computed ADRF values.
    
    Notes
    -----
    - Either `x_values` or (`x_min`, `x_max`, `nb_intervals`) must be provided.
    - Supported datasets:
        - 'Imbens': ADRF = x + 2 / (1 + x)^3
        - 'Sun': ADRF = x - 1/2 + exp(-0.5) + 1
        - 'Lee': ADRF = 1.2 * x + x^3
    """
    # Validate dataset name
    valid_datasets = {'Imbens', 'Sun', 'Lee'}
    if dataset not in valid_datasets:
        raise ValueError(f"`dataset` must be one of {valid_datasets}, but got '{dataset}'.")

    # Input validation for x_values or range parameters
    if x_values is not None:
        if not isinstance(x_values, (list, np.ndarray)):
            raise ValueError("`x_values` must be a list or numpy array.")
        x_values = np.array(x_values, dtype='float32')
    elif x_min is not None and x_max is not None and nb_intervals is not None:
        if x_min >= x_max:
            raise ValueError("`x_min` must be less than `x_max`.")
        if nb_intervals <= 0:
            raise ValueError("`nb_intervals` must be a positive integer.")
        x_values = np.linspace(x_min, x_max, nb_intervals, dtype='float32')
    else:
        raise ValueError("Either `x_values` or (`x_min`, `x_max`, `nb_intervals`) must be provided.")
    
    # Compute ADRF values based on the selected dataset
    if dataset == 'Imbens':
        true_values = x_values + 2 / (1 + x_values)**3
    elif dataset == 'Sun':
        true_values = x_values - 0.5 + np.exp(-0.5) + 1
    elif dataset == 'Lee':
        true_values = 1.2 * x_values + x_values**3

    return true_values