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

class GMM_indep_sampler(object):
    def __init__(self, N, sd, dim, n_components, weights=None, bound=1):
        np.random.seed(1024)
        self.total_size = N
        self.dim = dim
        self.sd = sd
        self.n_components = n_components
        self.bound = bound
        self.centers = np.linspace(-bound, bound, n_components)
        self.X = np.vstack([self.generate_gmm() for _ in range(dim)]).T
        self.X_train, self.X_val,self.X_test = self.split(self.X)
        self.nb_train = self.X_train.shape[0]
        self.Y=None
    def generate_gmm(self,weights = None):
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        Y = np.random.choice(self.n_components, size=self.total_size, replace=True, p=weights)
        return np.array([np.random.normal(self.centers[i],self.sd) for i in Y],dtype='float64')
    def split(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
    
    def get_density(self, data):
        assert data.shape[1]==self.dim
        from scipy.stats import norm
        centers = np.linspace(-self.bound, self.bound, self.n_components)
        prob = []
        for i in range(self.dim):
            p_mat = np.zeros((self.n_components,len(data)))
            for j in range(len(data)):
                for k in range(self.n_components):
                    p_mat[k,j] = norm.pdf(data[j,i], loc=centers[k], scale=self.sd) 
            prob.append(np.mean(p_mat,axis=0))
        prob = np.stack(prob)        
        return np.prod(prob, axis=0)

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        return self.X_train[indx, :]

    def load_all(self):
        return self.X, self.Y

#Swiss roll (r*sin(scale*r),r*cos(scale*r)) + Gaussian noise
class Swiss_roll_sampler(object):
    def __init__(self, N, theta=2*np.pi, scale=2, sigma=0.4):
        np.random.seed(1024)
        self.total_size = N
        self.theta = theta
        self.scale = scale
        self.sigma = sigma
        params = np.linspace(0,self.theta,self.total_size)
        self.X_center = np.vstack((params*np.sin(scale*params),params*np.cos(scale*params)))
        self.X = self.X_center.T + np.random.normal(0,sigma,size=(self.total_size,2))
        np.random.shuffle(self.X)
        self.X_train, self.X_val,self.X_test = self.split(self.X)
        self.Y = None
        self.mean = 0
        self.sd = 0

    def split(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
        
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_density(self,x_points):
        assert len(x_points.shape)==2
        c = 1./(2*np.pi*self.sigma)
        px = [c*np.mean(np.exp(-np.sum((np.tile(x,[self.total_size,1])-self.X_center.T)**2,axis=1)/(2*self.sigma))) for x in x_points]
        return np.array(px)

    def load_all(self):
        return self.X, self.Y
        
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