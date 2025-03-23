import numpy as np
import scipy.linalg as linalg
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_low_rank_matrix


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
        
def simulate_regression(n_samples, n_features, n_targets, effective_rank=None, variance=None, random_state=123):
    np.random.seed(random_state)
    if effective_rank is None:
        X = np.random.normal(size=(n_samples, n_features))
    else:
        X = 100*make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=effective_rank, random_state=random_state)

    X_aug = np.c_[np.ones(n_samples), X]  # n x (p+1) matrix
    beta = 0.1 * np.random.uniform(low=0.0, high=1.0, size=(1+n_features, n_targets))    # Coefficients for the mean (includes intercept)
    mu = np.dot(X_aug, beta)
    if variance is None:
        variance = 0.01*np.mean(X**2, axis=1)
    variance = np.tile(variance,(n_targets,1)).T
    Y = np.random.normal(loc=mu, scale=np.sqrt(variance))
    return X, Y


def simulate_low_rank_data(n_samples=10000, z_dim=2, x_dim=4, rank=2, sigma_z=False, random_state=123):
    """
    Simulate data from a generative process where:
      - Z ~ N(0, I)
      - X | Z ~ N(μ(Z), Σ(Z))
    
    Two options for Σ(Z):
      1. Constant covariance:
            Σ = diag(diag_values) + W W^T,
         where W is fixed.
      2. Z-dependent covariance:
            Σ(z) = diag(diag_values) + (W * z[0]) (W * z[0])^T,
         so that the low-rank component is scaled by the first element of z.
    
    Args:
        n_samples (int): Number of samples.
        z_dim (int): Dimension of latent variable Z.
        x_dim (int): Dimension of observed variable X.
        rank (int): Rank for the low-rank component.
        sigma_z (bool): If True, covariance Σ becomes a function of z (using z[0]); 
                               otherwise, Σ is constant.
        random_state (int): Seed for reproducibility.
    
    Returns:
        X (np.ndarray): Observed data of shape (n_samples, x_dim).
        Z (np.ndarray): Latent variables of shape (n_samples, z_dim).
        (For reference, in the constant case, the marginal of X is 
         N(b, A A^T + Σ) where A and b define the mean function.)
    """
    np.random.seed(random_state)

    # Generate Z ~ N(0, I)
    Z = np.random.randn(n_samples, z_dim).astype(np.float32) 

    # Set fixed parameters for the simulation
    # Randomly generate A and b, or choose them to illustrate particular structure.
    A = np.array([[ 1.0, -0.5],
                  [ 0.3,  0.8],
                  [-0.7,  0.2],
                  [ 0.5,  1.0]])
    b = np.array([0.0, 0.5, 1.0, 2.0])

    # Compute the mean for X given Z: μ(Z) = A Z + b
    mu = Z.dot(A.T) + b  # Shape: (n_samples, x_dim)

    W = np.array([[ 1., 0.],
                  [ 1., 0.],
                  [ 0., 1.],
                  [ 0,  1.]])
    diag_values = np.array([0.2, 0.2, 1., 1.])
    D = np.diag(diag_values)
    
    X = np.zeros((n_samples, x_dim), dtype=np.float32)
    
    for i in range(n_samples):
        if sigma_z:
            scale_factor = Z[i, 0]
            W_scaled = W * scale_factor  # Element-wise multiplication
            Sigma = D + np.dot(W_scaled, W_scaled.T)
        else:
            # Use constant covariance: Σ = D + W W^T
            Sigma = D + np.dot(W, W.T)

        # Sample X_i ~ N(μ_i, Σ)
        X[i] = np.random.multivariate_normal(mean=mu[i], cov=Sigma)
    
    return X, Z

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


def slice_y(y, n_slices=10):
    """Determine non-overlapping slices based on the target variable, y.

    Parameters
    ----------
    y : array_like, shape (n_samples,)
        The target values (class labels in classification, real numbers
        in regression).

    n_slices : int (default=10)
        The number of slices used when calculating the inverse regression
        curve. Truncated to at most the number of unique values of ``y``.

    Returns
    -------
    slice_indicator : ndarray, shape (n_samples,)
        Index of the slice (from 0 to n_slices) that contains this
        observation.
    slice_counts :  ndarray, shape (n_slices,)
        The number of counts in each slice.
    """
    unique_y_vals, counts = np.unique(y, return_counts=True)
    cumsum_y = np.cumsum(counts)

    # `n_slices` must be less-than or equal to the number of unique values
    # of `y`.
    n_y_values = unique_y_vals.shape[0]
    if n_y_values == 1:
        raise ValueError("The target only has one unique y value. It does "
                         "not make sense to fit SIR or SAVE in this case.")
    elif n_slices >= n_y_values:
        if n_slices > n_y_values:
            warnings.warn(
                "n_slices greater than the number of unique y values. "
                "Setting n_slices equal to {0}.".format(counts.shape[0]))
        # each y value gets its own slice. usually the case for classification
        slice_partition = np.hstack((0, cumsum_y))
    else:
        # attempt to put this many observations in each slice.
        # not always possible since we need to group common y values together
        n_obs = np.floor(y.shape[0] / n_slices)

        # Loop through the unique y value sums and group
        # slices together with the goal of 2 <= # in slice <= n_obs
        # Find index in y unique where slice begins and ends
        n_samples_seen = 0
        slice_partition = [0]  # index in y of start of a new slice
        while n_samples_seen < y.shape[0] - 2:
            slice_start = np.where(cumsum_y >= n_samples_seen + n_obs)[0]
            if slice_start.shape[0] == 0:  # this means we've reached the end
                slice_start = cumsum_y.shape[0] - 1
            else:
                slice_start = slice_start[0]

            n_samples_seen = cumsum_y[slice_start]
            slice_partition.append(n_samples_seen)

    # turn partitions into an indicator
    slice_indicator = np.ones(y.shape[0], dtype='int64')
    for j, (start_idx, end_idx) in enumerate(
            zip(slice_partition, slice_partition[1:])):

        # this just puts any remaining observations in the last slice
        if j == len(slice_partition) - 2:
            slice_indicator[start_idx:] = j
        else:
            slice_indicator[start_idx:end_idx] = j

    slice_counts = np.bincount(slice_indicator)
    return slice_indicator, slice_counts

def get_SDR_dim(X, y, n_slices = 10, ratio = 0.8):
    '''
    Calculate the SDR dimension of the X.
    Input:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples, 1)
        n_slices: int, the number of slices used when calculating the inverse regression curve.
    Output:
        dim: int, the SDR dimension of X.
    '''
    if len(y.shape) == 2:
        assert y.shape[1] == 1, "The shape of y should be (n_samples, 1)."
        y = np.squeeze(y)
    n_samples, n_features = X.shape

    # normalize the data
    X = X - np.mean(X, axis=0)
    Q, R = linalg.qr(X, mode='economic')
    Z = np.sqrt(n_samples) * Q
    Z = Z[np.argsort(y), :]

    # determine slice indices and counts per slice
    slices, counts = slice_y(y, n_slices)

    # Sums an array by groups. Groups are assumed to be contiguous by row.
    inv_idx = np.concatenate(([0], np.diff(slices).nonzero()[0] + 1))
    Z_sum = np.add.reduceat(Z, inv_idx)
    # means in each slice (sqrt factor takes care of the weighting)
    Z_means = Z_sum / np.sqrt(counts.reshape(-1, 1))
    
    M = np.dot(Z_means.T, Z_means) / n_samples
    # eigen-decomposition of slice matrix
    evals, evecs = linalg.eigh(M)
    evals = evals[::-1]
    #n_directions = np.argmax(np.abs(np.diff(evals))) + 1
    total_sum = np.sum(evals)
    cumulative_sum = np.cumsum(evals)
    threshold_index = np.argmax(cumulative_sum >= ratio * total_sum)
    n_directions = threshold_index + 1
    return n_directions

def estimate_latent_dims(x,y,v, v_ratio = 0.7, z0_dim=3, min_dim = 1):
    v = StandardScaler().fit_transform(v)
    y = StandardScaler().fit_transform(y)
    z1_dim = get_SDR_dim(v, y, n_slices=10, ratio=0.8)
    z2_dim = get_SDR_dim(v, x, n_slices=10, ratio=0.8)
    pca = PCA().fit(v)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    threshold_index = np.argmax(cumulative_variance >= v_ratio)
    total_z_dim = threshold_index + 1
    z3_dim = total_z_dim - z0_dim - z1_dim - z2_dim
    if z3_dim<=0:
        z3_dim = min_dim
    return [z0_dim, z1_dim, z2_dim, z3_dim]