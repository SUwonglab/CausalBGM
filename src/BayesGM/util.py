import numpy as np
import math
import os
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
from scipy.sparse import diags
from scipy.stats import norm


def make_swiss_roll(n_samples=100, noise=(0.1, 0.6), random_state=None):
    """Generate a swiss roll dataset.
    Parameters
    ----------
    n_samples : int, default=100
        The number of sample points on the Swiss Roll.

    random_state : int, RandomState instance or None, default=None.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The points.

    t : ndarray of shape (n_samples,)
        The univariate position of the sample according to the main dimension
        of the points in the manifold.
    """
    np.random.seed(random_state)
    t = 1.5 * np.pi * (1 + 2 * np.random.uniform(size=n_samples))
    y = 10 * np.random.uniform(size=n_samples)

    x = 0.5*t * np.cos(t)
    z = 0.5*t * np.sin(t)
    X = np.vstack((x, y, z))
    lb, ub = noise
    sd_t = ((t-np.min(t))*(ub-lb))/(np.max(t)-np.min(t))+lb
    X += np.random.normal(0,sd_t,size=(3, n_samples))
    X = X.T
    return X, t

def make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.0, random_state=None):
    """Generate isotropic Gaussian blobs for clustering.
    Parameters
    ----------
    n_samples : int, default=100
        The total number of points equally divided among clusters.

    n_features : int, default=2
        The number of features for each sample.

    centers : int or array of shape [n_centers, n_features], default=3
        The number of centers to generate, or the fixed center locations.

    cluster_std : float or sequence of floats, default=1.0
        The standard deviation of the clusters.

    random_state : int, RandomState instance or None, default=None.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each sample.
    """
    np.random.seed(random_state)
    X, y = datasets.make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std)
    return X, y

def make_sim_data(n_samples=2000, x_dim=5, y_dim=1, z_dim=5, w_dim=5, order='linear', random_state=None):
    """Generate simulation data for regression.
    Parameters
    ----------
    n_samples : int, default=2000
        The total number of points.

    x_dim : int, default=5
        The number of features for X.

    y_dim : int, default=1
        The number of features for Y.

    z_dim : int, default=5
        The number of features for Z.

    w_dim : int, default=4
        The number of features for W.

    order : int, default=4
        The number of features for W.

    random_state : int, RandomState instance or None, default=None.

    Returns
    -------
    X : ndarray of shape (n_samples, x_dim)
        The predictor variable samples.

    y : ndarray of shape (n_samples, y_dim)
        The response variable samples.
    """
    np.random.seed(random_state)
    assert z_dim >= w_dim, "The dimension of Z must be greater than or equal to the dimension of W."
    Z = np.random.normal(size=(n_samples, z_dim))
    X = np.random.normal(loc=Z, scale=0.5).astype('float32')
    W = np.random.uniform(low=-1, high=1, size=(w_dim, y_dim))
    if order == 'linear':
        center = np.dot(Z[:,:w_dim], W)
    elif order == 'quadratic':
        center = np.dot(Z[:,:w_dim] ** 2, W)
    else:
        print('The order of the model is not recognized.')
        sys.exit()
    y = np.random.normal(loc=center, scale=0.5).astype('float32')
    return X, y

def Dataset_selector(name):
    if name == 'Semi_acic':
        return Semi_acic_sampler
    elif name=='Semi_ihdp':
        return Semi_ihdp_sampler
    elif name=='Sim_Hirano_Imbens':
        return Sim_Hirano_Imbens_sampler
    elif name=='Sim_Sun':
        return Sim_Sun_sampler
    elif name=='Sim_Colangelo':
        return Sim_Colangelo_sampler
    elif name=='Semi_Twins':
        return Semi_Twins_sampler
    else:
        print('Cannot find the example data sampler: %s!'%name)
        sys.exit()

class Base_sampler(object):
    def __init__(self, x, y, v, batch_size=32, normalize=False, random_seed=123):
        assert len(x)==len(y)==len(v)
        np.random.seed(random_seed)
        self.data_x = np.array(x, dtype='float32')
        self.data_y = np.array(y, dtype='float32')
        self.data_v = np.array(v, dtype='float32')
        if len(self.data_x.shape) == 1:
            self.data_x = self.data_x.reshape(-1,1)
        if len(self.data_y.shape) == 1:
            self.data_y = self.data_y.reshape(-1,1)
        self.batch_size = batch_size
        if normalize:
            self.data_v = StandardScaler().fit_transform(self.data_v)
            #self.data_v = MinMaxScaler().fit_transform(self.data_v)
        self.sample_size = len(x)
        self.full_index = np.arange(self.sample_size)
        np.random.shuffle(self.full_index)
        self.idx_gen = self.create_idx_generator(sample_size=self.sample_size)
        
    def create_idx_generator(self, sample_size, random_seed=123):
        while True:
            for step in range(math.ceil(sample_size/self.batch_size)):
                if (step+1)*self.batch_size <= sample_size:
                    yield self.full_index[step*self.batch_size:(step+1)*self.batch_size]
                else:
                    yield np.hstack([self.full_index[step*self.batch_size:],
                                    self.full_index[:((step+1)*self.batch_size-sample_size)]])
                    np.random.shuffle(self.full_index)

    def next_batch(self):
        indx = next(self.idx_gen)
        return self.data_x[indx,:], self.data_y[indx,:], self.data_v[indx, :]
    
    def load_all(self):
        return self.data_x, self.data_y, self.data_v

class Semi_acic_sampler(Base_sampler):
    """ACIC 2018 competition dataset (binary treatment) sampler (inherited from Base_sampler).

    Parameters
    ----------
    batch_size
        Int object denoting the batch size for mini-batch training. Default: ``32``.
    path
        Str object denoting the path to the original dataset.
    ufid
        Str object denoting the unique id of a specific semi-synthetic setting.
    Examples
    --------
    >>> from CausalEGM import Semi_acic_sampler
    >>> import numpy as np
    >>> x = np.random.normal(size=(2000,))
    >>> y = np.random.normal(size=(2000,))
    >>> v = np.random.normal(size=(2000,100))
    >>> ds = Semi_acic_sampler(path='../data/ACIC_2018',ufid='d5bd8e4814904c58a79d7cdcd7c2a1bb')
    """
    def __init__(self, batch_size=32, path='../data/ACIC_2018', 
                ufid='d5bd8e4814904c58a79d7cdcd7c2a1bb'):
        self.df_covariants = pd.read_csv('%s/x.csv'%path, index_col='sample_id',header=0, sep=',')
        self.df_sim = pd.read_csv('%s/scaling/factuals/%s.csv'%(path, ufid),index_col='sample_id',header=0, sep=',')
        dataset = self.df_covariants.join(self.df_sim, how='inner')
        x = dataset['z'].values.reshape(-1,1)
        y = dataset['y'].values.reshape(-1,1)
        v = dataset.values[:,:-2]
        super().__init__(x,y,v,batch_size=batch_size,normalize=True)

class Sim_Hirano_Imbens_sampler(Base_sampler):
    """Hirano Imbens simulation dataset (continuous treatment) sampler (inherited from Base_sampler).

    Parameters
    ----------
    batch_size
        Int object denoting the batch size for mini-batch training. Default: ``32``.
    N
        Sample size. Default: ``20000``.
    v_dim
        Int object denoting the dimension for covariates. Default: ``200``.
    seed
        Int object denoting the random seed. Default: ``0``.
    Examples
    --------
    >>> from CausalEGM import Sim_Hirano_Imbens_sampler
    >>> ds = Sim_Hirano_Imbens_sampler(batch_size=32, N=20000, v_dim=200, seed=0)
    """
    def __init__(self, batch_size=32, N=20000, v_dim=200, seed=0):
        np.random.seed(seed)
        v = np.random.exponential(scale=1.0, size=(N, v_dim))
        rate = v[:,0] + v[:,1]
        scale = 1/rate
        x = np.random.exponential(scale=scale)
        y = np.random.normal(x + (v[:,0] + v[:,2]) * np.exp(-x * (v[:,0] + v[:,2])) , 1)
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        super().__init__(x,y,v,batch_size=batch_size,normalize=True)

class Sim_Sun_sampler(Base_sampler):
    """Sun simulation dataset (continuous treatment) sampler (inherited from Base_sampler).

    Parameters
    ----------
    batch_size
        Int object denoting the batch size for mini-batch training. Default: ``32``.
    N
        Sample size. Default: ``20000``.
    v_dim
        Int object denoting the dimension for covariates. Default: ``200``.
    seed
        Int object denoting the random seed. Default: ``0``.
    Examples
    --------
    >>> from CausalEGM import Sim_Sun_sampler
    >>> ds = Sim_Sun_sampler(batch_size=32, N=20000, v_dim=200, seed=0)
    """
    def __init__(self, batch_size=32, N=20000, v_dim=200, seed=0):
        np.random.seed(seed)
        v = np.random.normal(0, 1, size=(N, v_dim))        
        x = np.random.normal(-2*(np.sin(2*v[:,0]))+ ((v[:,1])**2 - 1/3) + (v[:,2]-1/2) + np.cos(v[:,3]), 1)
        y = np.random.normal(((v[:,0] - 1/2)+ np.cos(v[:,1]) + (v[:,4])**2 + (v[:,5])) + x, 1)       
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        super().__init__(x,y,v,batch_size=batch_size,normalize=True)

class Sim_Colangelo_sampler(Base_sampler):
    """Colangelo simulation dataset (continuous treatment) sampler (inherited from Base_sampler).

    Parameters
    ----------
    batch_size
        Int object denoting the batch size for mini-batch training. Default: ``32``.
    N
        Sample size. Default: ``20000``.
    v_dim
        Int object denoting the dimension for covariates. Default: ``200``.
    seed
        Int object denoting the random seed. Default: ``0``.
    Examples
    --------
    >>> from CausalEGM import Sim_Colangelo_sampler
    >>> ds = Sim_Colangelo_sampler(batch_size=32, N=20000, v_dim=100, seed=0)
    """
    def __init__(self, batch_size=32, N=20000, v_dim=100, seed=0,
                rho=0.5, offset = [-1,0,1], d=1, a=3, b=0.75):
        np.random.seed(seed)
        k = np.array([rho*np.ones(v_dim-1),np.ones(v_dim),rho*np.ones(v_dim-1)],dtype=object)
        sigma = diags(k,offset).toarray()
        theta = np.array([(1/(l**2)) for l in list(range(1,(v_dim+1)))])
        epsilon = np.random.normal(0,1,N)
        nu = np.random.normal(0,1,N)
        v = np.random.multivariate_normal(np.zeros(v_dim),sigma,size=[N,])
        x = d*norm.cdf((a*v@theta)) + b*nu - 0.5
        y = 1.2*x + (x**3) + (x*v[:,0]) + 1.2*(v@theta) + epsilon
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        super().__init__(x,y,v,batch_size=batch_size,normalize=True)

class Semi_Twins_sampler(Base_sampler):
    """Twins semi synthetic  dataset sampler (inherited from Base_sampler).

    Parameters
    ----------
    batch_size
        Int object denoting the batch size for mini-batch training. Default: ``32``.
    seed
        Int object denoting the random seed. Default: ``0``.
    path
        Str obejct denoting the path to the original data.
    Examples
    --------
    >>> from CausalEGM import Semi_Twins_sampler
    >>> ds = Semi_Twins_sampler(batch_size=32, path='../data/Twins')
    """
    def __init__(self, batch_size=32, seed=0,
                path='../data/Twins'):
        covariate_df = pd.read_csv('%s/twin_pairs_X_3years_samesex.csv'%path).iloc[:,2:].drop(['infant_id_0', 'infant_id_1'], axis=1)
        treatment_df_ = pd.read_csv('%s/twin_pairs_T_3years_samesex.csv'%path).iloc[:,1:]
        outcome_df = pd.read_csv('%s/twin_pairs_Y_3years_samesex.csv'%path).iloc[:,1:]
        #### discard NAN values
        rows_with_nan = [index for index, row in covariate_df.iterrows() if row.isnull().any()]
        covariate_df = covariate_df.drop(rows_with_nan)
        treatment_df_ = treatment_df_.drop(rows_with_nan)
        outcome_df = outcome_df.drop(rows_with_nan)
        #### select those below 2kg:
        rows_less2kg = [index for index, row in treatment_df_.iterrows() if (row['dbirwt_1']>=2000)]
        covariate_df = covariate_df.drop(rows_less2kg)
        treatment_df_ = treatment_df_.drop(rows_less2kg)
        outcome_df = outcome_df.drop(rows_less2kg)

        x = np.concatenate([treatment_df_.values[:,0], treatment_df_.values[:,1]])/1000
        v =  np.concatenate([covariate_df.values, covariate_df.values])
        np.random.seed(seed)
        eps = np.random.normal(0, 0.25, size=(v.shape[0],))
        gamma = np.random.normal(0, 0.025, size=(v.shape[1],))
        y = -2 * 1/(1 + np.exp(-3 * x)) + np.dot(v, gamma) + eps
        self.auxiliary_constant =  np.mean(np.dot(v, gamma))
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        super().__init__(x,y,v,batch_size=batch_size,normalize=True)

class Gaussian_sampler(object):
    def __init__(self, mean, sd=1, N=20000):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean)))
        self.X = self.X.astype('float32')
        self.Y = None

    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean))).astype('float32')

    def load_all(self):
        return self.X, self.Y

class Assump_valid_sampler(object):
    def __init__(self, v_dim, z_dim, N=100000, n_heldout=10000, random_seed=123):
        from scipy.stats import ortho_group
        np.random.seed(random_seed)
        self.sample_size = N
        #preset diagonal matrix M
        eigenvalues = np.hstack([np.linspace(5,4,10),np.linspace(0.1,0.01,v_dim-10)])
        M = np.diag(eigenvalues)
        PCA_recon = sum(eigenvalues[z_dim:])
        print('PCA_reconstruction error: %.5f'%PCA_recon)
        #randomly generating othornomal bases and generate V
        U = ortho_group.rvs(v_dim)
        self.Sigma = np.dot(np.dot(U,M), U.T)
        self.mu = np.random.uniform(low=-1.0, high=1.0,size=(v_dim,))
        V = np.random.multivariate_normal(mean=self.mu, cov=self.Sigma,size = self.sample_size)
        V_heldout = np.random.multivariate_normal(mean=self.mu, cov=self.Sigma,size = n_heldout)
        #construct features of V
        self.A = np.dot(np.diag(eigenvalues**(-0.5)),U.T)
        T = np.dot(self.A, (V-self.mu).T).T
        T_heldout = np.dot(self.A, (V_heldout-self.mu).T).T
        self.A = self.A.astype('float32')
        self.mu = self.mu.astype('float32')
        self.data_v = V.astype('float32')
        self.data_t = T.astype('float32')
        self.data_v_heldout = V_heldout.astype('float32')
        self.data_t_heldout = T_heldout.astype('float32')

    def train(self, batch_size):
        indx = np.random.randint(low = 0, high = self.sample_size, size = batch_size)
        return self.data_v[indx, :], self.data_t[indx, :]
    def load_all(self):
        return self.data_v, self.data_t


def parse_file(path, sep='\t', header = 0, normalize=True):
    assert os.path.exists(path)
    if path[-3:] == 'npz':
        data = np.load(path)
        data_x, data_y, data_v = data['x'],data['y'],data['v']
    elif  path[-3:] == 'csv':
        data = pd.read_csv(path, header=0, sep=sep).values
        data_x = data[:,0].reshape(-1, 1).astype('float32')
        data_y = data[:,1].reshape(-1, 1).astype('float32')
        data_v = data[:,2:].astype('float32')
    elif path[-3:] == 'txt':
        data = np.loadtxt(path,delimiter=sep)
        data_x = data[:,0].reshape(-1, 1).astype('float32')
        data_y = data[:,1].reshape(-1, 1).astype('float32')
        data_v = data[:,2:].astype('float32')
    else:
        print('File format not recognized, please use .npz, .csv or .txt as input.')
        sys.exit()
    if normalize:
        data_v = StandardScaler().fit_transform(data_v)
    return data_x, data_y, data_v
