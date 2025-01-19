import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler


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