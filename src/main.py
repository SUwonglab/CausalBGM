import yaml
import argparse
from BayesGM import *
import numpy as np 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    args = parser.parse_args()
    config = args.config
    with open(config, 'r') as f:
        params = yaml.safe_load(f)
    model = BayesGM(params=params, random_seed=123)
    #data_obs, color = make_swiss_roll(n_samples=5000, random_state=123)
    #model=BayesClusterGM(params=params, random_seed=123)
    data_obs, y = make_blobs(n_samples=3000, n_features=5, centers=3, cluster_std = 1, random_state=123)
    #data_obs = np.random.normal(3,1,size = (3000, 10)).astype('float32')
    #model.train(data_obs=[data_obs,y],n_iter=30000,batches_per_eval=1000)
    model.train_epoch([data_obs,y], epochs=1000, epochs_per_eval=10)
    #model.train(data_obs=[data_obs,y],n_iter=50000,batches_per_eval=1000)
