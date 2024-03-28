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
    data_obs, color = make_swiss_roll(n_samples=10000, random_state=123)
    #data_obs = np.random.normal(3,1,size = (3000, 10)).astype('float32')
    model.train(data_obs=data_obs,n_iter=30000,batches_per_eval=1000)
