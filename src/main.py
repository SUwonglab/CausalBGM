import yaml
import argparse
import numpy as np
import sys
from BayesGM import (
    CausalBGM,  
    Sim_Hirano_Imbens_sampler, 
    Sim_Sun_sampler, 
    Sim_Colangelo_sampler, 
    Semi_Twins_sampler, 
    Semi_acic_sampler
)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    args = parser.parse_args()
    config = args.config

    with open(config, 'r') as f:
        params = yaml.safe_load(f)

    if params['dataset'] == 'Sim_Hirano_Imbens':
        x,y,v = Sim_Hirano_Imbens_sampler(N=20000, v_dim=200).load_all()

        # Instantiate a CausalBGM model
        model = CausalBGM(params=params, random_seed=None)

        # Perform Encoding Generative Modeling (EGM) initialization
        model.egm_init(data=(x,y,v), n_iter=30000, batches_per_eval=500, verbose=1)

        # Train the CausalBGM model with an iterative updating algorithm
        model.fit(data=(x,y,v), epochs=100, epochs_per_eval=10, verbose=1)

        # Make predictions using the trained CausalBGM model
        causal_pre, pos_intervals = model.predict(data=(x,y,v), alpha=0.01, n_mcmc=3000, x_values=np.linspace(0,3,20), q_sd=1.0)

    elif params['dataset'] == 'Semi_acic':
        x,y,v = Semi_acic_sampler(ufid='629e3d2c63914e45b227cc913c09cebe').load_all()

        # Instantiate a CausalBGM model
        model = CausalBGM(params=params, random_seed=None)

        # Perform Encoding Generative Modeling (EGM) initialization
        model.egm_init(data=(x,y,v), n_iter=30000, batches_per_eval=500, verbose=1)

        # Train the CausalBGM model with an iterative updating algorithm
        model.fit(data=(x,y,v), epochs=100, epochs_per_eval=10, verbose=1)
        
        # Make predictions using the trained CausalBGM model
        causal_pre, pos_intervals = model.predict(data=(x,y,v), alpha=0.01, n_mcmc=3000, q_sd=1.0)

        