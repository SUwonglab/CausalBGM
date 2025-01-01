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
        model = CausalBGM(params=params, random_seed=None)
        # Perform Encoding Generative Modeling (EGM) initialization
        # n_iter=30000: Number of iterations for the initialization process
        # batches_per_eval=500: Frequency of evaluations (e.g., every 500 batches)
        # verbose=1: Controls verbosity level, showing progress and evaluation metrics
        model.egm_init(data=(x,y,v), n_iter=30000, batches_per_eval=500, verbose=1)
        # Train the CausalBGM model with an iterative updating algorithm
        # epochs=100: Total number of training epochs
        # epochs_per_eval=10: Frequency of evaluation during training (e.g., every 10 epochs)
        model.fit(data=(x,y,v), epochs=100, epochs_per_eval=10)
        # Make predictions using the trained CausalBGM model
        # alpha=0.01: Significance level for the posterior intervals
        # n_mcmc=3000: Number of MCMC posterior samples for inference
        # nb_intervals=20: treatment values from np.linspace(x_min,x_max,nb_intervals)
        # q_sd=1.0: Standard deviation for the proposal distribution in Metropolis-Hastings sampling,q_sd=-1 enables adaptive S.D.
        # Returns:
        #   causal_pre: Estimated causal effects (ADRF for continuous treatment) with shape (nb_intervals,)
        #   pos_intervals: Posterior intervals for the estimated causal effects with shape (nb_intervals, 2)
        causal_pre, pos_intervals = model.predict(data=(x,y,v), alpha=0.01, n_mcmc=3000, nb_intervals=20, q_sd=1.0)

    elif params['dataset'] == 'Semi_acic':
        x,y,v = Semi_acic_sampler(ufid='629e3d2c63914e45b227cc913c09cebe').load_all()
        model = CausalBGM(params=params, random_seed=None)
        model.egm_init(data=(x,y,v), n_iter=30000, batches_per_eval=500, verbose=1)
        model.fit(data=(x,y,v), epochs=100, epochs_per_eval=10)
        causal_pre, pos_intervals = model.predict(data=(x,y,v), alpha=0.01, n_mcmc=3000, q_sd=1.0)

        