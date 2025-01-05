from .bayesGM import CausalBGM
from .util import parse_file, save_data
import argparse
from BayesGM import __version__

def main(args=None):
    # General parameters
    parser = argparse.ArgumentParser('CausalBGM',
                                     description=f'CausalBGM: An AI-powered Bayesian generative modeling approach for causal inference in observational studies - v{__version__}')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help="Output directory")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Input data file must be in csv or txt or npz format")
    parser.add_argument('--delimiter', type=str, default='\t',
                        help="Delimiter for txt or csv files (default: tab '\\t').")
    parser.add_argument('-d', '--dataset', type=str,default='Mydata',
                        help="Dataset name")
    parser.add_argument('-F', '--save_format', type=str,default='txt',
                        help="Saving format (default: txt)")
    parser.add_argument('-save_model', default=False, action=argparse.BooleanOptionalAction,
                        help="whether to save model.")
    parser.add_argument('-save_res', default=True, action=argparse.BooleanOptionalAction,
                        help="whether to save intermediate results.")
    parser.add_argument('-B', '--binary_treatment', default=True, action=argparse.BooleanOptionalAction,
                        help="whether use binary treatment setting.")
    parser.add_argument('--use_egm_init', default=True, action=argparse.BooleanOptionalAction,
                        help="whether use EGM initialization.")
    parser.add_argument('--use_bnn', default=True, action=argparse.BooleanOptionalAction,
                        help="whether use Bayesian neural nets.")

    # Parameters for iterative updating algorithm
    parser.add_argument('-Z', '--z_dims', type=int, nargs='+', default=[3,3,6,6],
                        help='Latent dimensions of Z (default: [3, 3, 6, 6]).')
    parser.add_argument('--lr_theta', type=float, default=0.0001,
                        help="Learning rate for updating model parameters (default: 0.0001).")
    parser.add_argument('--lr_z', type=float, default=0.0001,
                        help="Learning rate for updating latent variables (default: 0.0001).")
    parser.add_argument('--x_min', type=float, default=0.,
                        help="Lower bound for treatment interval (default: 0.0).")
    parser.add_argument('--x_max', type=float, default=3.,
                        help="Upper bound for treatment interval (default: 3.0).")
    parser.add_argument('--x_values', type=float, nargs='+',
                        help="List of treatment values to be predicted. Provide space-separated values. Example: --x_values 0.5 1.0 1.5")
    parser.add_argument('--g_units', type=int, nargs='+', default=[64,64,64,64,64],
                        help='Number of units for covariates generative model (default: [64,64,64,64,64]).')
    parser.add_argument('--f_units', type=int, nargs='+', default=[64,32,8],
                        help='Number of units for outcome generative model (default: [64,32,8]).')
    parser.add_argument('--h_units', type=int, nargs='+', default=[64,32,8],
                        help='Number of units for treatment generative model (default: [64,32,8]).')

    # Parameters for EGM initialization
    parser.add_argument('--kl_weight', type=float, default=0.0001,
                        help="coefficient for KL divergence term in BNNs (default: 0.0001).")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Learning rate for EGM initialization (default: 0.0002).")
    parser.add_argument('--g_d_freq', type=int, default=5,
                        help="Frequency for updating discriminators and generators (default: 5).")
    parser.add_argument('--e_units', type=int, nargs='+', default=[64,64,64,64,64],
                        help='Number of units for encoder network (default: [64,64,64,64,64]).')
    parser.add_argument('--dz_units', type=int, nargs='+', default=[64,32,8],
                        help='Number of units for discriminator network in latent space (default: [64,32,8]).')
    parser.add_argument('--use-z-rec', default=True, action=argparse.BooleanOptionalAction,
                        help="Use the reconstruction for latent features (default: True).")

    # Parameters for fitting && predicting
    parser.add_argument('-N','--n_iter', type=int, default=30000,
                        help="Number of iterations (default: 30000).")
    parser.add_argument('--startoff', type=int, default=0,
                        help="Iteration for starting evaluation (default: 0).")
    parser.add_argument('--batches_per_eval', type=int, default=500,
                        help="Number of iterations per evaluation (default: 500).")
    parser.add_argument('-E', '--epochs', type=int, default=100,
                        help="Number of epochs in iterative updating algorithm (default: 100).")
    parser.add_argument('-M', '--n_mcmc', type=int, default=3000,
                        help="MCMC sample size (default: 3000).")
    parser.add_argument('--epochs_per_eval', type=int, default=10,
                        help="Number of epochs per evaluation (default: 10).")
    parser.add_argument('--alpha', type=float, default=0.01,
                        help="Significant level (default: 0.01).")

    #Random seed control
    parser.add_argument('--seed', type=int, default=123,
                        help="Random seed for reproduction (default: 123).")
    args = parser.parse_args()
    params = vars(args)
    data = parse_file(args.input)
    params['v_dim'] = data[-1].shape[1]

    # Instantiate a CausalBGM model
    model = CausalBGM(params=params, random_seed=None)

    # Perform Encoding Generative Modeling (EGM) initialization
    # n_iter=30000: Number of iterations for the initialization process
    # batches_per_eval=500: Frequency of evaluations (e.g., every 500 batches)
    # verbose=1: Controls verbosity level, showing progress and evaluation metrics
    model.egm_init(data=data, n_iter=params['n_iter'], batches_per_eval=params['batches_per_eval'], verbose=1)

    # Train the CausalBGM model with an iterative updating algorithm
    # epochs=100: Total number of training epochs
    # epochs_per_eval=10: Frequency of evaluation during training (e.g., every 10 epochs)
    model.fit(data=data, epochs=params['epochs'], epochs_per_eval=params['epochs_per_eval'])

    # Make predictions using the trained CausalBGM model
    # alpha=0.01: Significance level for the posterior intervals
    # n_mcmc=3000: Number of MCMC posterior samples for inference
    # x_values: treatment values to be predicted for ADRF
    # q_sd=1.0: Standard deviation for the proposal distribution in Metropolis-Hastings sampling,q_sd=-1 enables adaptive S.D.
    # Returns:
    #   causal_pre: Estimated causal effects (ADRF for continuous treatment) with shape (len(x_values),)
    #   pos_intervals: Posterior intervals for the estimated causal effects with shape (len(x_values), 2)
    if params['binary_treatment']:
        causal_pre, pos_intervals = model.predict(data=data, alpha=params['alpha'], n_mcmc=params['n_mcmc'], q_sd=1.0)
    else:
        causal_pre, pos_intervals = model.predict(data=data, alpha=params['alpha'], n_mcmc=params['n_mcmc'], x_values=params['x_values'], q_sd=1.0)

    # Save results to 'save_dir'
    save_data('{}/causal_effect_point_estimate.{}'.format(model.save_dir, params['save_format']), causal_pre)
    save_data('{}/causal_effect_poterior_interval.{}'.format(model.save_dir, params['save_format']), pos_intervals)

if __name__ == "__main__":
   main()
