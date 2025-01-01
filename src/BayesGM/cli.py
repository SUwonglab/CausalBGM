from .bayesGM import CausalBGM
from .util import parse_file, save_data
import argparse
from BayesGM import __version__


def main(args=None):

    parser = argparse.ArgumentParser('BayesGM',
                                     description=f'BayesGM: A general Beyesian framework for generative modeling - v{__version__}')
    parser.add_argument('-output_dir', dest='output_dir', type=str,
                        help="Output directory", required=True)
    parser.add_argument('-input', dest='input', type=str,
                        help="Input data file must be in csv or txt or npz format", required=True)
    parser.add_argument('-dataset', dest='dataset', type=str,default='Mydata',
                        help="Dataset name")
    parser.add_argument('--save-model', default=True, action=argparse.BooleanOptionalAction,
                        help="whether to save model.")
    parser.add_argument('--binary-treatment', default=True, action=argparse.BooleanOptionalAction,
                        help="whether use binary treatment setting.")

    #model hypterparameters
    parser.add_argument('-z_dims', dest='z_dims', type=int, nargs='+', default=[3,3,6,6],
                        help='Latent dimensions of the four encoder outputs e(V)_0~3.')
    parser.add_argument('-lr', dest='lr', type=float, default=0.0002,
                        help="Learning rate for the optimizer (default: 0.0002).")
    parser.add_argument('-alpha', dest='alpha', type=float, default=1.,
                        help="Coefficient for reconstruction loss (default: 1).")
    parser.add_argument('-beta', dest='beta', type=float, default=1.,
                        help="Coefficient for treatment and outcome MSE loss (default: 1).")
    parser.add_argument('-gamma', dest='gamma', type=float, default=10.,
                        help="Coefficient for gradient penalty loss (default: 10).")
    parser.add_argument('-g_d_freq', dest='g_d_freq', type=int, default=5,
                        help="Frequency for updating discriminators and generators (default: 5).")
    #network hyperparameters
    parser.add_argument('-g_units', dest='g_units', type=int, nargs='+', default=[64,64,64,64,64],
                        help='Number of units for generator/decoder network (default: [64,64,64,64,64]).')
    parser.add_argument('-e_units', dest='e_units', type=int, nargs='+', default=[64,64,64,64,64],
                        help='Number of units for encoder network (default: [64,64,64,64,64]).')
    parser.add_argument('-f_units', dest='f_units', type=int, nargs='+', default=[64,32,8],
                        help='Number of units for f network (default: [64,32,8]).')
    parser.add_argument('-h_units', dest='h_units', type=int, nargs='+', default=[64,32,8],
                        help='Number of units for h network (default: [64,32,8]).')
    parser.add_argument('-dz_units', dest='dz_units', type=int, nargs='+', default=[64,32,8],
                        help='Number of units for discriminator network in latent space (default: [64,32,8]).')
    parser.add_argument('-dv_units', dest='dv_units', type=int, nargs='+', default=[64,32,8],
                        help='Number of units for discriminator network in confounder space (default: [64,32,8]).')
    parser.add_argument('--use-z-rec', default=True, action=argparse.BooleanOptionalAction,
                        help="Use the reconstruction for latent features.")
    parser.add_argument('--use-v-gan', default=True, action=argparse.BooleanOptionalAction,
                        help="Use the GAN distribution match for covariates.")
    #training parameters
    parser.add_argument('-batch_size', dest='batch_size', type=int,
                        default=32, help='Batch size (default: 32).')
    parser.add_argument('-n_iter', dest='n_iter', type=int, default=30000,
                        help="Number of iterations (default: 30000).")
    parser.add_argument('-startoff', dest='startoff', type=int, default=0,
                        help="Iteration for starting evaluation (default: 0).")
    parser.add_argument('-batches_per_eval', dest='batches_per_eval', type=int, default=500,
                        help="Number of iterations per evaluation (default: 500).")
    parser.add_argument('-save_format', dest='save_format', type=str,default='txt',
                        help="Saving format (default: txt)")
    parser.add_argument('--save_res', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether to save results during training.")
    #Random seed control
    parser.add_argument('-seed', dest='seed', type=int, default=123,
                        help="Random seed for reproduction (default: 123).")
    args = parser.parse_args()
    params = vars(args)
    data = parse_file(args.input)
    params['v_dim'] = data[-1].shape[1]
    model = CausalBGM(params=params, random_seed=None)
    print('Start training...')
    model.egm_init(data=(x,y,v), n_iter=30000, batches_per_eval=500, verbose=1)
    model.fit(data=(x,y,v), epochs=100, epochs_per_eval=10)
    causal_pre, pos_intervals = model.predict(data=(x,y,v), alpha=0.01, n_mcmc=3000, q_sd=1.0)
    save_data('{}/causal_effect_point_estimate.txt'.format(model.save_dir), causal_pre)
    save_data('{}/causal_effect_poterior_interval.txt'.format(model.save_dir), pos_intervals)


if __name__ == "__main__":
   main()
