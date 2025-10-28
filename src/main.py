import yaml
import argparse
import numpy as np
import sys
from bayesgm.models import CausalBGM, BayesGM, BayesGM_v2, BayesGM_v0, BGM_IMG
from bayesgm.utils import (
    GMM_indep_sampler, 
    Swiss_roll_sampler, 
    simulate_regression, 
    simulate_low_rank_data, 
    simulate_heteroskedastic_data, 
    simulate_z_hetero
)
from bayesgm.datasets import (
    Sim_Hirano_Imbens_sampler, 
    Sim_Sun_sampler, 
    Sim_Colangelo_sampler, 
    Semi_Twins_sampler, 
    Semi_acic_sampler
)
from sklearn.model_selection import train_test_split
import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')  # 显式禁止使用 GPU

#k l z e b u
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    parser.add_argument('-k', '--kl_weight', type=float, help='KL divergence weight')
    parser.add_argument('-l', '--learning', type=float, help='learning rate', default=0.0001)
    parser.add_argument('-a', '--alpha', type=float, help='coefficient in EGM init', default=0.0)
    parser.add_argument('-z','--Z_dim', type=int,help="Latent dimension Z")
    parser.add_argument('-x','--X_dim', type=int,help="Latent dimension X")
    parser.add_argument('-e','--epochs', type=int, default=500, help="Epoches for iterative updating")
    parser.add_argument('-b','--batches', type=int, default=100000, help="Batches for initialization")
    parser.add_argument('-r','--rank', type=int, default=2, help="rank of low-rank approximation")
    parser.add_argument('-u','--units', type=int, nargs='+', default=[64,64,64,64,64],
                        help='Number of units for covariates generative model (default: [64,64,64,64,64]).')
    args = parser.parse_args()
    config = args.config
    kl_weight = args.kl_weight
    lr = args.learning
    alpha = args.alpha
    z_dim = args.Z_dim
    x_dim = args.X_dim
    units = args.units
    E=args.epochs
    B=args.batches
    rank = args.rank

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

    elif params['dataset'] == 'Sim_indep_gmm':
        xs = GMM_indep_sampler(N=20000, sd=0.1, dim=2, n_components=3, bound=1)
        x,_ = xs.load_all()
        x = x.astype('float32')
        print('data',x.shape)
        #import pyroundtrip as pyrt
        #params = yaml.safe_load(open('configs/config_indep_gmm.yaml', 'r'))
        #model = pyrt.Roundtrip(params=params,random_seed=123)
        #model.train(data=x, save_format='npy', n_iter=40000, batches_per_eval=10000)
        #z = model.e_net(x)
        #x_gen = model.g_net(np.random.normal(0, 1.0, (5000, 2)).astype('float32'))
        #np.savez('rt.npy',z = z, x_gen=x_gen)
        params['kl_weight'] = kl_weight
        params['lr_theta']=lr
        params['lr_z']=lr
        params['dataset'] = 'Sim_indep_gmm_%s_%s'%(kl_weight, lr)
        model = BayesGM(params=params, random_seed=None)
        data_gen_0 = model.generate(nb_samples=5000)
        model.egm_init(data=x, n_iter=50000, batch_size=32, batches_per_eval=5000, verbose=1)
        model.fit(data=x, epochs=1000, epochs_per_eval=50, verbose=1)

    elif params['dataset'] == 'Sim_Swiss_roll':
        params['dataset'] = 'Sim_Swiss_roll_%s_%s'%(kl_weight, lr)
        xs = Swiss_roll_sampler(N=20000)
        x,_ = xs.load_all()
        x = x.astype('float32')
        print('data',x.shape)
        #import pyroundtrip as pyrt
        #params = yaml.safe_load(open('configs/config_indep_gmm.yaml', 'r'))
        #model = pyrt.Roundtrip(params=params,random_seed=123)
        #model.train(data=x, save_format='npy', n_iter=40000, batches_per_eval=10000)
        #z = model.e_net(x)
        #x_gen = model.g_net(np.random.normal(0, 1.0, (5000, 2)).astype('float32'))
        #np.savez('rt.npy',z = z, x_gen=x_gen)
        #sys.exit()
        params['kl_weight'] = kl_weight
        params['lr_theta']=lr
        params['lr_z']=lr
        model = BayesGM(params=params, random_seed=None)
        data_gen_0 = model.generate(nb_samples=5000)
        model.egm_init(data=x, n_iter=50000, batch_size=32, batches_per_eval=5000, verbose=1)
        
        model.fit(data=x, epochs=1000, epochs_per_eval=50, verbose=1)
    elif params['dataset'] == 'Sim_regression':
        params['dataset'] = 'Sim_regression_%s_%s_%d_%d_%d'%(kl_weight, lr, z_dim, E, B)
        params['kl_weight'] = kl_weight
        params['lr_theta'] = lr
        params['lr_z'] = lr
        params['g_units'] = units
        params['e_units'] = units
        params['z_dim'] = z_dim

        X, Y = simulate_regression(n_samples=20000, n_features=10, n_targets=1)
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=123)
        print(Y[:5])
        data = np.c_[X_train, Y_train].astype('float32')
        params['x_dim'] = data.shape[1]
        #model = BayesGM(params=params, random_seed=None)
        model = BayesGM_v2(params=params, random_seed=None)
        model.egm_init(data=data, n_iter=B, batch_size=32, batches_per_eval=5000, verbose=1)
        model.fit(data=data, epochs=E, epochs_per_eval=50, verbose=1)
        #model.ckpt.restore('checkpoints/Sim_regression_0.0001_0.0001/20250217_135642/ckpt-1000')
        #model.ckpt.restore('checkpoints/Sim_regression_0.002_0.0001_5_2000_20000/20250302_150648/ckpt-%d'%E)
        data_posterior_z, accept_rate = model.gradient_mcmc_sampler(data = X_test.astype('float32'),
                                                                    ind_x1=list(range(10)),
                                                                    kernel='hmc',
                                                                    n_mcmc = 5000,
                                                                    burn_in = 3000,
                                                                    step_size = 0.05,
                                                                    num_leapfrog_steps=25,seed=None)

        bs=100
        data_x_pred = []
        # Iterate over the data_posterior_z in batches
        for i in range(0, data_posterior_z.shape[1], bs):
            batch_posterior_z = data_posterior_z[:,i:i + bs,:]
            data_x_batch_pred = model.predict_on_posteriors(batch_posterior_z)
            data_x_batch_pred = data_x_batch_pred.numpy()
            data_x_pred.append(data_x_batch_pred)

        data_x_pred = np.concatenate(data_x_pred, axis=1)
        y_pre = np.mean(data_x_pred[:,:,-1],axis=0)
        from scipy.stats import pearsonr
        corr, _ = pearsonr(Y_test[:,0], y_pre)
        print('Mean Acce.Rate',np.mean(accept_rate))
        print('MSE',np.mean((y_pre-Y_test[:,0])**2))
        print(f"Pearson's correlation coefficient: {corr}")
        sys.exit()
        # Write results to the file
        with open("logs/res_sim_regression_0.002_0.0001_5_2000_20000.txt", "a") as f:
            f.write(f"{E},{np.mean(accept_rate)},{np.mean((y_pre-Y_test[:,0])**2)},{corr}\n")

    elif params['dataset'] == 'Sim_low_rank':
        params['dataset'] = 'Sim_low_rank_v3_%s_%s_%d_%d_%d'%(kl_weight, alpha, z_dim, E, B)
        params['kl_weight'] = kl_weight
        params['lr_theta'] = lr
        params['lr_z'] = lr
        params['alpha'] = alpha
        params['g_units'] = units
        params['e_units'] = units
        params['z_dim'] = z_dim
        params['rank'] = rank

        X, Z = simulate_low_rank_data(n_samples=20000, sigma_z=True)
        X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.1, random_state=123)
        model = BayesGM_v2(params=params, random_seed=None)
        model.egm_init(data=X_train, n_iter=B, batch_size=32, batches_per_eval=5000, verbose=1)
        model.fit(data=X_train, epochs=E, epochs_per_eval=50, verbose=1)
        #print(model.fcn_net)

    elif params['dataset'] == 'Sim_heteroskedastic':
        params['dataset'] = 'Sim_heteroskedastic_%s_%s_%d_%d_%d'%(kl_weight, alpha, z_dim, E, B)
        params['kl_weight'] = kl_weight
        params['lr_theta'] = lr
        params['lr_z'] = lr
        params['alpha'] = alpha
        params['g_units'] = units
        params['e_units'] = units
        params['z_dim'] = z_dim
        params['x_dim'] = x_dim
        params['rank'] = rank
        params['use_bnn'] = False
        X,Y = simulate_z_hetero(n=20000, k=params['z_dim'], d=params['x_dim']-1)
        np.random.seed(123)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=123)
        data = np.c_[X_train, Y_train].astype('float32')
        X_test = X_test.astype('float32')
        Y_test = Y_test.astype('float32')
        model = BayesGM(params=params, random_seed=None)
        ind_x1 = list(range(params['x_dim']-1))

        model.egm_init(data=data, n_iter=B, batch_size=32, batches_per_eval=5000, verbose=1)
        model.fit(data=data, epochs=E, epochs_per_eval=20, verbose=1)

        data_x_pred, pred_interval = model.predict(data=X_test, 
                                                    ind_x1=ind_x1, 
                                                    alpha=0.05, 
                                                    bs=100, 
                                                    n_mcmc=5000, 
                                                    burn_in=5000, 
                                                    step_size=0.01, 
                                                    num_leapfrog_steps=10, 
                                                    seed=42)
        print(data_x_pred.shape, pred_interval.shape)

        X_test_pred = data_x_pred[:,:,-1]
        X_test_pred_mean = np.mean(X_test_pred, axis=0)
        X_test_pred_median = np.median(X_test_pred, axis=0)
        print(X_test_pred.shape, X_test_pred_mean.shape, X_test_pred_median.shape)
        from scipy.stats import pearsonr, spearmanr
        corr_pred_mean, _ = pearsonr(Y_test, X_test_pred_mean)
        print(f"Pearson's correlation coefficient mean: {corr_pred_mean}")
        corr_pred_median, _ = pearsonr(Y_test, X_test_pred_median)
        print(f"Pearson's correlation coefficient median: {corr_pred_median}")
        np.savez('data_pred_heter_10_100.npz', data_x_pred=data_x_pred, pred_interval=pred_interval)
        length_gt = np.load('heter_length_gt_10_100.npy')
        length_bgm = pred_interval[:,0,1]-pred_interval[:,0,0]
        print('Average interval length:', np.mean(length_bgm), np.mean(length_gt))
        print('interval length PCC:', pearsonr(length_bgm, length_gt)[0])
        print('interval length Spearman:', spearmanr(length_bgm, length_gt)[0])
        np.savez('data_pred_heter_10_100.npz', data_x_pred=data_x_pred, pred_interval=pred_interval)
        
    elif params['dataset'] == 'MNIST':
        params['dataset'] = 'MNIST_%s_%s_%d_%d_%d'%(kl_weight, alpha, z_dim, E, B)
        params['kl_weight'] = kl_weight
        params['lr_theta'] = lr
        params['lr_z'] = lr
        params['alpha'] = alpha
        params['g_units'] = units
        params['e_units'] = units
        params['z_dim'] = z_dim
        params['use_bnn'] = False

        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        model = BGM_IMG(params=params, random_seed=None)
        if False:
            print('Initializing EGM...')
            #model.egm_init(data=x_train, n_iter=B, batch_size=32, batches_per_eval=5000, verbose=1)
            print('Fitting...')
            model.fit(data=x_train, epochs=E, epochs_per_eval=20, verbose=1)
        else:
            from bayesgm.utils import mnist_mask_indices
            #ind_x1, ind_x2 = mnist_mask_indices(mode="holes",center=(14,14),num_holes=1, hole_size=5, seed=1)
            ind_x1, ind_x2 = mnist_mask_indices(mode="edge_stripe", orientation="horizontal", stripe_pos=14, stripe_width=2)
            epoch = 1000
            checkpoint_path = 'checkpoints/MNIST_0.001_0.0_10_1000_50000/20251025_173803'
            print(f"Epoch {epoch}")
            base_path = checkpoint_path + f"/weights_at_{epoch}"
            model.g_net.load_weights(f"{base_path}_generator.weights.h5")
            n_test = 100
            data_x_pred, pred_interval = model.predict(data=x_test[:n_test].reshape((n_test, -1))[:,ind_x1], 
                                                        ind_x1=ind_x1, 
                                                        alpha=0.05, 
                                                        bs=10, 
                                                        n_mcmc=5000, 
                                                        burn_in=5000, 
                                                        step_size=0.01, 
                                                        num_leapfrog_steps=10, 
                                                        seed=42)
            print(data_x_pred.shape, pred_interval.shape)
            np.savez('data_pred_mnist_edge_stripe.npz', data_x_pred=data_x_pred, pred_interval=pred_interval)