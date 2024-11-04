import yaml
import argparse
import numpy as np
import sys
from BayesGM import (
    BayesCausalGM, 
    BayesPredGM, 
    BayesPredGM_Partition, 
    Sim_Hirano_Imbens_sampler, 
    Sim_Sun_sampler, 
    Sim_Colangelo_sampler, 
    Semi_Twins_sampler, 
    Semi_acic_sampler, 
    make_swiss_roll, 
    make_blobs, 
    make_sim_data
)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    parser.add_argument('-z_dims', dest='z_dims', type=int, nargs='+', default=[3,6,3,6],
                        help='Latent dimensions')
    parser.add_argument('-lr1', dest='lr1', type=float, default=0.0001,
                        help="Learning rate for theta")
    parser.add_argument('-lr2', dest='lr2', type=float, default=0.0001,
                        help="Learning rate for z")
    parser.add_argument('-sigma_v', dest='sigma_v', type=float, default=1,
                        help="sigma for ccovariates")
    parser.add_argument('-sigma_y', dest='sigma_y', type=float, default=1.,
                        help="sigma for outcome")
    parser.add_argument('-ufid','--ufid',type=str, help='ufid of the dataset', default='629e3d2c63914e45b227cc913c09cebe')
    args = parser.parse_args()
    config = args.config
    z_dims = args.z_dims
    lr_theta = args.lr1
    lr_z = args.lr2
    sigma_v = args.sigma_v
    sigma_y = args.sigma_y
    ufid = args.ufid
    with open(config, 'r') as f:
        params = yaml.safe_load(f)
    
    params['lr_theta'] = lr_theta
    params['lr_z'] = lr_z
    
    if params['dataset'] == 'Semi_acic':
        x,y,v = Semi_acic_sampler(ufid=ufid).load_all()
        if params['pretrain']:
            params['dataset'] = 'Semi_acic_%s_pretrain_softplus_lr_theta=%s_lr_z=%s'%(ufid, lr_theta, lr_z)
        else:
            params['dataset'] = 'Semi_acic_%s_lr_theta=%s_lr_z=%s'%(ufid, lr_theta, lr_z)
        model = BayesCausalGM(params=params, random_seed=None)
        epochs = 100 if len(x)==50000 else 300
        model.fit(data_obs=[x,y,v], epochs=epochs, epochs_per_eval=10, pretrain_iter=20000, batches_per_eval=500)
    elif params['dataset'] == 'Sim_Hirano_Imbens':
        x,y,v = Sim_Hirano_Imbens_sampler(N=20000, v_dim=200).load_all()
        if params['pretrain']:
            params['dataset'] = 'Sim_Hirano_Imbens_pretrain'
        model = BayesCausalGM(params=params, random_seed=123)
        model.fit(data_obs=[x,y,v], epochs=500, epochs_per_eval=30, pretrain_iter=20000, batches_per_eval=500)
    elif params['dataset'] == 'Sim_Sun':
        x,y,v = Sim_Sun_sampler(N=20000, v_dim=200).load_all()
        if params['pretrain']:
            params['dataset'] = 'Sim_Sun_pretrain'
        model = BayesCausalGM(params=params, random_seed=123)
        model.fit(data_obs=[x,y,v], epochs=500, epochs_per_eval=30, pretrain_iter=20000, batches_per_eval=500)
    elif params['dataset'] == 'Sim_Colangelo':
        x,y,v = Sim_Colangelo_sampler(N=20000, v_dim=100).load_all()
        if params['pretrain']:
            params['dataset'] = 'Sim_Colangelo_pretrain'
        model = BayesCausalGM(params=params, random_seed=123)
        model.fit(data_obs=[x,y,v], epochs=500, epochs_per_eval=30, pretrain_iter=20000, batches_per_eval=500)
    elif params['dataset'] == 'Semi_Twins':
        x,y,v = Semi_Twins_sampler().load_all()
        if params['pretrain']:
            params['dataset'] = 'Semi_Twins_pretrain'
        model = BayesCausalGM(params=params, random_seed=123)
        model.fit(data_obs=[x,y,v], epochs=500, epochs_per_eval=30, pretrain_iter=20000, batches_per_eval=500)
    else:
        print('Error: Dataset not recognized')
        sys.exit()
        
        
    # Prediction interval settings
    # from sklearn.datasets import make_regression
    # X, y = make_regression(n_samples=2000, n_features=5, n_targets=1, noise=1, random_state=123)
    # y = np.expm1((y + abs(y.min())) / 200)
    # y = np.log1p(y)
    # y = y.reshape(-1,1)
    # X = X.astype('float32')
    # y = y.astype('float32')


    # Simulation regression data
    # X, y = make_sim_data(random_state=1)
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    # #model = BayesPredGM(params = params, random_seed = 123)
    # model = BayesPredGM_Partition(params = params, random_seed = 123)
    # model.train_epoch(data_train = [X_train,y_train], 
    #                   data_test = [X_test,y_test],
    #                   epochs=2,
    #                   epochs_per_eval=1)

