import yaml
import argparse
from BayesGM import BayesCausalGM, Sim_Hirano_Imbens_sampler, Semi_acic_sampler
import numpy as np 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    parser.add_argument('-z_dims', dest='z_dims', type=int, nargs='+', default=[3,6,3,6],
                        help='Latent dimensions')
    parser.add_argument('-lr1', dest='lr1', type=float, default=0.0002,
                        help="Learning rate for theta")
    parser.add_argument('-lr2', dest='lr2', type=float, default=0.0002,
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
    #model = BayesGM(params=params, random_seed=123)
    #data_obs, y = make_blobs(n_samples=3000, n_features=5, centers=3, cluster_std = 1, random_state=123)
    #model.train_epoch([data_obs,y], epochs=1000, epochs_per_eval=10)

    #data_obs, color = make_swiss_roll(n_samples=5000, random_state=123)
    #model=BayesClusterGM(params=params, random_seed=123)
    #data_obs = np.random.normal(3,1,size = (3000, 10)).astype('float32')
    #model.train(data_obs=[data_obs,y],n_iter=30000,batches_per_eval=1000)
    
#     model = BayesCausalGM(params=params, random_seed=123)
#     x,y,v = Sim_Hirano_Imbens_sampler(N=2000, v_dim=10).load_all()
#     model.train_epoch(data_obs=[x,y,v], epochs=1000, epochs_per_eval=10)


#     for sigma_v in [0.1,0.5,1,2,5]:
#         for sigma_y in [0.1,0.5,1,2,5]:
#             for lr_theta in [0.0001,0.001,0.005, 0.01, 0.05, 0.1]:
#                 for lr_z in [0.0001,0.001,0.005, 0.01, 0.05, 0.1]:

    #params['sigma_v'] = sigma_v
    #params['sigma_y'] = sigma_y
    z0,z1,z2,z3 = z_dims
    params['lr_theta'] = lr_theta
    params['lr_z'] = lr_z
    params['dataset'] = 'Semi_acic_%s_%d_%d_%d_%d_lr_theta=%s_lr_z=%s'%(ufid, z0,z1,z2,z3, lr_theta, lr_z)
    model = BayesCausalGM(params=params, random_seed=123)
    x,y,v = Semi_acic_sampler(ufid=ufid).load_all()
    #x = np.random.normal(0,1,size = (2000, 1)).astype('float32')
    #y = np.random.normal(0,1,size = (2000, 1)).astype('float32')
    #v = np.random.normal(0,1,size = (2000, 177)).astype('float32')
    #model.train_epoch(data_obs=[x,y,v], epochs=5, epochs_per_eval=2, epochs_per_save=2, pretrain_iter=5, batches_per_eval=2)
    model.train_epoch(data_obs=[x,y,v], epochs=200, epochs_per_eval=10, epochs_per_save=10, pretrain_iter=20000, batches_per_eval=500)
    
    
