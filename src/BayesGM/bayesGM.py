import tensorflow as tf
from .model import BaseFullyConnectedNet
import numpy as np
import copy
from .util import *
import dateutil.tz
import datetime
import os, time

class BayesGM(object):
    def __init__(self, params, timestamp=None, random_seed=None):
        super(BayesGM, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        self.g_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = params['x_dim'], 
                                        model_name='g_net', nb_units=params['g_units'])
        self.e_net = BaseFullyConnectedNet(input_dim=params['x_dim'],output_dim = params['z_dim'], 
                                        model_name='e_net', nb_units=params['e_units'])
        
        self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.initialize_nets()
        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        self.save_dir = "{}/results/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_res'] and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   

        self.ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                   e_net = self.e_net,
                                   g_e_optimizer = self.g_e_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=3)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!') 

    def get_config(self):
        """Get the parameters BayesGM model."""

        return {
                "params": self.params,
        }

    def initialize_nets(self, print_summary = False):
        """Initialize all the networks in CausalEGM."""

        self.g_net(np.zeros((1, self.params['z_dim'])))
        self.e_net(np.zeros((1, self.params['x_dim'])))
        if print_summary:
            print(self.g_net.summary())
            print(self.h_net.summary())

    @tf.function
    def update_generator(self, data_z, data_x):
        with tf.GradientTape(persistent=True) as gen_tape:
            data_x_ = self.g_net(data_z)
            loss_x = tf.reduce_mean((data_x - data_x_)**2)

        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(loss_x, self.g_net.trainable_variables+self.e_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_e_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables))
        return loss_x

    def get_log_posterior(self, data_z, x):
        x = np.expand_dims(x, axis=0)
        data_x_ = self.g_net(data_z) 
        log_likelihood = -np.sum((x-data_x_)**2,axis=1)/(2*self.params['sigma']**2)
        log_prior = -np.sum(data_z**2,axis=1)/2
        log_posterior = log_likelihood + log_prior
        return log_posterior
    
    def rejective_sampling(self, x, size = 1000):
        #z_q~N(0,I_m)
        z_q = np.random.normal(size=(size, self.params['z_dim']))
        log_q = -np.sum(z_q**2,axis=1)/2
        log_u = np.log(np.random.uniform(0, 1, size=size))
        K = np.max(self.get_log_posterior(z_q, x)-log_q)
        mask = self.get_log_posterior(z_q, x)-K-log_q > log_u
        return z_q[mask], K
    
    def MH_sampling(self, z, x, steps = 20):
        z_chain = []
        for _ in range(steps):
            z_candidate = z + np.random.normal(0,0.2,size=(self.params['z_dim'],))
            log_posterior_proposal = self.get_log_posterior(np.expand_dims(z_candidate, axis=0),x)[0]
            log_posterior_current = self.get_log_posterior(np.expand_dims(z, axis=0),x)[0]
            ratio = min(1., np.exp(log_posterior_proposal-log_posterior_current))
            if ratio > np.random.uniform(0,1):
                z = z_candidate
            z_chain.append(z_candidate)
        return z_chain[-1]

    
    def update_latent_variable(self, batch_z, batch_x, method='RS'):
        #return np.array(list(map(self.MH_sampling, batch_z, batch_x))), [1,1],[0,0]
        
        z_posterior_list = []
        nb_z_list = []
        K_list = []
        for i in range(len(batch_x)):
            z_posterior, K = self.rejective_sampling(batch_x[i])
            nb_z_list.append(len(z_posterior))
            K_list.append(K)
            z_posterior_list.append(np.mean(z_posterior,axis=0))
        return np.array(z_posterior_list),nb_z_list,K_list

    def train(self, data_obs, data_z_init=None, normalize=False,
            batch_size=32, n_iter=30000, batches_per_eval=500, batches_per_save=10000,
            startoff=0, verbose=1):
        t0 = time.time()
        self.history_z = []
        self.data_obs = data_obs
        if data_z_init is None:
            data_z_init = np.random.normal(1,1,size = (len(data_obs), self.params['z_dim'])).astype('float32')
        else:
            assert len(data_z_init) == len(data_obs), "Sample size does not match!"
        self.data_z = data_z_init

        for batch_idx in range(n_iter+1):
            #update parameters of G with SGD
            sample_idx = np.random.choice(len(data_obs), batch_size, replace=False)
            batch_z = self.data_z[sample_idx]
            batch_x = self.data_obs[sample_idx]
            loss_x = self.update_generator(batch_z, batch_x)

            #update Z by maximizing a posterior or posterior mean
            batch_z_new, nb_z_list, K_list = self.update_latent_variable(batch_z, batch_x)
            self.data_z[sample_idx] = batch_z_new
            
            if batch_idx % batches_per_eval == 0:
                print(nb_z_list)
                loss_contents = '''Iteration [%d, %.1f] : loss_x [%.4f], mean effective ss [%.4f], mean constant K [%.4f]''' \
                %(batch_idx, time.time()-t0, loss_x, np.mean(nb_z_list), np.mean(K_list))
                if verbose:
                    print(loss_contents)
                self.history_z.append(copy.copy(self.data_z))