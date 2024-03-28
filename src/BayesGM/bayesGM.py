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
        self.g_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = params['x_dim']+1, 
                                        model_name='g_net', nb_units=params['g_units'])
        self.e_net = BaseFullyConnectedNet(input_dim=params['x_dim'],output_dim = params['z_dim'], 
                                        model_name='e_net', nb_units=params['e_units'])

        self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.posterior_optimizer = tf.keras.optimizers.legacy.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
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
                                   g_e_optimizer = self.g_e_optimizer,
                                   posterior_optimizer = self.posterior_optimizer)
        
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
        """Initialize all the networks in BayesGM."""

        self.g_net(np.zeros((1, self.params['z_dim'])))
        self.e_net(np.zeros((1, self.params['x_dim'])))
        if print_summary:
            print(self.g_net.summary())
            print(self.e_net.summary())

    #@tf.function
    def update_generator(self, data_z, data_x, eps=1e-6):
        with tf.GradientTape(persistent=True) as gen_tape:
            mu = self.g_net(data_z)[:,:self.params['x_dim']]
            if 'sigma' in self.params:
                sigma_square = self.params['sigma']**2
            else:
                sigma_square = tf.nn.relu(self.g_net(data_z)[:,-1:])+eps
            #loss = -log(p(x|z))
            loss_mse = tf.reduce_mean((data_x - mu)**2)
            loss_x = tf.reduce_mean((data_x - mu)**2/(2*sigma_square)) + \
                    tf.reduce_mean(tf.math.log(sigma_square))/2

        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(loss_x, self.g_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_e_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables))
        return loss_x, loss_mse

    def get_log_posterior(self, data_z, x, eps=1e-6):
        x = np.expand_dims(x, axis=0)
        mu = self.g_net(data_z)[:,:self.params['x_dim']]
        if 'sigma' in self.params:
            sigma_square = self.params['sigma']**2
        else:
            sigma_square = tf.nn.relu(self.g_net(data_z)[:,-1:])+eps
        log_likelihood = -np.sum((x-mu)**2/(2*sigma_square),axis=1)-np.log(sigma_square)*self.params['x_dim']/2
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
    
    #@tf.function
    def update_latent_variable_sgd(self, data_z, data_x, eps=1e-6):
        #print('1',data_z,data_z.trainable)
        #print(self.g_net.trainable_variables)
        #sys.exit()
        #print(self.g_net.trainable_variables[0])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(data_z)
            mu = self.g_net(data_z)[:,:self.params['x_dim']]
            if 'sigma' in self.params:
                sigma_square = self.params['sigma']**2
            else:
                sigma_square = tf.nn.relu(self.g_net(data_z)[:,-1:])+eps
            loss_postrior = tf.reduce_mean((data_x - mu)**2/(2*sigma_square)) + \
                    tf.reduce_mean(tf.math.log(sigma_square))/2 + \
                    tf.reduce_mean(data_z**2)/2

        #self.posterior_optimizer.build(data_z)
        # Calculate the gradients for generators and discriminators
        posterior_gradients = tape.gradient(loss_postrior, [data_z])
        #print(posterior_gradients)
        #sys.exit()
        # Apply the gradients to the optimizer
        self.posterior_optimizer.apply_gradients(zip(posterior_gradients, [data_z]))
        #print('2',data_z)
        #print(self.g_net.trainable_variables[0])
        #sys.exit()
        return loss_postrior, data_z

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
        self.history_loss = []
        self.data_obs = data_obs
        if data_z_init is None:
            #data_z_init = np.random.uniform(0, 2, size = (len(data_obs), self.params['z_dim'])).astype('float32')
            data_z_init = np.random.normal(1, 1, size = (len(data_obs), self.params['z_dim'])).astype('float32')
        else:
            assert len(data_z_init) == len(data_obs), "Sample size does not match!"
        #self.data_z = data_z_init
        self.data_z = tf.Variable(data_z_init, name="Latent Variable")
        self.data_z_init = tf.identity(self.data_z)
        for batch_idx in range(n_iter+1):
            #update model parameters of G with SGD
            sample_idx = np.random.choice(len(data_obs), batch_size, replace=False)
            #batch_z = self.data_z[sample_idx]
            batch_z = tf.Variable(tf.gather(self.data_z, sample_idx, axis = 0), name='batch_z')

            batch_x = self.data_obs[sample_idx]
            loss_x, loss_mse = self.update_generator(batch_z, batch_x)
            #update Z by maximizing a posterior or posterior mean
            #batch_z_new, nb_z_list, K_list = self.update_latent_variable(batch_z, batch_x)
            #print('before',batch_z)
            loss_postrior, batch_z_new = self.update_latent_variable_sgd(batch_z, batch_x)
            #print('after',batch_z,batch_z_new)
            #print(batch_idx,batch_z_new)
            nb_z_list, K_list = 0, 0
            # if batch_idx == 3:
            #     sys.exit()
            #variable update rows
            #self.data_z[sample_idx] = batch_z_new
            self.data_z = tf.compat.v1.scatter_update(self.data_z, sample_idx, batch_z_new)
            #print(self.data_z.device, batch_z.device)
            if batch_idx % batches_per_eval == 0:
                self.history_loss.append([loss_x, loss_postrior])
                loss_contents = '''Iteration [%d, %.1f] : loss_x [%.4f],loss_mse [%.4f] loss_postrior [%.4f], mean effective ss [%.4f], mean constant K [%.4f]''' \
                %(batch_idx, time.time()-t0, loss_x, loss_mse, loss_postrior, np.mean(nb_z_list), np.mean(K_list))
                if verbose:
                    print(loss_contents)
                self.history_z.append(copy.copy(self.data_z))
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                print('mean',tf.reduce_mean(self.data_z))
                plt.violinplot(self.data_z.numpy())
                plt.savefig('%s/violinplot_%d.png'%(self.save_dir, batch_idx))
                plt.close()
        np.save('%s/history_loss.npy'%(self.save_dir),np.array(self.history_loss))
        plt.plot(np.array(self.history_loss)[:,0])
        plt.xlabel('Per 1000 iteration')
        plt.ylabel('MLE loss')
        plt.savefig('%s/MLE_loss.png'%(self.save_dir))
        plt.close()

        plt.plot(np.array(self.history_loss)[:,1])
        plt.xlabel('Per 1000 iteration')
        plt.ylabel('Posterior loss')
        plt.savefig('%s/Posterior_loss.png'%(self.save_dir))
        plt.close()
        