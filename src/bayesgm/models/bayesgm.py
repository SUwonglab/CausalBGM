import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfm = tfp.mcmc

from .base import BaseFullyConnectedNet,Discriminator,BayesianFullyConnectedNet,BayesianVariationalNet
import numpy as np
import copy
from bayesgm.utils.helpers import Gaussian_sampler
from bayesgm.utils.data_io import save_data
from bayesgm.datasets import Base_sampler
import dateutil.tz
import datetime
import os
from tqdm import tqdm

class BayesGM(object):
    def __init__(self, params, timestamp=None, random_seed=None):
        super(BayesGM, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            tf.config.experimental.enable_op_determinism()
        if self.params['use_bnn']:
            self.g_net = BayesianVariationalNet(input_dim=params['z_dim'],output_dim = params['x_dim'], 
                                           model_name='g_net', nb_units=params['g_units'])
#             self.e_net = BayesianFullyConnectedNet(input_dim=params['x_dim'],output_dim = params['z_dim'], 
#                                             model_name='e_net', nb_units=params['e_units'])
        else:
            self.g_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = params['x_dim']+1, 
                                           model_name='g_net', nb_units=params['g_units'])

        self.e_net = BaseFullyConnectedNet(input_dim=params['x_dim'],output_dim = params['z_dim'], 
                                        model_name='e_net', nb_units=params['e_units'])
            
        self.dz_net = Discriminator(input_dim=params['z_dim'],model_name='dz_net',
                                        nb_units=params['dz_units'])
        self.dx_net = Discriminator(input_dim=params['x_dim'],model_name='dx_net',
                                        nb_units=params['dx_units'])

        #self.g_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.9, beta_2=0.99)
        self.g_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        #self.d_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.9, beta_2=0.99)
        self.d_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(params['z_dim']), sd=1.0)

        self.g_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.posterior_optimizer = tf.keras.optimizers.Adam(params['lr_z'], beta_1=0.9, beta_2=0.99)
        
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
                                    dz_net = self.dz_net,
                                    dx_net = self.dx_net,
                                    g_pre_optimizer = self.g_pre_optimizer,
                                    d_pre_optimizer = self.d_pre_optimizer,
                                    g_optimizer = self.g_optimizer,
                                    posterior_optimizer = self.posterior_optimizer)
        
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=100)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!') 

    def get_config(self):
        """Get the parameters BayesGM model."""

        return {
                "params": self.params,
        }

    def initialize_nets(self, print_summary = False):
        """Initialize all the networks in CausalBGM."""

        self.g_net(np.zeros((1, self.params['z_dim'])))
        if print_summary:
            print(self.g_net.summary())

    # Update generative model for X
    @tf.function
    def update_g_net(self, data_z, data_x, eps=1e-6):
        with tf.GradientTape() as gen_tape:
            mu_x, sigma_square_x = self.g_net(data_z)
            if 'sigma_x' in self.params:
                sigma_square_x = tf.fill(tf.shape(mu_x), self.params['sigma_x']**2)
            else:
                sigma_square_x = tf.nn.softplus(sigma_square_x) + eps
            #loss = -log(p(x|z))
            loss_mse = tf.reduce_mean((data_x - mu_x)**2)
            loss_x = tf.reduce_sum(((data_x - mu_x) ** 2) / (2 * sigma_square_x) + 0.5 * tf.math.log(sigma_square_x), axis=1)
            loss_x = tf.reduce_mean(loss_x)  # Average over batch
            
            if self.params['use_bnn']:
                loss_kl = sum(self.g_net.losses)
                loss_x += loss_kl * self.params['kl_weight']

        # Calculate the gradients for generators and discriminators
        g_gradients = gen_tape.gradient(loss_x, self.g_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_net.trainable_variables))
        return loss_x, loss_mse
        
    # Update posterior of latent variables Z
    @tf.function
    def update_latent_variable_sgd(self, data_z, data_x, eps=1e-6):
        with tf.GradientTape() as tape:
            
            # logp(x|z) for covariate model
            mu_x, sigma_square_x = self.g_net(data_z)
            if 'sigma_x' in self.params:
                sigma_square_x = tf.fill(tf.shape(mu_x), self.params['sigma_x']**2)
            else:
                sigma_square_x = tf.nn.softplus(sigma_square_x) + eps
                
            loss_px_z = tf.reduce_sum(((data_x - mu_x) ** 2) / (2 * sigma_square_x) + 0.5 * tf.math.log(sigma_square_x), axis=1)
            loss_px_z = tf.reduce_mean(loss_px_z)

            loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2
            loss_prior_z = tf.reduce_mean(loss_prior_z)

            loss_postrior_z = loss_px_z + loss_prior_z
            #loss_postrior_z = loss_postrior_z/self.params['x_dim']

        # Calculate the gradients
        posterior_gradients = tape.gradient(loss_postrior_z, [data_z])
        # Apply the gradients to the optimizer
        self.posterior_optimizer.apply_gradients(zip(posterior_gradients, [data_z]))
        return loss_postrior_z
    
#################################### EGM initialization ###########################################
    @tf.function
    def train_disc_step(self, data_z, data_x):
        """train discrinimators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: data tensor with shape [batch_size, x_dim].
        Returns:
                returns various of discrinimator loss functions.
        """  
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            with tf.GradientTape() as gpz_tape:
                data_z_ = self.e_net(data_x)
                data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
                data_dz_hat = self.dz_net(data_z_hat)
            with tf.GradientTape() as gpx_tape:
                data_x_, _ = self.g_net(data_z)
                data_x_hat = data_x*epsilon_x + data_x_*(1-epsilon_x)
                data_dx_hat = self.dx_net(data_x_hat)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            data_dx = self.dx_net(data_x)
            data_dz = self.dz_net(data_z)
            
            #dz_loss = -tf.reduce_mean(data_dz) + tf.reduce_mean(data_dz_)
            #dx_loss = -tf.reduce_mean(data_dx) + tf.reduce_mean(data_dx_)
            dz_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dz) - data_dz)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dz_) - data_dz_)**2))/2.0
            dx_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dx) - data_dx)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dx_) - data_dx_)**2))/2.0
            
            #gradient penalty for z
            grad_z = gpz_tape.gradient(data_dz_hat, data_z_hat)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            #gradient penalty for x
            grad_x = gpx_tape.gradient(data_dx_hat, data_x_hat)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))#(bs,) 
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
                
            d_loss = dx_loss + dz_loss + \
                    self.params['gamma']*(gpz_loss + gpx_loss)


        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d_pre_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dx_net.trainable_variables))
        
        return dz_loss, dx_loss, d_loss
    
    @tf.function
    def train_gen_step(self, data_z, data_x):
        """train generators step.
        Args:
            inputs: input tensor list of 4
                First item:  latent tensor with shape [batch_size, z_dim].
                Second item: date tensor with shape [batch_size, x_dim].
        Returns:
                returns various of generator loss functions.
        """  
        with tf.GradientTape(persistent=True) as gen_tape:
            data_x_, sigma_square_x_ = self.g_net(data_z)
            sigma_square_loss = tf.reduce_mean(tf.square(tf.nn.softplus(sigma_square_x_)))
            #sigma_square_loss = tf.reduce_mean(tf.square(self.g_net(data_z)[:,-1]))
            data_z_ = self.e_net(data_x)

            data_z__= self.e_net(data_x_)
            data_x__, _ = self.g_net(data_z_)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_x = tf.reduce_mean((data_x - data_x__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            #g_loss_adv = -tf.reduce_mean(data_dx_)
            #e_loss_adv = -tf.reduce_mean(data_dz_)
            g_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dx_)  - data_dx_)**2)
            e_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dz_)  - data_dz_)**2)

            g_e_loss = g_loss_adv + e_loss_adv + 10 * (l2_loss_x + l2_loss_z) + 10 * sigma_square_loss

#             if self.params['use_bnn']:
#                 loss_g_kl = sum(self.g_net.losses)
#                 loss_e_kl = sum(self.e_net.losses)
#                 g_e_loss += self.params['kl_weight'] * (loss_g_kl+loss_e_kl)
                
        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_pre_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables))

        return g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, sigma_square_loss, g_e_loss
    

    def egm_init(self, data, n_iter=10000, batch_size=32, batches_per_eval=500, verbose=1):
        
        # Set the EGM initialization indicator to be True
        self.params['use_egm_init'] = True
        self.data_sampler = Base_sampler(x=data,y=data,v=data, batch_size=batch_size, normalize=False)
        print('EGM Initialization Starts ...')
        for batch_iter in range(n_iter+1):
            # Update model parameters of Discriminator
            for _ in range(self.params['g_d_freq']):
                batch_x,_,_ = self.data_sampler.next_batch()
                batch_z = self.z_sampler.get_batch(batch_size)
                dz_loss, dx_loss, d_loss = self.train_disc_step(batch_z, batch_x)

            # Update model parameters of G,E with SGD
            batch_x,_,_ = self.data_sampler.next_batch()
            batch_z = self.z_sampler.get_batch(batch_size)
            g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, sigma_square_loss, g_e_loss = self.train_gen_step(batch_z, batch_x)
            if batch_iter % batches_per_eval == 0:
                
                loss_contents = (
                    'EGM Initialization Iter [%d] : g_loss_adv[%.4f], e_loss_adv [%.4f], l2_loss_z [%.4f], l2_loss_x [%.4f], '
                    'sd^2_loss[%.4f], g_e_loss [%.4f], dz_loss [%.4f], dx_loss[%.4f], d_loss [%.4f]'
                    % (batch_iter, g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, sigma_square_loss, g_e_loss, dz_loss, dx_loss, d_loss)
                )
                if verbose:
                    print(loss_contents)
                data_z_ = self.e_net(data)
                data_x__, _ = self.g_net(data_z_)
                MSE = tf.reduce_mean((data - data_x__)**2)
                data_gen_1, sigma_square_x_1 = self.generate(nb_samples=5000)
                data_gen_12, sigma_square_x_12 = self.generate(nb_samples=5000,use_x_sd=False)
                np.savez('%s/init_data_gen_at_%d.npz'%(self.save_dir, batch_iter),
                        gen1=data_gen_1, gen12=data_gen_12,
                        z=data_z_, x_rec=data_x__, var1=sigma_square_x_1, var12=sigma_square_x_12
                        )
                print('MSE_x', MSE.numpy())
                mse_x = self.evaluate(data = data, use_x_sd = True)
                print('iter [%d/%d]: MSE_x: %.4f\n' % (batch_iter, n_iter, mse_x))
                mse_x = self.evaluate(data = data, use_x_sd = False)
                print('iter [%d/%d]: MSE_x no x_sd: %.4f\n' % (batch_iter, n_iter, mse_x))
                if self.params['save_model']:
                    ckpt_save_path = self.ckpt_manager.save(batch_iter)
        print('EGM Initialization Ends.')
#################################### EGM initialization #############################################

    def fit(self, data,
            batch_size=32, epochs=100, epochs_per_eval=5, startoff=0,
            verbose=1, save_format='txt'):

        if self.params['save_res']:
            f_params = open('{}/params.txt'.format(self.save_dir),'w')
            f_params.write(str(self.params))
            f_params.close()
        
        if 'use_egm_init' in self.params and self.params['use_egm_init']:
            print('Initialize latent variables Z with e(V)...')
            data_z_init = self.e_net(data)
        else:
            print('Random initialization of latent variables Z...')
            data_z_init = np.random.normal(0, 1, size = (len(data), self.params['z_dim'])).astype('float32')

        self.data_z = tf.Variable(data_z_init, name="Latent Variable",trainable=True)

        best_loss = np.inf
        self.history_loss = []
        print('Iterative Updating Starts ...')
        for epoch in range(epochs+1):
            sample_idx = np.random.choice(len(data), len(data), replace=False)
            
            # Create a progress bar for batches
            with tqdm(total=len(data) // batch_size, desc=f"Epoch {epoch}/{epochs}", unit="batch") as batch_bar:
                for i in range(0,len(data) - batch_size + 1,batch_size): ## Skip the incomplete last batch
                    batch_idx = sample_idx[i:i+batch_size]
                    # Update model parameters of G, H, F with SGD
                    batch_z = tf.Variable(tf.gather(self.data_z, batch_idx, axis = 0), name='batch_z', trainable=True)
                    batch_x = data[batch_idx,:]
                    loss_x, loss_mse_x = self.update_g_net(batch_z, batch_x)

                    # Update Z by maximizing a posterior or posterior mean
                    loss_postrior_z = self.update_latent_variable_sgd(batch_z, batch_x)

                    # Update data_z with updated batch_z
                    self.data_z.scatter_nd_update(
                        indices=tf.expand_dims(batch_idx, axis=1),
                        updates=batch_z                             
                    )
                    
                    # Update the progress bar with the current loss information
                    loss_contents = (
                        'loss_x: [%.4f], loss_mse_x: [%.4f], loss_postrior_z: [%.4f]'
                        % (loss_x, loss_mse_x, loss_postrior_z)
                    )
                    batch_bar.set_postfix_str(loss_contents)
                    batch_bar.update(1)
            
            # Evaluate the full training data and print metrics for the epoch
            if epoch % epochs_per_eval == 0:
                mse_x = self.evaluate(data = data, data_z = self.data_z)
                self.history_loss.append(mse_x)

                if verbose:
                    print('Epoch [%d/%d]: MSE_x: %.4f\n' % (epoch, epochs, mse_x))

                #if epoch >= startoff and mse_x < best_loss:
                #    best_loss = mse_x
                #    self.best_epoch = epoch
                if self.params['save_model']:
                    ckpt_save_path = self.ckpt_manager.save(epoch)
                    print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
                        
                #data_z_ = self.e_net(data)# the same, no meaning
                #data_x__ = self.g_net(data_z_)[:,:self.params['x_dim']]
                data_gen_1, sigma_square_x_1 = self.generate(nb_samples=5000)
                data_gen_12, sigma_square_x_12 = self.generate(nb_samples=5000,use_x_sd=False)
                if self.params['save_res']:
                    np.savez('%s/data_gen_at_%d.npz'%(self.save_dir, epoch),
                            gen1=data_gen_1, gen12=data_gen_12,
                            z=self.data_z.numpy(), var1=sigma_square_x_1, var12=sigma_square_x_12
                            )

    @tf.function
    def evaluate(self, data, data_z=None, nb_intervals=200, use_x_sd=True, eps=1e-6):
        if data_z is None:
            data_z = self.e_net(data)

        mu_x, sigma_square_x = self.g_net(data_z)
        if 'sigma_x' in self.params:
            sigma_square_x = tf.fill(tf.shape(mu_x), self.params['sigma_x']**2)
        else:
            sigma_square_x = tf.nn.softplus(sigma_square_x) + eps


        if use_x_sd:
            data_x_pred = tf.random.normal(
                shape=tf.shape(mu_x), mean=mu_x, stddev=tf.sqrt(sigma_square_x)
            )
        else:
            data_x_pred = mu_x

        mse_x = tf.reduce_mean((data-data_x_pred)**2)
        return mse_x

    @tf.function
    def generate(self, nb_samples=1000, use_x_sd=True, eps=1e-6):

        data_z = np.random.normal(np.zeros(self.params['z_dim']), 1.0, (nb_samples, self.params['z_dim'])).astype('float32')

        mu_x, sigma_square_x = self.g_net(data_z)
        if 'sigma_x' in self.params:
            sigma_square_x = tf.fill(tf.shape(mu_x), self.params['sigma_x']**2)
        else:
            sigma_square_x = tf.nn.softplus(sigma_square_x) + eps

        if use_x_sd:
            data_x_gen = tf.random.normal(
                shape=tf.shape(mu_x), mean=mu_x, stddev=tf.sqrt(sigma_square_x)
            )
        else:
            data_x_gen = mu_x
        return data_x_gen, sigma_square_x

    @tf.function
    def predict_on_posteriors(self, data_posterior_z, eps=1e-6):
        n_mcmc = tf.shape(data_posterior_z)[0]
        n_samples = tf.shape(data_posterior_z)[1]

        # Flatten data
        data_posterior_z_flat = tf.reshape(data_posterior_z, [-1, self.params['z_dim']])  # Flatten: Shape: (n_IS * n_samples, z_dim)
        mu_x_flat, sigma_square_x_flat = self.g_net(data_posterior_z_flat)  # Output shape: (n_MCMC*n_samples, x_dim)

        if 'sigma_x' in self.params:
            # Use fixed variance if provided
            sigma_square_x_flat = tf.fill(tf.shape(mu_x_flat), self.params['sigma_x']**2)
        else:
            # Dynamically compute variance for all dimensions
            sigma_square_x_flat = tf.nn.softplus(sigma_square_x_flat) + eps  # Shape: (n_MCMC*n_samples, x_dim)

        # Correctly reshape mean and variance
        mu_x = tf.reshape(mu_x_flat, [n_mcmc, n_samples, self.params['x_dim']])  # Shape: (n_MCMC, n_samples, x_dim)
        sigma_square_x = tf.reshape(sigma_square_x_flat, [n_mcmc, n_samples, self.params['x_dim']])  # Shape: (n_MCMC, n_samples, x_dim)

        data_x_pred = tf.random.normal(
            shape=tf.shape(mu_x), mean=mu_x, stddev=tf.sqrt(sigma_square_x)
        )
        return data_x_pred

    def cond_generate(self, data_x1, ind_x1, n_mcmc=3000, q_sd=1.0, bs=100):
        assert data_x1.shape[1] == len(ind_x1), "The index length must be consistent with input"
        # P(z|x1)
        data_posterior_z = self.metropolis_hastings_sampler(data_x1, ind_x1=ind_x1, n_mcmc=n_mcmc, q_sd=q_sd)
        
        # P(x1,x2|z)
        data_x_pred = []
        # Iterate over the data_posterior_z in batches
        for i in range(0, data_posterior_z.shape[1], bs):
            batch_posterior_z = data_posterior_z[:,i:i + bs,:]
            data_x_batch_pred = self.predict_on_posteriors(batch_posterior_z)
            data_x_batch_pred = data_x_batch_pred.numpy()
            data_x_pred.append(data_x_batch_pred)
        
        data_x_pred = np.concatenate(data_x_pred, axis=1)
        return data_x_pred

    # Predict with MCMC sampling
    def predict(self, data, alpha=0.01, n_mcmc=3000, n_IS=3000, x_values=None, q_sd=1.0, sample_y=True, bs=100):
        """
        Evaluate the model on the test data and provide both point estimates and posterior intervals for causal effects.
        - For binary treatment, the Individual Treatment Effect (ITE) is estimated.
        - For continuous treatment, the Average Dose Response Function (ADRF) is estimated.

        Parameters:
        -----------
        data : list
            Input data for X.
        alpha : float
            Significance level for the posterior interval (default: 0.01).
        n_mcmc : int
            Number of posterior MCMC samples to draw (default: 3000).
        n_IS : int
            Number of importance sampling samples to draw (default: 3000).
        x_values : list of floats or np.ndarray
            Treatment values for dose-response function to be predicted (default: None).
        q_sd : float
            Standard deviation for the proposal distribution used in Metropolis-Hastings (MH) sampling (default: 1.0).
        sample_y : bool
            Whether to consider the variance function in the outcome generative model (default: True).
        bs : int
            Batch size for processing posterior samples to improve efficiency (default: 100).

        Returns:
        --------
        Binary treatment setting:
            ITE : np.ndarray
                Point estimates of the Individual Treatment Effect, with shape (n,).
        """
        assert 0 < alpha < 1, "The significance level 'alpha' must be greater than 0 and less than 1."

        # Initialize list to store densities
        density_values = []
        print('MCMC Latent Variable Sampling ...')
        #data_posterior_z = self.metropolis_hastings_sampler(data, n_mcmc=n_mcmc, q_sd=q_sd)
        data_posterior_z = self.gradient_mcmc_sampler(data, n_mcmc=n_mcmc, burn_in=500, step_size=0.01, num_leapfrog_steps=5,seed=None)[0]

        # Iterate over the data_posterior_z in batches
        for i in range(0, len(data), bs):
            batch_posterior_z = data_posterior_z[:,i:i + bs,:]
            batch_x = data[i:i + bs]
            density_values_batch = self.get_density_from_latent_posterior(batch_x, batch_posterior_z, 
                                                                         n_samples = batch_posterior_z.shape[1],
                                                                         n_mcmc = n_mcmc,
                                                                         n_IS = n_IS)
            density_values_batch = density_values_batch.numpy()
            density_values.append(density_values_batch)
        
        density_values = np.concatenate(density_values, axis=0)
        return density_values

    @tf.function
    def get_density_from_latent_posterior(self, data_x, data_posterior_z, n_samples, n_mcmc=3000, n_IS=3000, df=1, scale=0.5, eps=1e-6):
        """
        Estimate the log-density log(p(x)) for each sample in n_samples using importance sampling.
        p(x) = int p(z)p(x|z)dz=int q(z)p(x|z)w(z)dz=sum_i^{n_IS} p(x|z_i)w(z_i) where z_i ~ q(z)

        Args:
            data_x: Tensor of shape (n_samples, x_dim), observed data in x-space.
            data_posterior_z: Tensor of shape (n_mcmc, n_samples, z_dim), posterior samples of latent Z.
            n_IS: Number of importance sampling points to generate.
            df: Degrees of freedom for the Student's t-distribution.
            scale: Scale parameter for the Student's t-distribution.
            eps: Small constant for numerical stability.

        Returns:
            log_density_x: Tensor of shape (n,), estimated log-density log(p(x)) for each sample in data_x.
        """
        # Step 1: Extract dimensions
        n_samples = tf.get_static_value(n_samples)
        n_mcmc = tf.get_static_value(n_mcmc)
        
        # Step 2: Create a Student's t-distribution mixture for each sample in n
        data_posterior_z = tf.transpose(data_posterior_z, perm=[1, 0, 2])  # Shape: (n_samples, n_mcmc, z_dim)

        def create_mixture_for_sample(sample_idx):
            student_t_components = [
                tfp.distributions.MultivariateStudentTLinearOperator(
                    loc=data_posterior_z[sample_idx, i, :],
                    scale=tf.linalg.LinearOperatorDiag(tf.ones([self.params['z_dim']]) * scale),
                    df=df
                ) for i in range(n_mcmc)
            ]
            return tfp.distributions.Mixture(
                cat=tfp.distributions.Categorical(logits=tf.zeros([n_mcmc])),
                components=student_t_components
            )

        mixture_dists = [create_mixture_for_sample(i) for i in range(n_samples)] # List of mixtures, one for each sample

        # Step 3: Generate IS samples and log-importance weights
        IS_samples_z = tf.stack([mixture_dists[i].sample(n_IS) for i in range(n_samples)], axis=1)  # Shape: (n_IS, n_samples, z_dim)
        log_student_t_density = tf.stack([mixture_dists[i].log_prob(IS_samples_z[:, i, :]) for i in range(n_samples)], axis=1)  # Shape: (n_IS, n_samples)

        # Standard normal log-density in Z-space
        standard_normal = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros([self.params['z_dim']]), scale_diag=tf.ones([self.params['z_dim']]))
        log_standard_normal_density = tf.stack([standard_normal.log_prob(IS_samples_z[:, i, :]) for i in range(n_samples)], axis=1)  # Shape: (n_IS, n_samples)

        # Log-importance weights
        log_importance_weights = log_standard_normal_density - log_student_t_density  # Shape: (n_IS, n_samples)

        # Step 4: Map IS samples to X-space using self.g_net
        IS_samples_z_flat = tf.reshape(IS_samples_z, [-1, self.params['z_dim']])  # Flatten: Shape: (n_IS * n_samples, z_dim)
        g_net_output = self.g_net(IS_samples_z_flat)  # Output shape: (n_IS * n_samples, z_dim + 1)

        # Extract mean and variance
        mu_x_flat = g_net_output[:, :self.params['x_dim']]  # Mean: Shape (n_IS * n_samples, x_dim)
        if 'sigma_x' in self.params:
            # Use fixed variance if provided
            sigma_square_x_flat = tf.fill([tf.shape(mu_x_flat)[0], self.params['x_dim']], self.params['sigma_x'] ** 2)
        else:
            # Dynamically compute variance for all dimensions
            sigma_scalar_flat = tf.nn.softplus(g_net_output[:, -1:]) + eps  # Shape: (n_IS * n_samples, 1)
            sigma_square_x_flat = tf.tile(sigma_scalar_flat, [1, self.params['x_dim']])  # Shape: (n_IS * n_samples, x_dim)

        # Correctly reshape mean and variance
        mu_x = tf.reshape(mu_x_flat, [n_IS, n_samples, self.params['x_dim']])  # Shape: (n_IS, n_samples, x_dim)
        sigma_square_x = tf.reshape(sigma_square_x_flat, [n_IS, n_samples, self.params['x_dim']])  # Shape: (n_IS, n_samples, x_dim)

        # Step 5: Compute log p(x|z)
        data_x_expanded = tf.expand_dims(data_x, axis=0)  # Shape: (1, n_samples, x_dim)
        log_gaussian_density = -0.5 * tf.reduce_sum(((data_x_expanded - mu_x) ** 2) / sigma_square_x, axis=-1)
        log_gaussian_density -= 0.5 * tf.reduce_sum(tf.math.log(2 * np.pi * sigma_square_x), axis=-1)  # Shape: (n_IS, n_samples)

        # Step 6: Combine log p(x|z) and log w(z)
        log_weighted_density = log_gaussian_density + log_importance_weights  # Shape: (n_IS, n_samples)

        # Step 7: Use log-sum-exp trick to compute log p(x)
        max_log_density = tf.reduce_max(log_weighted_density, axis=0, keepdims=True)  # Shape: (1, n_samples)
        log_density_x = max_log_density + tf.math.log(
            tf.reduce_mean(tf.exp(log_weighted_density - max_log_density), axis=0)
        )  # Shape: (1, n_samples)
        
        log_density_x = tf.squeeze(log_density_x, axis=0)
        return log_density_x
        

    @tf.function
    def get_log_posterior(self, data_z, data_x, ind_x1=None, eps=1e-6):
        """
        Calculate log posterior.
        data_z: (np.ndarray): Input data with shape (n, q), where q is the dimension of Z.
        data_x: (np.ndarray): Input data with shape (n, q), where p is the dimension of X.
        ind_x1: Indices of features to extract from mu_x and sigma_square_x (optional).
        return (np.ndarray): Log posterior with shape (n, ).
        """

        mu_x, sigma_square_x = self.g_net(data_z)
        if 'sigma_x' in self.params:
            sigma_square_x = tf.fill(tf.shape(mu_x), self.params['sigma_x']**2)
        else:
            sigma_square_x = tf.nn.softplus(sigma_square_x) + eps

        if ind_x1 is not None:
            mu_x = tf.gather(mu_x, ind_x1, axis=1)
            sigma_square_x = tf.gather(sigma_square_x, ind_x1, axis=1)
    
        loss_px_z = tf.reduce_sum(((data_x - mu_x) ** 2) / (2 * sigma_square_x) + 0.5 * tf.math.log(sigma_square_x), axis=1)

        loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2

        loss_posterior_z = loss_prior_z + loss_px_z

        log_posterior = -loss_posterior_z
        return log_posterior


    def metropolis_hastings_sampler(self, data, ind_x1=None, initial_q_sd = 1.0, q_sd = None, burn_in = 5000, n_mcmc = 3000, target_acceptance_rate=0.25, tolerance=0.05, adjustment_interval=50, adaptive_sd=None, window_size=100):
        """
        Samples from the posterior distribution P(Z|X,Y,V) using the Metropolis-Hastings algorithm with adaptive proposal adjustment.

        Args:
            data (tuple): Tuple containing data_x, data_y, data_v.
            ind_x1 (list): Index for the X1 to be conditioned.
            q_sd (float or None): Fixed standard deviation for the proposal distribution. If None, `q_sd` will adapt.
            initial_q_sd (float): Initial standard deviation of the proposal distribution.
            burn_in (int): Number of samples for burn-in, set to 1000 as an initial estimate.
            n_mcmc (int): Number of samples retained after burn-in.
            target_acceptance_rate (float): Target acceptance rate for the Metropolis-Hastings algorithm.
            tolerance (float): Acceptable deviation from the target acceptance rate.
            adjustment_interval (int): Number of iterations between each adjustment of `q_sd`.
            window_size (int): The size of the sliding window for acceptance rate calculation.

        Returns:
            np.ndarray: Posterior samples with shape (n_mcmc, n, q), where q is the dimension of Z.
        """
        

        # Initialize the state of n chains
        current_state = np.random.normal(0, 1, size = (len(data), self.params['z_dim'])).astype('float32')

        # Initialize the list to store the samples
        samples = []
        counter = 0
        
        # Sliding window for acceptance tracking
        recent_acceptances = []
        
        # Determine if q_sd should be adaptive
        if adaptive_sd is None:
            adaptive_sd = (q_sd is None or q_sd <= 0)

        # Set the initial q_sd
        if adaptive_sd:
            q_sd = initial_q_sd
            
        # Run the Metropolis-Hastings algorithm
        while len(samples) < n_mcmc:
            # Propose a new state by sampling from a multivariate normal distribution
            proposed_state = current_state + np.random.normal(0, q_sd, size = (len(data), self.params['z_dim'])).astype('float32')

            # Compute the acceptance ratio
            proposed_log_posterior = self.get_log_posterior(proposed_state, data, ind_x1)
            current_log_posterior  = self.get_log_posterior(current_state, data, ind_x1)
            #acceptance_ratio = np.exp(proposed_log_posterior-current_log_posterior)
            acceptance_ratio = np.exp(np.minimum(proposed_log_posterior - current_log_posterior, 0))
            # Accept or reject the proposed state
            indices = np.random.rand(len(data)) < acceptance_ratio
            current_state[indices] = proposed_state[indices]
            
            # Update the sliding window
            recent_acceptances.append(indices)
            if len(recent_acceptances) > window_size:
                # Keep only the most recent `window_size` elements
                recent_acceptances = recent_acceptances[-window_size:]
            
            # Adjust q_sd periodically during the burn-in phase
            if adaptive_sd and counter < burn_in and counter % adjustment_interval == 0 and counter > 0:
                # Calculate the current acceptance rate
                current_acceptance_rate = np.sum(recent_acceptances) / (len(recent_acceptances)*len(data))
                
                print(f"Current MCMC Acceptance Rate: {current_acceptance_rate:.4f}")
                
                # Adjust q_sd based on the acceptance rate
                if current_acceptance_rate < target_acceptance_rate - tolerance:
                    q_sd *= 0.9  # Decrease q_sd to increase acceptance rate
                elif current_acceptance_rate > target_acceptance_rate + tolerance:
                    q_sd *= 1.1  # Increase q_sd to decrease acceptance rate
                    
                print(f"MCMC Proposal Standard Deviation (q_sd): {q_sd:.4f}")

            # Append the current state to the list of samples
            if counter >= burn_in:
                samples.append(current_state.copy())
            
            counter += 1
            
        # Calculate the acceptance rate
        acceptance_rate = np.sum(recent_acceptances) / (len(recent_acceptances)*len(data))
        print(f"Final MCMC Acceptance Rate: {acceptance_rate:.4f}")
        #print(f"Final Proposal Standard Deviation (q_sd): {q_sd:.4f}")
        return np.array(samples)

    def gradient_mcmc_sampler(self,
                             data,
                             ind_x1=None,
                             kernel='nut',
                             n_mcmc=3000,
                             burn_in=5000,
                             step_size=0.01,
                             num_leapfrog_steps=5,
                             seed=42):
        """
        Runs HMC or NUTS in parallel for each data point (n independent chains).

        Args:
            data: np.ndarray, shape (n, p). Each row is one data point.
            kernel (str): 'hmc' or 'nut' (NUTS).
            n_mcmc (int): Number of post-burn-in samples to collect.
            burn_in (int): Number of warm-up (burn-in) steps.
            step_size (float): Step size for the leapfrog integrator.
            num_leapfrog_steps (int): Number of leapfrog steps per HMC iteration.
            seed (int): Random seed for reproducibility.

        Returns:
            samples: Tensor of shape (n_mcmc, n, q).
            acceptance_rate: Scalar, average acceptance probability.
        """
        tf.random.set_seed(seed)
        n, p = data.shape
        q = self.params['z_dim']

        # 1) Initialize each chain's latent state: shape (n, q)
        init_state = tf.random.normal([n, q], mean=0.0, stddev=1.0, seed=seed)

        # 2) Define target log-prob function
        @tf.function
        def _target_log_prob_fn(z):
            # z shape: (n, q). We pass it along with data_x of shape (n, p).
            # get_log_posterior returns shape (n,).
            return self.get_log_posterior(z, data, ind_x1)

        if kernel=='hmc':
            # 3) Build the HMC kernel
            hmc_kernel = tfm.HamiltonianMonteCarlo(
                target_log_prob_fn=_target_log_prob_fn,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps
            )

            # 4) Sample from the chain
            samples, kernel_results = tfm.sample_chain(
                num_results=n_mcmc,
                num_burnin_steps=burn_in,
                current_state=init_state,
                kernel=hmc_kernel,
                trace_fn=lambda cs, kr: kr.is_accepted
            )
            # samples shape: (n_mcmc, n, q)
            # is_accepted shape: (n_mcmc, n)
        elif kernel=='nut':
            nuts_kernel = tfm.NoUTurnSampler(target_log_prob_fn=_target_log_prob_fn,
                step_size=step_size)
            # adaptive_kernel = tfm.DualAveragingStepSizeAdaptation(
            #     nuts_kernel,
            #     num_adaptation_steps=int(0.8 * burn_in)
            # )

            samples, kernel_results = tfm.sample_chain(
            num_results=n_mcmc,
            num_burnin_steps=burn_in,
            current_state=init_state,
            kernel=nuts_kernel,
            trace_fn=lambda cs, kr: kr.is_accepted
            )

        else:
            raise ValueError("Invalid kernel choice. Use 'hmc' or 'nut'.")
        # 5) Compute average acceptance probability across all chains/time
        acceptance_rate = tf.reduce_mean(tf.cast(kernel_results, tf.float32),axis=0)
        #accept_tensor = tf.stack(kernel_results, axis=0)  # shape (n_mcmc, n)
        #acceptance_rate = tf.reduce_mean(tf.cast(accept_tensor, tf.float32), axis=0)

        # Convert final samples to numpy
        return samples.numpy(), acceptance_rate.numpy()