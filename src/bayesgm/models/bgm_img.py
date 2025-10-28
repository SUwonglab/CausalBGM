import tensorflow as tf
import tensorflow_probability as tfp
tfm = tfp.mcmc

from .base import (
    BaseFullyConnectedNet,
    Discriminator
)
from .img import MNISTEncoderConv, MNISTGenerator, MNISTDiscriminator
import numpy as np
import copy
from bayesgm.utils.helpers import Gaussian_sampler
from bayesgm.utils.data_io import save_data
from bayesgm.datasets import Base_sampler
import dateutil.tz
import datetime
import os
from tqdm import tqdm

class BGM_IMG(object):
    def __init__(self, params, timestamp=None, random_seed=None):
        super(BGM_IMG, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            tf.config.experimental.enable_op_determinism()

        self.g_net = MNISTGenerator(z_dim=params['z_dim'],filters=32,use_bnn=params['use_bnn'], 
                                        name='g_net')
        self.e_net = MNISTEncoderConv(z_dim=params['z_dim'],filters=32,name='e_net')
            
        self.dz_net = Discriminator(input_dim=params['z_dim'],model_name='dz_net',
                                        nb_units=params['dz_units'])
        self.dx_net = MNISTDiscriminator(filters=64, name='dx_net')

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
    def update_g_net(self, data_z, data_x):
        """
        Updates the generative model g_net using MNISTGenerator.
        Args:
            data_z: Tensor of shape (batch, z_dim), latent variable.
            data_x: Tensor of shape (batch, 28, 28, 1), observed MNIST data.
        Returns:
            loss_x: Scalar loss value for training g_net.
            loss_mse: Mean squared error between observed and predicted x.
        """
        with tf.GradientTape() as gen_tape:
            mu_x, sigma_square_x = self.g_net(data_z)
            x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)

            # Convert logits to probabilities for MSE calculation
            x_probs = tf.nn.sigmoid(x_logits)
            loss_mse = tf.reduce_mean((data_x - x_probs)**2)
            
            # Bernoulli log-likelihood: -log p(x|z)
            # For Bernoulli: p(x) = x * p + (1-x) * (1-p) where p = sigmoid(logits)
            # log p(x) = x * log(p) + (1-x) * log(1-p)
            # log p(x) = x * logits - log(1 + exp(logits))  (using log-sum-exp trick)
            x_logits = tf.clip_by_value(x_logits, -10, 10)  # Prevent overflow
            log_px_z = tf.reduce_sum(
                data_x * x_logits - tf.nn.softplus(x_logits), 
                axis=[1, 2, 3]  # Sum over spatial dimensions
            )
            loss_x = -tf.reduce_mean(log_px_z)  # Negative log-likelihood
            
            if self.params['use_bnn']:
                loss_kl = sum(self.g_net.losses)
                loss_x += loss_kl * self.params['kl_weight']

        # Calculate the gradients for generators and discriminators
        g_gradients = gen_tape.gradient(loss_x, self.g_net.trainable_variables)
        
        # Apply gradient clipping to prevent explosion
        #g_gradients = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in g_gradients]
        
        # Apply the gradients to the optimizer
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_net.trainable_variables))
        return loss_x, loss_mse
        
    # Update posterior of latent variables Z
    @tf.function
    def update_latent_variable_sgd(self, data_z, data_x):
        with tf.GradientTape() as tape:
            
            # logp(x|z) for Bernoulli model
            mu_x, sigma_square_x = self.g_net(data_z)
            x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)
            
            # Bernoulli log-likelihood: -log p(x|z)
            x_logits = tf.clip_by_value(x_logits, -10, 10)  # Prevent overflow
            log_px_z = tf.reduce_sum(
                data_x * x_logits - tf.nn.softplus(x_logits), 
                axis=[1, 2, 3]  # Sum over spatial dimensions
            )
            loss_px_z = -tf.reduce_mean(log_px_z)  # Negative log-likelihood

            loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2
            loss_prior_z = tf.reduce_mean(loss_prior_z)

            loss_postrior_z = loss_px_z + loss_prior_z

        # Calculate the gradients
        posterior_gradients = tape.gradient(loss_postrior_z, [data_z])
        
        # Apply gradient clipping to prevent explosion
        #posterior_gradients = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in posterior_gradients]
        
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
                mu_x_, sigma_square_x_ = self.g_net(data_z)
                x_logits_ = self.g_net.reparameterize(mu_x_, sigma_square_x_)
                data_x_ = tf.nn.sigmoid(x_logits_)
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
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=[1, 2, 3]))#(bs,) 
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
                
            d_loss = dx_loss + dz_loss + \
                    self.params['gamma']*(gpz_loss + gpx_loss)


        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        
        # Apply gradient clipping to prevent explosion
        #d_gradients = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in d_gradients]
        
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
            mu_x_, sigma_square_x_ = self.g_net(data_z)
            x_logits_ = self.g_net.reparameterize(mu_x_, sigma_square_x_)
            data_x_ = tf.nn.sigmoid(x_logits_)
            reg_loss = tf.reduce_mean(tf.square(sigma_square_x_))
            data_z_ = self.e_net(data_x)

            data_z__= self.e_net(data_x_)
            mu_x__, sigma_square_x__ = self.g_net(data_z_)
            x_logits__ = self.g_net.reparameterize(mu_x__, sigma_square_x__)
            data_x__ = tf.nn.sigmoid(x_logits__)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_x = tf.reduce_mean((data_x - data_x__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            #g_loss_adv = -tf.reduce_mean(data_dx_)
            #e_loss_adv = -tf.reduce_mean(data_dz_)
            g_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dx_)  - data_dx_)**2)
            e_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dz_)  - data_dz_)**2)

            g_e_loss = g_loss_adv + e_loss_adv + 10 * (l2_loss_x + l2_loss_z) + self.params['alpha'] * reg_loss

#             if self.params['use_bnn']:
#                 loss_g_kl = sum(self.g_net.losses)
#                 loss_e_kl = sum(self.e_net.losses)
#                 g_e_loss += self.params['kl_weight'] * (loss_g_kl+loss_e_kl)
                
        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables)
        
        # Apply gradient clipping to prevent explosion
        #g_e_gradients = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in g_e_gradients]
        
        # Apply the gradients to the optimizer
        self.g_pre_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables))

        return g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, reg_loss, g_e_loss
    

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
                mu_x__, sigma_square_x__ = self.g_net(data_z_)
                x_logits__ = self.g_net.reparameterize(mu_x__, sigma_square_x__)
                data_x__ = tf.nn.sigmoid(x_logits__)
                MSE = tf.reduce_mean((data - data_x__)**2)
                data_gen = self.generate(nb_samples=5000)
                np.savez('%s/init_data_gen_at_%d.npz'%(self.save_dir, batch_iter),
                        data_gen=data_gen, z=data_z_, x_rec=data_x__)
                print('MSE_x', MSE.numpy())
                mse_x = self.evaluate(data = data)
                print('iter [%d/%d]: MSE_x: %.4f\n' % (batch_iter, n_iter, mse_x))
                if self.params['save_model']:
                    base_path = self.checkpoint_path + f"/weights_at_egm_init_{batch_iter}"
                    self.e_net.save_weights(f"{base_path}_encoder.weights.h5")
                    self.g_net.save_weights(f"{base_path}_generator.weights.h5")
                    print('Saving checkpoint for egm_init at {}'.format(base_path))


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

                if self.params['save_model']:
                    base_path = self.checkpoint_path + f"/weights_at_{epoch}"
                    self.g_net.save_weights(f"{base_path}_generator.weights.h5")
                    print('Saving checkpoint for epoch {} at {}'.format(epoch, base_path))
                        
                data_gen = self.generate(nb_samples=5000)
                if self.params['save_res']:
                    np.savez('%s/data_gen_at_%d.npz'%(self.save_dir, epoch),
                            gen=data_gen,
                            z=self.data_z.numpy()
                            )

    @tf.function
    def evaluate(self, data, data_z=None):
        if data_z is None:
            data_z = self.e_net(data, training=False)

        mu_x, sigma_square_x = self.g_net(data_z, training=False)
        x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)
        data_x_pred = tf.nn.sigmoid(x_logits)

        mse_x = tf.reduce_mean((data-data_x_pred)**2)
        return mse_x

    @tf.function
    def generate(self, nb_samples=1000):

        data_z = tf.random.normal(shape=(nb_samples, self.params['z_dim']), mean=0.0, stddev=1.0)

        mu_x, sigma_square_x = self.g_net(data_z, training=False)
        x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)
        data_x_pred = tf.nn.sigmoid(x_logits)

        return data_x_pred

    @tf.function
    def predict_on_posteriors(self, data_posterior_z):
        n_mcmc = tf.shape(data_posterior_z)[0]
        n_samples = tf.shape(data_posterior_z)[1]

        # Flatten data
        data_posterior_z_flat = tf.reshape(data_posterior_z, [-1, self.params['z_dim']])  # Flatten: Shape: (n_IS * n_samples, z_dim)
        mu_x_flat, sigma_square_x_flat = self.g_net(data_posterior_z_flat)  # Output shape: (n_MCMC*n_samples, 28, 28, 1)
        x_logits_flat = self.g_net.reparameterize(mu_x_flat, sigma_square_x_flat)

        data_x_pred_flat = tf.nn.sigmoid(x_logits_flat)
        # Correctly reshape mean and variance
        #mu_x = tf.reshape(mu_x_flat, [n_mcmc, n_samples, self.params['x_dim']])  # Shape: (n_MCMC, n_samples, 28, 28, 1)
        data_x_pred = tf.reshape(data_x_pred_flat, [n_mcmc, n_samples, 28, 28, 1])

        return data_x_pred

    def predict(self, data, ind_x1=None, alpha=0.05, bs=100, n_mcmc=5000, burn_in=5000, step_size=0.01, num_leapfrog_steps=10, seed=42):
        """
        Predict the posterior distribution of P(x2|x1) for MNIST images.
        
        Args:
            data: Observed data (flattened) with shape (n, p) where p = 28*28 - len(ind_x1) if ind_x1 is not None
                  Contains only the conditioned pixel values
                  If ind_x1 is None, data should be full flattened image (n, 784)
            ind_x1: List of pixel indices (flattened, 0-indexed) that are conditioned in 'data'
                    If None, no conditioning is applied
            alpha: Significance level for prediction intervals
            bs: Batch size for processing
            n_mcmc: Number of MCMC samples to retain
            burn_in: Number of burn-in samples
            step_size: HMC step size
            num_leapfrog_steps: Number of leapfrog steps
            seed: Random seed
            
        Returns:
            Tuple of (data_x_pred, pred_interval):
            - data_x_pred: Flattened predicted samples
                          Shape: (n_mcmc, n, 28*28) if ind_x1 is None
                          Shape: (n_mcmc, n, 28*28 - len(ind_x1)) if ind_x1 is not None
            - pred_interval: Prediction intervals
                            Shape: (n, 28*28, 2) if ind_x1 is None
                            Shape: (n, 28*28 - len(ind_x1), 2) if ind_x1 is not None
                            Last dimension is [lower, upper]
        """
        assert 0 < alpha < 1, "The significance level 'alpha' must be greater than 0 and less than 1."

        data_posterior_z = self.tfp_mcmc_sampler(
            data=data,
            ind_x1=ind_x1,
            n_mcmc=n_mcmc,
            burn_in=burn_in,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            seed=seed
        )

        data_x_pred = []
        # Iterate over the data_posterior_z in batches
        for i in range(0, data_posterior_z.shape[1], bs):
            batch_posterior_z = data_posterior_z[:,i:i + bs,:]
            data_x_batch_pred = self.predict_on_posteriors(batch_posterior_z)
            data_x_batch_pred = data_x_batch_pred.numpy()
            data_x_pred.append(data_x_batch_pred)

        data_x_pred = np.concatenate(data_x_pred, axis=1)

        # Compute prediction intervals
        # Compute quantiles along the MCMC dimension (axis=0)
        pred_interval_upper = np.quantile(data_x_pred, 1-alpha/2, axis=0)
        pred_interval_lower = np.quantile(data_x_pred, alpha/2, axis=0)
        pred_interval = np.stack([pred_interval_lower, pred_interval_upper], axis=-1)

        # # Always flatten predictions: data_x_pred has shape (n_mcmc, n, 28, 28, 1)
        # n_mcmc_samples = data_x_pred.shape[0]
        # n_data_samples = data_x_pred.shape[1]
        # data_x_pred_flat = data_x_pred.reshape(n_mcmc_samples, n_data_samples, -1)  # (n_mcmc, n, 784)
        # pred_interval_flat = pred_interval.reshape(n_data_samples, -1, 2)  # (n, 784, 2)

        return data_x_pred, pred_interval

    @tf.function
    def get_log_posterior(self, data_z, data_x, ind_x1=None, eps=1e-6):
        """
        Calculate log posterior.
        data_z: (tf.Tensor): Input data with shape (n, q), where q is the dimension of Z.
        data_x: (tf.Tensor): Observed data (flattened) with shape (n, p) where p = 28*28 - len(ind_x1) if ind_x1 is not None
                  Contains only the conditioned pixel values
                  If ind_x1 is None, data_x should be full flattened image (n, 784)
        ind_x1: (tf.Tensor or np.ndarray): Indices of conditioned pixels (flattened, 0-indexed).
                  If None, use all pixels.
        return (tf.Tensor): Log posterior with shape (n, ).
        """

        mu_x, sigma_square_x = self.g_net(data_z)
        x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)

        # Clip logits to prevent overflow
        x_logits = tf.clip_by_value(x_logits, -10, 10)

        # Flatten both tensors for indexing
        batch_size = tf.shape(data_x)[0]
        data_x_flat = tf.reshape(data_x, [batch_size, -1])  # (n, 784)
        x_logits_flat = tf.reshape(x_logits, [batch_size, -1])  # (n, 784)

        if ind_x1 is not None:
            
            # Extract conditioned pixels from both data_x and x_logits
            data_x_cond = tf.gather(data_x_flat, ind_x1, axis=1)
            x_logits_cond = tf.gather(x_logits_flat, ind_x1, axis=1)
            
            # Compute log-likelihood only on conditioned pixels
            log_px_z = tf.reduce_sum(data_x_cond * x_logits_cond - tf.nn.softplus(x_logits_cond), axis=1)
        else:
            # Use all pixels
            log_px_z = tf.reduce_sum(data_x_flat * x_logits_flat - tf.nn.softplus(x_logits_flat), axis=1)
        
        log_prior_z = -0.5 * tf.reduce_sum(data_z**2, axis=1)
        return log_prior_z + log_px_z

    def tfp_mcmc_sampler(self, data, ind_x1=None, n_mcmc=3000, burn_in=5000, 
                        step_size=0.01, num_leapfrog_steps=10, seed=42):
        """
        Samples from the posterior distribution P(Z|X) using TensorFlow Probability MCMC.
        
        Args:
            data: (tf.Tensor): Observed data with shape (n, 28, 28, 1) for MNIST images.
                        Full image with non-conditioned pixels set to zero.
            ind_x1: (Optional[List[int]]): Indices of conditioned pixels (flattened, 0-indexed).
            n_mcmc: (int): Number of samples retained after burn-in.
            burn_in: (int): Number of samples for burn-in.
            step_size: (float): Step size for HMC kernel.
            num_leapfrog_steps: (int): Number of leapfrog steps for HMC.
            seed: (int): Random seed for reproducibility.
            
        Returns:
            tf.Tensor: Posterior samples with shape (n_mcmc, n, z_dim).
        """
        # Convert data to tensor if not already
        if not isinstance(data, tf.Tensor):
            data = tf.constant(data, dtype=tf.float32)

        # Convert ind_x1 to tensor if not None
        if ind_x1 is not None and not isinstance(ind_x1, tf.Tensor):
            ind_x1 = tf.constant(ind_x1, dtype=tf.int32)
        
        n_samples = data.shape[0]
        z_dim = self.params['z_dim']
        
        # Initialize chains with standard normal distribution
        initial_state = tf.random.normal(
            shape=(n_samples, z_dim), 
            seed=seed, 
            dtype=tf.float32
        )
        
        # Define the target log probability function
        def target_log_prob_fn(z):
            """
            Target log probability function for MCMC.
            
            Args:
                z: (tf.Tensor): Latent variables with shape (n_samples, z_dim).
                
            Returns:
                tf.Tensor: Log probability with shape (n_samples,).
            """
            return self.get_log_posterior(z, data, ind_x1)
        
        # Create HMC kernel
        hmc_kernel = tfm.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps
        )
        
        # Add adaptive step size adjustment
        adaptive_kernel = tfm.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel,
            num_adaptation_steps=int(burn_in * 0.8),
            target_accept_prob=0.75
        )
        
        # Run MCMC
        @tf.function
        def run_mcmc():
            samples, kernel_results = tfm.sample_chain(
                num_results=n_mcmc,
                num_burnin_steps=burn_in,
                current_state=initial_state,
                kernel=adaptive_kernel,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
            )
            return samples, kernel_results
        
        # Execute MCMC
        samples, is_accepted = run_mcmc()
        
        # Calculate acceptance rate
        acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
        print(f"TFP MCMC Acceptance Rate: {acceptance_rate:.4f}")
        
        return samples
