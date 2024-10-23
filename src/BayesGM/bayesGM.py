import tensorflow as tf
from .model import BaseFullyConnectedNet,Discriminator
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
        self.g_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = params['x_dim'],#+1 
                                        model_name='g_net', nb_units=params['g_units'])
        self.e_net = BaseFullyConnectedNet(input_dim=params['x_dim'],output_dim = params['z_dim'], 
                                        model_name='e_net', nb_units=params['e_units'])

        self.g_e_optimizer = tf.keras.optimizers.Adam(0.005, beta_1=0.9, beta_2=0.99)
        #self.g_e_optimizer = tf.keras.optimizers.SGD(params['lr_theta'])
        self.posterior_optimizer = tf.keras.optimizers.legacy.Adam(0.005, beta_1=0.9, beta_2=0.99)
        #self.posterior_optimizer = tf.keras.optimizers.legacy.SGD(params['lr_z'])
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
        #data_z: (bs, z_dim), x: (x_dim,)
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
                    self.params['z_dim'] * tf.reduce_mean(data_z**2)/(2*self.params['x_dim'])

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
        data_obs,y = data_obs
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
            for _ in range(10):
                loss_postrior, batch_z = self.update_latent_variable_sgd(batch_z, batch_x)
            #print('after',batch_z)
            #print(batch_idx,batch_z_new)
            nb_z_list, K_list = 0, 0
            # if batch_idx == 3:
            #     sys.exit()
            #variable update rows
            #self.data_z[sample_idx] = batch_z_new
            self.data_z = tf.compat.v1.scatter_update(self.data_z, sample_idx, batch_z)
            #print(self.data_z.device, batch_z.device)
            if batch_idx % batches_per_eval == 0:
                self.history_loss.append([loss_x, loss_postrior])
                loss_contents = '''Iteration [%d, %.1f] : loss_x [%.4f],loss_mse [%.4f] loss_postrior [%.4f], mean effective ss [%.4f], mean constant K [%.4f]''' \
                %(batch_idx, time.time()-t0, loss_x, loss_mse, loss_postrior, np.mean(nb_z_list), np.mean(K_list))
                if verbose:
                    print(loss_contents)
                self.history_z.append(copy.copy(self.data_z))
                np.savez('%s/data_at_%d.npz'%(self.save_dir, batch_idx),data_x_rec=self.g_net(self.data_z).numpy(), data_z=self.data_z.numpy())
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                print('mean',tf.reduce_mean(self.data_z))
                plt.violinplot(self.data_z.numpy())
                plt.savefig('%s/violinplot_%d.png'%(self.save_dir, batch_idx))
                plt.close()
                plt.scatter(self.data_z.numpy()[:,0],self.data_z.numpy()[:,1],c=y)
                plt.savefig('%s/scatter_%d.png'%(self.save_dir, batch_idx))
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

    def train_epoch(self, data_obs, data_z_init=None, normalize=False,
            batch_size=32, epochs=1000, epochs_per_eval=10,
            startoff=0, verbose=1):
        from sklearn import metrics
        data_obs,y = data_obs
        t0 = time.time()
        self.history_z = []
        self.history_loss = []
        self.data_obs = data_obs
        if data_z_init is None:
            data_z_init = np.random.normal(0, 1, size = (len(data_obs), self.params['z_dim'])).astype('float32')
        else:
            assert len(data_z_init) == len(data_obs), "Sample size does not match!"
        #self.data_z = data_z_init
        self.data_z = tf.Variable(data_z_init, name="Latent Variable")
        self.data_z_init = tf.identity(self.data_z)
        for epoch in range(epochs+1):
            sample_idx = np.random.choice(len(data_obs), len(data_obs), replace=False)
            for i in range(0,len(data_obs),batch_size):
                batch_idx = sample_idx[i:i+batch_size]
                #update model parameters of G with SGD
                #batch_z = self.data_z[batch_idx]
                batch_z = tf.Variable(tf.gather(self.data_z, batch_idx, axis = 0), name='batch_z')
                batch_x = self.data_obs[batch_idx]
                loss_x, loss_mse = self.update_generator(batch_z, batch_x)

                #update Z by maximizing a posterior or posterior mean
                #batch_z_new, nb_z_list, K_list = self.update_latent_variable(batch_z, batch_x)
                #print('before',batch_z)
                for _ in range(1):
                    loss_postrior, batch_z= self.update_latent_variable_sgd(batch_z, batch_x)
                #loss_postrior = 0,0
                #print('after',batch_z)
                #print(i, batch_z)
                nb_z_list, K_list = 0, 0
                #if i / batch_size  == 3:
                #    sys.exit()
                #variable update rows
                #self.data_z[batch_idx] = batch_z
                self.data_z = tf.compat.v1.scatter_update(self.data_z, batch_idx, batch_z)
                #print(self.data_z.device, batch_z.device)
            if epoch % epochs_per_eval == 0:
                self.history_loss.append([loss_x, loss_postrior])
                loss_contents = '''Epoch [%d, %.1f] : loss_x [%.4f],loss_mse [%.4f] loss_postrior [%.4f], mean effective ss [%.4f], mean constant K [%.4f]''' \
                %(epoch, time.time()-t0, loss_x, loss_mse, loss_postrior, np.mean(nb_z_list), np.mean(K_list))
                if verbose:
                    print(loss_contents)
                self.history_z.append(copy.copy(self.data_z))
                np.savez('%s/data_at_%d.npz'%(self.save_dir, epoch),data_x_rec=self.g_net(self.data_z).numpy(), data_z=self.data_z.numpy())
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                print('mean',tf.reduce_mean(self.data_z))
                plt.violinplot(self.data_z.numpy())
                plt.title('Epoch %d'%(epoch))
                plt.savefig('%s/violinplot_%d.png'%(self.save_dir, epoch))
                plt.close()
                plt.scatter(self.data_z.numpy()[:,0],self.data_z.numpy()[:,1],c=y)
                plt.title('Epoch %d'%(epoch))
                plt.savefig('%s/scatter_%d.png'%(self.save_dir, epoch))
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

class BayesClusterGM(object):
    def __init__(self, params, timestamp=None, random_seed=None):
        super(BayesClusterGM, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        self.g_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = params['x_dim']+1, 
                                        model_name='g_net', nb_units=params['g_units'])
        self.e_net = BaseFullyConnectedNet(input_dim=params['x_dim'],output_dim = params['z_dim'], 
                                        model_name='e_net', nb_units=params['e_units'])

        self.g_e_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.5, beta_2=0.9)
        self.posterior_optimizer = tf.keras.optimizers.legacy.Adam(params['lr_z'], beta_1=0.5, beta_2=0.9)
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
            loss_z_cont = tf.reduce_mean(data_z[:,:-self.params['K']]**2)/2
            class_indx = tf.math.argmax(data_z[:,-self.params['K']:],axis=1)
            #print('1',class_indx)
            #print('2',tf.gather(data_z[:,-self.params['K']:],class_indx,axis=1))
            w_max = tf.linalg.diag_part(tf.gather(data_z[:,-self.params['K']:],class_indx,axis=1))
            loss_z_disc = -tf.reduce_mean(tf.math.log(w_max+eps))/(self.params['z_dim']-self.params['K'])
            #print('3',w_max,tf.experimental.numpy.max(w_max),tf.experimental.numpy.min(w_max)) #negative value
            #print('4',loss_z_disc)
            #sys.exit()
            loss_postrior = tf.reduce_mean((data_x - mu)**2/(2*sigma_square)) + \
                    tf.reduce_mean(tf.math.log(sigma_square))/2 + \
                    loss_z_cont + loss_z_disc

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
        #normalize last k digits
        normalized = tf.nn.softmax(data_z[:,-self.params['K']:])
        data_z = tf.concat([data_z[:,:-self.params['K']],normalized],axis=1)
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
        from sklearn import metrics
        data_obs,y = data_obs
        t0 = time.time()
        self.history_z = []
        self.history_loss = []
        self.data_obs = data_obs
        if data_z_init is None:
            #data_z_init = np.random.uniform(0, 2, size = (len(data_obs), self.params['z_dim'])).astype('float32')
            data_z_cont_init = np.random.normal(0, 1, size = (len(data_obs), self.params['z_dim']-self.params['K']))
            data_z_disc_init = np.ones((len(data_obs), self.params['K']))/self.params['K']
            data_z_init = np.concatenate([data_z_cont_init, data_z_disc_init],axis=1).astype('float32')
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
            #print('after',batch_z_new)
            #print(batch_idx,batch_z_new)
            nb_z_list, K_list = 0, 0
            #if batch_idx == 3:
            #    sys.exit()
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
                np.savez('%s/data_at_%d.npz'%(self.save_dir, batch_idx),data_x_rec=self.g_net(self.data_z).numpy(), data_z=self.data_z.numpy())
                y_pre = np.argmax(self.data_z.numpy()[:,-self.params['K']:],axis=1)
                print('ARI',metrics.adjusted_rand_score(y, y_pre))
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                print('mean',tf.reduce_mean(self.data_z))
                print('Last three digits', batch_z_new[0,-3:].numpy())
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



class BayesCausalGM(object):
    def __init__(self, params, timestamp=None, random_seed=None):
        super(BayesCausalGM, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        self.g_net = BaseFullyConnectedNet(input_dim=sum(params['z_dims']),output_dim = params['v_dim']+1, 
                                        model_name='g_net', nb_units=params['g_units'])
        self.e_net = BaseFullyConnectedNet(input_dim=params['v_dim'],output_dim = sum(params['z_dims']), 
                                        model_name='e_net', nb_units=params['e_units'])
        self.f_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][1]+1,
                                        output_dim = 2, model_name='f_net', nb_units=params['f_units'])
        self.h_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][2],
                                        output_dim = 2, model_name='h_net', nb_units=params['h_units'])
        self.dz_net = Discriminator(input_dim=sum(params['z_dims']),model_name='dz_net',
                                        nb_units=params['dz_units'])
        
        self.g_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.9, beta_2=0.99)
        self.d_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.9, beta_2=0.99)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(sum(params['z_dims'])), sd=1.0)

        self.g_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.f_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.h_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        #self.g_optimizer = tf.keras.optimizers.SGD(params['lr_theta'])
        self.posterior_optimizer = tf.keras.optimizers.legacy.Adam(params['lr_z'], beta_1=0.9, beta_2=0.99)
        #self.posterior_optimizer = tf.keras.optimizers.legacy.SGD(params['lr_z'])
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
                                    f_net = self.f_net,
                                    h_net = self.h_net,
                                    dz_net = self.dz_net,
                                    g_pre_optimizer = self.g_pre_optimizer,
                                    d_pre_optimizer = self.d_pre_optimizer,
                                    g_optimizer = self.g_optimizer,
                                    f_optimizer = self.f_optimizer,
                                    h_optimizer = self.h_optimizer,
                                    posterior_optimizer = self.posterior_optimizer)
        
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)                 

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

        self.g_net(np.zeros((1, sum(self.params['z_dims']))))
        self.f_net(np.zeros((1, self.params['z_dims'][0]+self.params['z_dims'][1]+1)))
        self.h_net(np.zeros((1, self.params['z_dims'][0]+self.params['z_dims'][2])))
        if print_summary:
            print(self.g_net.summary())
            print(self.f_net.summary())    
            print(self.h_net.summary()) 

    #update network for covariates V
    @tf.function
    def update_g_net(self, data_z, data_v, eps=1e-6):
        with tf.GradientTape(persistent=True) as gen_tape:
            mu_v = self.g_net(data_z)[:,:self.params['v_dim']]
            if 'sigma_v' in self.params:
                sigma_square_v = self.params['sigma_v']**2
            else:
                #sigma_square_v = tf.nn.relu(self.g_net(data_z)[:,-1])+eps
                sigma_square_v = tf.nn.softplus(self.g_net(data_z)[:,-1])
            #loss = -log(p(x|z))
            loss_mse = tf.reduce_mean((data_v - mu_v)**2)
            loss_v = tf.reduce_sum((data_v - mu_v)**2, axis=1)/(2*sigma_square_v) + \
                    self.params['v_dim'] * tf.math.log(sigma_square_v)/2
            loss_v = tf.reduce_mean(loss_v)

        # Calculate the gradients for generators and discriminators
        g_gradients = gen_tape.gradient(loss_v, self.g_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_net.trainable_variables))
        return loss_v, loss_mse
    
    #update network for treatment X
    @tf.function
    def update_h_net(self, data_z, data_x, eps=1e-6):
        with tf.GradientTape(persistent=True) as gen_tape:
            data_z0 = data_z[:,:self.params['z_dims'][0]]
            data_z2 = data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
            mu_x = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
            if self.params['binary_treatment']:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, 
                                                       logits=mu_x))
                #loss = tf.reduce_mean((tf.sigmoid(mu_x) - data_x)**2)
                loss_x =  loss
            else:
                if 'sigma_x' in self.params:
                    sigma_square_x = self.params['sigma_x']**2
                else:
                    #sigma_square_x = tf.nn.relu(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1])+eps
                    sigma_square_x = tf.nn.softplus(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1])
                #loss = -log(p(x|z))
                loss = tf.reduce_mean((data_x - mu_x)**2)
                loss_x = tf.reduce_sum((data_x - mu_x)**2, axis=1)/(2*sigma_square_x) + \
                        tf.math.log(sigma_square_x)/2
                loss_x = tf.reduce_mean(loss_x)
        
        # Calculate the gradients for generators and discriminators
        h_gradients = gen_tape.gradient(loss_x, self.h_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.h_optimizer.apply_gradients(zip(h_gradients, self.h_net.trainable_variables))
        return loss_x, loss
    
    #update network for outcome Y
    @tf.function
    def update_f_net(self, data_z, data_x, data_y, eps=1e-6):
        with tf.GradientTape(persistent=True) as gen_tape:
            data_z0 = data_z[:,:self.params['z_dims'][0]]
            data_z1 = data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
            mu_y = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                #sigma_square_y = tf.nn.relu(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1])+eps
                sigma_square_y = tf.nn.softplus(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1])
            #loss = -log(p(y|z,x))
            loss_mse = tf.reduce_mean((data_y - mu_y)**2)
            loss_y = tf.reduce_sum((data_y - mu_y)**2, axis=1)/(2*sigma_square_y) + \
                    tf.math.log(sigma_square_y)/2
            loss_y = tf.reduce_mean(loss_y)

        # Calculate the gradients for generators and discriminators
        f_gradients = gen_tape.gradient(loss_y, self.f_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.f_optimizer.apply_gradients(zip(f_gradients, self.f_net.trainable_variables))
        return loss_y, loss_mse
    
    # update posterior of latent variables Z
    #@tf.function
    def update_latent_variable_sgd(self, data_z, data_x, data_y, data_v, eps=1e-6):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(data_z)
            data_z0 = data_z[:,:self.params['z_dims'][0]]
            data_z1 = data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
            data_z2 = data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
            data_z3 = data_z[:-self.params['z_dims'][3]:]
            
            # logp(v|z) for covariate model
            mu_v = self.g_net(data_z)[:,:self.params['v_dim']]
            if 'sigma_v' in self.params:
                sigma_square_v = self.params['sigma_v']**2
            else:
                #sigma_square_v = tf.nn.relu(self.g_net(data_z)[:,-1])+eps
                sigma_square_v = tf.nn.softplus(self.g_net(data_z)[:,-1])
                
            loss_pv_z = tf.reduce_sum((data_v - mu_v)**2, axis=1)/(2*sigma_square_v) + \
                    self.params['v_dim'] * tf.math.log(sigma_square_v)/2
            loss_pv_z = tf.reduce_mean(loss_pv_z)
            
            # log(x|z) for treatment model
            mu_x = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
            if 'sigma_x' in self.params:
                sigma_square_x = self.params['sigma_x']**2
            else:
                #sigma_square_x = tf.nn.relu(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1])+eps
                sigma_square_x = tf.nn.softplus(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1])

            if self.params['binary_treatment']:
                loss_px_z = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, 
                                                       logits=mu_x))
                #loss_px_z = tf.reduce_mean((tf.sigmoid(mu_x) - data_x)**2)
            else:
                loss_px_z = tf.reduce_sum((data_x - mu_x)**2, axis=1)/(2*sigma_square_x) + \
                        tf.math.log(sigma_square_x)/2
                loss_px_z = tf.reduce_mean(loss_px_z)
                
            # log(y|z,x) for outcome model
            mu_y = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                #sigma_square_y = tf.nn.relu(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1])+eps
                sigma_square_y = tf.nn.softplus(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1])

            loss_py_zx = tf.reduce_sum((data_y - mu_y)**2, axis=1)/(2*sigma_square_y) + \
                    tf.math.log(sigma_square_y)/2
            loss_py_zx = tf.reduce_mean(loss_py_zx)

            loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2
            loss_prior_z = tf.reduce_mean(loss_prior_z)

            loss_postrior_z = loss_pv_z + loss_px_z + loss_py_zx + loss_prior_z
            #loss_postrior_z = loss_postrior_z/self.params['v_dim']

        # self.posterior_optimizer.build(data_z)
        # calculate the gradients
        posterior_gradients = tape.gradient(loss_postrior_z, [data_z])
        # apply the gradients to the optimizer
        self.posterior_optimizer.apply_gradients(zip(posterior_gradients, [data_z]))
        return loss_postrior_z, data_z
    
#################################### Pretrain ###########################################
    @tf.function
    def train_disc_step(self, data_z, data_v):
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            with tf.GradientTape() as gp_tape:
                data_z_ = self.e_net(data_v)
                data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
                data_dz_hat = self.dz_net(data_z_hat)

            data_dz_ = self.dz_net(data_z_)
            data_dz = self.dz_net(data_z)
            dz_loss = -tf.reduce_mean(data_dz) + tf.reduce_mean(data_dz_)

            #gradient penalty 
            grad_z = gp_tape.gradient(data_dz_hat, data_z_hat) #(bs,z_dim)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            d_loss = dz_loss + self.params['gamma']*gpz_loss

        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d_pre_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables))
        return dz_loss, d_loss
    
    @tf.function
    def train_gen_step(self, data_z, data_v, data_x, data_y):
        with tf.GradientTape(persistent=True) as gen_tape:
            sigma_square_loss = 0
            data_v_ = self.g_net(data_z)[:,:self.params['v_dim']]
            sigma_square_loss += tf.reduce_mean(tf.square(self.g_net(data_z)[:,-1]))
            data_z_ = self.e_net(data_v)
            
            data_z0 = data_z_[:,:self.params['z_dims'][0]]
            data_z1 = data_z_[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
            data_z2 = data_z_[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]

            data_z__= self.e_net(data_v_)
            data_v__ = self.g_net(data_z_)[:,:self.params['v_dim']]
            
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_v = tf.reduce_mean((data_v - data_v__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            e_loss_adv = -tf.reduce_mean(data_dz_)

            data_y_ = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
            sigma_square_loss += tf.reduce_mean(
                tf.square(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1]))
            data_x_ = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
            sigma_square_loss += tf.reduce_mean(
                tf.square(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1]))

            if self.params['binary_treatment']:
                data_x_ = tf.sigmoid(data_x_)
            l2_loss_x = tf.reduce_mean((data_x_ - data_x)**2)
            l2_loss_y = tf.reduce_mean((data_y_ - data_y)**2)
            g_e_loss = e_loss_adv+self.params['alpha']*(l2_loss_v + self.params['use_z_rec']*l2_loss_z) \
                        + self.params['beta']*(l2_loss_x+l2_loss_y) + 0.001 * sigma_square_loss

        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables+\
                                        self.f_net.trainable_variables+self.h_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_pre_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables+\
                                            self.f_net.trainable_variables+self.h_net.trainable_variables))
        return e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss
    

    def pretrain(self, n_iter=10000, batch_size=32, batches_per_eval=500, verbose=1):
        for batch_iter in range(n_iter+1):
            # update model parameters of Discriminator
            for _ in range(self.params['g_d_freq']):
                batch_idx = np.random.choice(len(self.data_obs), batch_size, replace=False)
                batch_obs = self.data_obs[batch_idx]
                batch_z = self.z_sampler.get_batch(batch_size)
                batch_v = batch_obs[:,2:]
                dz_loss, d_loss = self.train_disc_step(batch_z, batch_v)

            # update model parameters of G, H, F with SGD
            batch_z = self.z_sampler.get_batch(batch_size)
            batch_idx = np.random.choice(len(self.data_obs), batch_size, replace=False)
            batch_obs = self.data_obs[batch_idx]
            batch_x = batch_obs[:,:1]
            batch_y = batch_obs[:,1:2]
            batch_v = batch_obs[:,2:]
            e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss = self.train_gen_step(batch_z, batch_v, batch_x, batch_y)
            if batch_iter % batches_per_eval == 0:
                loss_contents = '''Pretrain Iter [%d] : e_loss_adv [%.4f],\
                l2_loss_v [%.4f], l2_loss_z [%.4f], l2_loss_x [%.4f],\
                l2_loss_y [%.4f], g_e_loss [%.4f], dz_loss [%.4f], d_loss [%.4f]''' \
                %(batch_iter, e_loss_adv , l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss,
                dz_loss, d_loss)
                if verbose:
                    print(loss_contents)
                    causal_pre, mse_x, mse_y, mse_v = self.evaluate(stage='pretrain')
                    if self.params['save_res']:
                        self.save('{}/causal_pre_at_pretrain_iter-{}.txt'.format(self.save_dir, batch_iter), causal_pre)
#################################### Pretrain #############################################

    def fit(self, data_obs, data_z_init=None,
            batch_size=32, epochs=100, epochs_per_eval=5, startoff=0,
            verbose=1, save_format='txt',pretrain_iter=10000, batches_per_eval=500):
        
        if self.params['save_res']:
            f_params = open('{}/params.txt'.format(self.save_dir),'w')
            f_params.write(str(self.params))
            f_params.close()
            
        self.data_x, self.data_y, self.data_v = data_obs
        if len(self.data_x.shape) == 1:
            self.data_x = self.data_x.reshape(-1,1)
        if len(self.data_y.shape) == 1:
            self.data_y = self.data_y.reshape(-1,1)
        t0 = time.time()
        self.history_z = []
        self.history_loss = []
        self.data_obs = np.concatenate([self.data_x, self.data_y, self.data_v], axis=-1)
        if data_z_init is None:
            if self.params['pretrain']:
                self.pretrain(n_iter=pretrain_iter, batch_size=batch_size, batches_per_eval=batches_per_eval, verbose=verbose)
                data_z_init = self.e_net(self.data_v)
            else:
                data_z_init = np.random.normal(0, 1, size = (len(self.data_obs), sum(self.params['z_dims']))).astype('float32')

        self.data_z = tf.Variable(data_z_init, name="Latent Variable")
        self.data_z_init = tf.identity(self.data_z)
        best_loss = np.inf
        for epoch in range(epochs+1):
            sample_idx = np.random.choice(len(self.data_obs), len(self.data_obs), replace=False)
            for i in range(0,len(self.data_obs),batch_size):
                batch_idx = sample_idx[i:i+batch_size]
                # update model parameters of G, H, F with SGD
                batch_z = tf.Variable(tf.gather(self.data_z, batch_idx, axis = 0), name='batch_z')
                batch_obs = self.data_obs[batch_idx]
                batch_x = batch_obs[:,:1]
                batch_y = batch_obs[:,1:2]
                batch_v = batch_obs[:,2:]
                loss_v, loss_mse_v = self.update_g_net(batch_z, batch_v)
                loss_x, loss_mse_x = self.update_h_net(batch_z, batch_x)
                loss_y, loss_mse_y = self.update_f_net(batch_z, batch_x, batch_y)

                # update Z by maximizing a posterior or posterior mean
                loss_postrior_z, batch_z= self.update_latent_variable_sgd(batch_z, batch_x, batch_y, batch_v)
                self.data_z = tf.compat.v1.scatter_update(self.data_z, batch_idx, batch_z)
            if epoch % epochs_per_eval == 0:
                loss_contents = '''Epoch [%d, %.1f]: loss_px_z [%.4f], loss_mse_x [%.4f], loss_py_z [%.4f], loss_mse_y [%.4f], loss_pv_z [%.4f], loss_mse_v [%.4f], loss_postrior_z [%.4f]''' \
                %(epoch, time.time()-t0, loss_x, loss_mse_x, loss_y, loss_mse_y, loss_v, loss_mse_v, loss_postrior_z)
                if verbose:
                    print(loss_contents)
                causal_pre, mse_x, mse_y, mse_v = self.evaluate(stage = 'train')
                self.history_loss.append([mse_x, mse_y, mse_v])
                if epoch >= startoff and mse_y < best_loss:
                    best_loss = mse_y
                    self.best_causal_pre = causal_pre
                    self.best_epoch = epoch
                    if self.params['save_model']:
                        ckpt_save_path = self.ckpt_manager.save(epoch)
                        #print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
                if self.params['save_res']:
                    self.save('{}/causal_pre_at_{}.{}'.format(self.save_dir, epoch, save_format), causal_pre)
                    
                self.history_z.append(copy.copy(self.data_z))
                data_z0 = self.data_z[:,:self.params['z_dims'][0]]
                data_z1 = self.data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
                data_z2 = self.data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
                np.savez('%s/data_at_%d.npz'%(self.save_dir, epoch), data_z=self.data_z.numpy(), 
                        data_x_rec=self.h_net(tf.concat([data_z0, data_z2], axis=-1)).numpy(),
                        data_y_rec=self.f_net(tf.concat([data_z0, data_z1, self.data_x], axis=-1)).numpy(),
                        data_v_rec=self.g_net(self.data_z).numpy())

            if epoch in np.linspace(0, epochs, 5):
                causal_pre_v2 = self.predict(data = data_obs)
                np.save('%s/causal_pre_v2_at_%s.npy'%(self.save_dir, epoch), causal_pre_v2)
        
        np.save('%s/history_loss.npy'%(self.save_dir),np.array(self.history_loss))

        if self.params['save_res']:
            self.save('{}/causal_pre_final.{}'.format(self.save_dir,save_format), self.best_causal_pre)

        if self.params['binary_treatment']:
            self.ATE = np.mean(self.best_causal_pre)
            print('The average treatment effect (ATE) is', self.ATE)

    def evaluate(self, nb_intervals=200, stage='pretrain'):
        if stage == 'pretrain':
            self.data_z = self.e_net(self.data_v)
        data_z0 = self.data_z[:,:self.params['z_dims'][0]]
        data_z1 = self.data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = self.data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
        data_v_pred = self.g_net(self.data_z)[:,:self.params['v_dim']]
        data_y_pred = self.f_net(tf.concat([data_z0, data_z1, self.data_x], axis=-1))[:,:1]
        data_x_pred = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
        if self.params['binary_treatment']:
            data_x_pred = tf.sigmoid(data_x_pred)
        mse_v = np.mean((self.data_v-data_v_pred)**2)
        mse_x = np.mean((self.data_x-data_x_pred)**2)
        mse_y = np.mean((self.data_y-data_y_pred)**2)
        if self.params['binary_treatment']:
            #individual treatment effect (ITE) && average treatment effect (ATE)
            y_pred_pos = self.f_net(tf.concat([data_z0, data_z1, np.ones((len(self.data_x),1))], axis=-1))[:,:1]
            y_pred_neg = self.f_net(tf.concat([data_z0, data_z1, np.zeros((len(self.data_x),1))], axis=-1))[:,:1]
            ite_pre = y_pred_pos-y_pred_neg
            return ite_pre, mse_x, mse_y, mse_v
        else:
            #average dose response function (ADRF)
            dose_response = []
            for x in np.linspace(self.params['x_min'], self.params['x_max'], nb_intervals):
                data_x = np.tile(x, (len(self.data_x), 1))
                y_pred = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
                dose_response.append(np.mean(y_pred))
            return np.array(dose_response), mse_x, mse_y, mse_v

    def predict(self, data, nb_intervals=200, n_samples=3000, compute_mse=False, sample_y=True):
        """Evaluate the model on the test data and give estimation interval. ITE is estimated for binary treatment and ADRF is estimated for continous treatment.
        data: (np.ndarray): Input data with shape (n, p), where p is the dimension of X.
        nb_intervals: (int): Number of intervals for the dose response function.
        n_samples: (int): Number of samples for the MCMC posterior.
        sample_y: (bool): sample y from a normal distribution.
        return (np.ndarray): 
            ITE with shape (n_samples, n) containing all the MCMC samples.
            ADRF with shape (nb_intervals, n_samples) containing all the MCMC samples for each treatment value.
        """
        #posterior samples of P(Z|X,Y,V) with shape (n_samples, n_test, q)
        data_posterior_z = self.metropolis_hastings_sampler(data, n_keep=n_samples)

        #extract the components of Z for X,Y
        data_z0 = data_posterior_z[:,:,:self.params['z_dims'][0]]
        data_z1 = data_posterior_z[:,:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_posterior_z[:,:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
        
        if compute_mse:
            data_v_pred = np.array(list(map(lambda x: self.g_net.predict(x, verbose=0), 
                                            data_posterior_z)))[:,:,:self.params['v_dim']]
            data_y_pred = np.array(list(map(lambda x: self.f_net.predict(x, verbose=0), 
                                        np.concatenate([data_z0, data_z1, np.tile(data[0], (n_samples, 1, 1))], axis=-1))))[:,:,:1]
            data_x_pred = np.array(list(map(self.h_net,
                                        np.concatenate([data_z0, data_z2], axis=-1))))[:,:,:1]
            if self.params['binary_treatment']:
                data_x_pred = tf.sigmoid(data_x_pred)
            mse_x = np.mean((data[0]-np.mean(data_x_pred, axis=0))**2)
            mse_y = np.mean((data[1]-np.mean(data_y_pred, axis=0))**2)
            mse_v = np.mean((data[2]-np.mean(data_v_pred, axis=0))**2)
            print('MSE for X, Y, V:',mse_x, mse_y, mse_v)

        if self.params['binary_treatment']:
            
            #extract mean and sigma^2 of positive samples both with shape (n_keep, n_test)
            y_out_pos_all = np.array(list(map(lambda x: self.f_net.predict(x, verbose=0),
                                    np.concatenate([data_z0, data_z1, np.ones((n_samples,len(data[0]),1))], axis=-1))))
            mu_y_pos_all = y_out_pos_all[:,:,0]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                sigma_square_y = tf.nn.softplus(y_out_pos_all[:,:,1])
                
            #whether sample the predicted outcome from a normal distribution
            if sample_y:
                y_pred_pos_all = np.random.normal(loc = mu_y_pos_all, scale = np.sqrt(sigma_square_y))
            else:
                y_pred_pos_all = mu_y_pos_all
            
            #extract mean and sigma^2 of negative samples both with shape (n_keep, n_test)
            y_out_neg_all = np.array(list(map(lambda x: self.f_net.predict(x, verbose=0),
                                    np.concatenate([data_z0, data_z1, np.zeros((n_samples,len(data[0]),1))], axis=-1))))
            mu_y_neg_all = y_out_neg_all[:,:,0]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                sigma_square_y = tf.nn.softplus(y_out_neg_all[:,:,1])
                
            #whether sample the predicted outcome from a normal distribution
            if sample_y:
                y_pred_neg_all = np.random.normal(loc = mu_y_neg_all, scale = np.sqrt(sigma_square_y))
            else:
                y_pred_neg_all = mu_y_neg_all
                
            ite_pred_all = y_pred_pos_all-y_pred_neg_all

            return ite_pred_all
        else:
            dose_response = []
            for x in np.linspace(self.params['x_min'], self.params['x_max'], nb_intervals):
                data_x = np.tile(x, (n_samples, len(data[0]), 1))
                
                #extract mean and sigma^2 of samples given a treatment value both with shape (n_keep, n_test)
                y_out_all = np.array(list(map(lambda x: self.f_net.predict(x, verbose=0), 
                                    tf.concat([data_z0, data_z1, data_x], axis=-1))))
                mu_y_all = y_out_all[:,:,0]
                if 'sigma_y' in self.params:
                    sigma_square_y = self.params['sigma_y']**2
                else:
                    sigma_square_y = tf.nn.softplus(y_out_all[:,:,1])
                    
                #whether sample the predicted outcome from a normal distribution
                if sample_y:
                    y_pred_all = np.random.normal(loc = mu_y_all, scale = np.sqrt(sigma_square_y))
                else:
                    y_pred_all = mu_y_all
                
                dose_response.append(np.mean(y_pred_all, axis=1))
            return np.array(dose_response)
        
    def get_log_posterior(self, data_x, data_y, data_v, data_z, eps=1e-6):
        """
        Calculate log posterior.
        data_x: (np.ndarray): Input data with shape (n, 1), where p is the dimension of X.
        data_y: (np.ndarray): Input data with shape (n, 1), where q is the dimension of Y.
        data_v: (np.ndarray): Input data with shape (n, p), where r is the dimension of V.
        data_z: (np.ndarray): Input data with shape (n, q), where q is the dimension of Z.
        return (np.ndarray): Log posterior with shape (n, ).
        """
        data_z0 = data_z[:,:self.params['z_dims'][0]]
        data_z1 = data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]

        mu_v = self.g_net(data_z)[:,:self.params['v_dim']]
        if 'sigma_v' in self.params:
            sigma_square_v = self.params['sigma_v']**2
        else:
            #sigma_square_v = tf.nn.relu(self.g_net(data_z)[:,-1])+eps
            sigma_square_v = tf.nn.softplus(self.g_net(data_z)[:,-1])

        mu_x = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
        if 'sigma_x' in self.params:
            sigma_square_x = self.params['sigma_x']**2
        else:
            #sigma_square_x = tf.nn.relu(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1])+eps
            sigma_square_x = tf.nn.softplus(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1])

        mu_y = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
        if 'sigma_y' in self.params:
            sigma_square_y = self.params['sigma_y']**2
        else:
            #sigma_square_y = tf.nn.relu(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1])+eps
            sigma_square_y = tf.nn.softplus(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1])

        loss_pv_z = tf.reduce_sum((data_v - mu_v)**2, axis=1)/(2*sigma_square_v) + \
                self.params['v_dim'] * tf.math.log(sigma_square_v)/2
        
        if self.params['binary_treatment']:
            loss_px_z = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x,logits=mu_x))
            #loss_px_z = tf.reduce_mean((tf.sigmoid(mu_x) - data_x)**2)
        else:
            loss_px_z = tf.reduce_sum((data_x - mu_x)**2, axis=1)/(2*sigma_square_x) + \
                    tf.math.log(sigma_square_x)/2

        loss_py_zx = tf.reduce_sum((data_y - mu_y)**2, axis=1)/(2*sigma_square_y) + \
                tf.math.log(sigma_square_y)/2

        loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2

        loss_postrior_z = loss_pv_z + loss_px_z + loss_py_zx + loss_prior_z

        log_posterior = -loss_postrior_z
        return log_posterior


    def metropolis_hastings_sampler(self, data, q_sd = 1., burn_in = 500, n_keep = 5000):
        """
        Samples from the posterior distribution P(Z|X,Y,V) using the Metropolis-Hastings algorithm.

        Args:
            x (np.ndarray): Input data with shape (n, p), where p is the dimension of X.
            burn_in (int): Number of samples for burn-in.
            n_keep (int): Number of samples retained after burn-in.

        Returns:
            np.ndarray: Posterior samples with shape (n_keep, n, q), where q is the dimension of Z.
        """
        data_x, data_y, data_v = data

        # Initialize the state of n chains
        current_state = np.random.normal(0, 1, size = (len(data_x), sum(self.params['z_dims']))).astype('float32')

        # Initialize the list to store the samples
        samples = []
        counter = 0
        # Run the Metropolis-Hastings algorithm
        while len(samples) < n_keep:
            # Propose a new state by sampling from a multivariate normal distribution
            proposed_state = current_state + np.random.normal(0, q_sd, size = (len(data_x), sum(self.params['z_dims']))).astype('float32')

            # Compute the acceptance ratio
            proposed_log_posterior = self.get_log_posterior(data_x, data_y, data_v, proposed_state)
            current_log_posterior  = self.get_log_posterior(data_x, data_y, data_v, current_state)
            acceptance_ratio = np.exp(proposed_log_posterior-current_log_posterior)
            # Accept or reject the proposed state
            indices = np.random.rand(len(data_x)) < acceptance_ratio
            current_state[indices] = proposed_state[indices]

            # Append the current state to the list of samples
            if counter >= burn_in:
                samples.append(current_state.copy())
            
            counter += 1
        return np.array(samples)
    
    def save(self, fname, data):

        """Save the data to the specified path."""
        if fname[-3:] == 'npy':
            np.save(fname, data)
        elif fname[-3:] == 'txt' or 'csv':
            np.savetxt(fname, data, fmt='%.6f')
        else:
            print('Wrong saving format, please specify either .npy, .txt, or .csv')
            sys.exit()


class BayesPredGM(object):
    def __init__(self, params, timestamp=None, random_seed=None):
        super(BayesPredGM, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        self.gx_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = params['x_dim']+1, 
                                        model_name='gx_net', nb_units=params['gx_units'])

        self.gy_net = BaseFullyConnectedNet(input_dim=params['z_dim'],output_dim = params['y_dim']+1, 
                                        model_name='gy_net', nb_units=params['gy_units'])
                                        
        self.gx_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.gy_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
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

        self.ckpt = tf.train.Checkpoint(gx_net = self.gx_net,
                                    gy_net = self.gy_net,
                                    gx_optimizer = self.gx_optimizer,
                                    gy_optimizer = self.gy_optimizer,
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

        self.gx_net(np.zeros((1, self.params['z_dim'])))
        self.gy_net(np.zeros((1, self.params['z_dim'])))
        if print_summary:
            print(self.gx_net.summary())
            print(self.gy_net.summary())

    #update network for x
    @tf.function
    def update_gx_net(self, data_z, data_x, eps=1e-6):
        with tf.GradientTape(persistent=True) as gen_tape_x:
            mu_x = self.gx_net(data_z)[:,:self.params['x_dim']]
            if 'sigma_x' in self.params:
                sigma_square_x = self.params['sigma_x']**2
            else:
                sigma_square_x = tf.nn.relu(self.gx_net(data_z)[:,-1])+eps

            #loss = -log(p(x|z))
            loss_mse = tf.reduce_mean((data_x - mu_x)**2)
            loss_x = tf.reduce_sum((data_x - mu_x)**2, axis=1)/(2*sigma_square_x) + \
                    self.params['x_dim'] * tf.math.log(sigma_square_x)/2
            loss_x = tf.reduce_mean(loss_x)

        # Calculate the gradients for generator
        gx_gradients = gen_tape_x.gradient(loss_x, self.gx_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.gx_optimizer.apply_gradients(zip(gx_gradients, self.gx_net.trainable_variables))
        return loss_x, loss_mse

    #update network for y
    @tf.function
    def update_gy_net(self, data_z, data_y, eps=1e-6):
        with tf.GradientTape(persistent=True) as gen_tape_y:
            mu_y = self.gy_net(data_z)[:,:self.params['y_dim']]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                sigma_square_y = tf.nn.relu(self.gy_net(data_z)[:,-1])+eps

            #loss = -log(p(y|z))
            loss_mse = tf.reduce_mean((data_y - mu_y)**2)
            loss_y = tf.reduce_sum((data_y - mu_y)**2, axis=1)/(2*sigma_square_y) + \
                    self.params['y_dim'] * tf.math.log(sigma_square_y)/2
            loss_y = tf.reduce_mean(loss_y)

        # Calculate the gradients for generator
        gy_gradients = gen_tape_y.gradient(loss_y, self.gy_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.gy_optimizer.apply_gradients(zip(gy_gradients, self.gy_net.trainable_variables))
        return loss_y, loss_mse

    # update posterior of latent variables Z
    #@tf.function
    def update_latent_variable_sgd(self, data_z, data_x, data_y, eps=1e-6):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(data_z)

            mu_x = self.gx_net(data_z)[:,:self.params['x_dim']]
            if 'sigma_x' in self.params:
                sigma_square_x = self.params['sigma_x']**2
            else:
                sigma_square_x = tf.nn.relu(self.gx_net(data_z)[:,-1])+eps

            mu_y = self.gy_net(data_z)[:,:self.params['y_dim']]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                sigma_square_y = tf.nn.relu(self.gy_net(data_z)[:,-1])+eps
            
            loss_px_z = tf.reduce_sum((data_x - mu_x)**2, axis=1)/(2*sigma_square_x) + \
                    self.params['x_dim'] * tf.math.log(sigma_square_x)/2

            loss_py_z = tf.reduce_sum((data_y - mu_y)**2, axis=1)/(2*sigma_square_y) + \
                    self.params['y_dim'] * tf.math.log(sigma_square_y)/2

            loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2
            loss_postrior_z = tf.reduce_mean(loss_px_z) + \
                                tf.reduce_mean(loss_py_z) + tf.reduce_mean(loss_prior_z)

            loss_postrior_z = loss_postrior_z/(self.params['x_dim']+self.params['y_dim'])

        # self.posterior_optimizer.build(data_z)
        # calculate the gradients for generators and discriminators
        posterior_gradients = tape.gradient(loss_postrior_z, [data_z])
        # apply the gradients to the optimizer
        self.posterior_optimizer.apply_gradients(zip(posterior_gradients, [data_z]))
        return loss_postrior_z, data_z
    
    def train_epoch(self, data_train, data_test, data_z_init=None,
            batch_size=32, epochs=100, epochs_per_eval=5, startoff=0,
            verbose=1, save_format='txt'):
        
        if self.params['save_res']:
            f_params = open('{}/params.txt'.format(self.save_dir),'w')
            f_params.write(str(self.params))
            f_params.close()

        t0 = time.time()
        self.history_z = []
        self.history_loss = []
        self.data_x, self.data_y = data_train

        assert len(self.data_x) == len(self.data_y), "X and Y should be the same length"

        if data_z_init is None:
            data_z_init = np.random.normal(0, 1, size = (len(self.data_x), self.params['z_dim'])).astype('float32')

        self.data_z = tf.Variable(data_z_init, name="Latent Variable")
        self.data_z_init = tf.identity(self.data_z)
        best_loss = np.inf
        for epoch in range(epochs+1):
            sample_idx = np.random.choice(len(self.data_x), len(self.data_x), replace=False)
            for i in range(0,len(self.data_x),batch_size):
                # get batch data
                batch_idx = sample_idx[i:i+batch_size]
                batch_z = tf.Variable(tf.gather(self.data_z, batch_idx, axis = 0), name='batch_z')
                batch_x = self.data_x[batch_idx]
                batch_y = self.data_y[batch_idx]

                # update model parameters of G_x with SGD
                loss_x, loss_x_mse = self.update_gx_net(batch_z, batch_x)

                # update model parameters of G_y with SGD
                loss_y, loss_y_mse = self.update_gy_net(batch_z, batch_y)

                # update Z by maximizing a posterior or posterior mean
                loss_postrior_z, batch_z= self.update_latent_variable_sgd(batch_z, batch_x, batch_y)
                self.data_z = tf.compat.v1.scatter_update(self.data_z, batch_idx, batch_z)
            
            if epoch % epochs_per_eval == 0:
                y_pred_all, sigma_square_y, mse_y, corr = self.evaluate(data_test)
                self.history_loss.append(mse_y)
                loss_contents = '''Epoch [%d, %.1f]: mse_y [%.4f], corr [%.4f], loss_x_mse [%.4f], loss_y_mse [%.4f], loss_postrior_z [%.4f]''' \
                %(epoch, time.time()-t0, mse_y, corr, loss_x_mse, loss_y_mse, loss_postrior_z)
                if verbose:
                    print(loss_contents)
                if epoch >= startoff and mse_y < best_loss:
                    best_loss = mse_y
                    self.best_epoch = epoch
                    if self.params['save_model']:
                        ckpt_save_path = self.ckpt_manager.save(epoch)
                        #print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

                self.history_z.append(copy.copy(self.data_z))
                np.savez('%s/data_at_%d.npz'%(self.save_dir, epoch), data_z=self.data_z.numpy(),
                        data_x_rec=self.gx_net(self.data_z).numpy(),
                        data_y_rec=self.gy_net(self.data_z).numpy(),
                        sigma_square_y = sigma_square_y,
                        y_pred_all = y_pred_all)
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                plt.violinplot(self.data_z.numpy())
                plt.title('Epoch %d'%(epoch))
                plt.savefig('%s/violinplot_%d.png'%(self.save_dir, epoch))
                plt.close()
                plt.scatter(self.data_z.numpy()[:,0],self.data_z.numpy()[:,1])
                plt.title('Epoch %d'%(epoch))
                plt.savefig('%s/scatter_%d.png'%(self.save_dir, epoch))
                plt.close()
        np.save('%s/history_loss.npy'%(self.save_dir),np.array(self.history_loss))

    def evaluate(self, data, eps=1e-6):
        from scipy.stats import pearsonr
        """Evaluate the model on the test data and give estimation interval of P(Y|X)."""
        data_x_test, data_y_test = data

        #posterior samples of P(Z|X) with shape (n_keep, n_test, q)
        data_pz_x = self.metropolis_hastings_sampler(data_x_test) 

        #mean of P(Y|Z): shape (n_keep, n, y_dim)
        gy_out = np.array(list(map(self.gy_net, data_pz_x)))
        mu_y_all = gy_out[:,:,:self.params['y_dim']]

        #sd of P(Y|Z): a constant or variable with shape (n_keep, n, 1)
        if 'sigma_y' in self.params:
            sigma_square_y = self.params['sigma_y']**2
        else:
            sigma_square_y = tf.nn.relu(gy_out[:,:,-1:])+eps

        #sample Y with a normal distribution N(mu_y, sigma_y**2) with shape (n_keep, n, y_dim)
        y_pred_all = np.random.normal(loc = mu_y_all, scale = np.sqrt(sigma_square_y))
        y_pred_mean = np.mean(y_pred_all, axis=0)
        mse_y = np.mean((data_y_test - y_pred_mean)**2)
        corr = pearsonr(data_y_test[:,0], y_pred_mean[:,0])[0]
        return y_pred_all, sigma_square_y, mse_y, corr
        
    def get_log_posterior(self, data_x, data_z, eps=1e-6):
        """
        Calculate log posterior.
        data_x: (np.ndarray): Input data with shape (n, p), where p is the dimension of X.
        data_z: (np.ndarray): Input data with shape (n, q), where q is the dimension of Z.
        return (np.ndarray): Log posterior with shape (n, ).
        """
        
        mu_x = self.gx_net(data_z)[:,:self.params['x_dim']]

        if 'sigma_x' in self.params:
            sigma_square_x = self.params['sigma_x']**2
        else:
            sigma_square_x = tf.nn.relu(self.gx_net(data_z)[:,-1])+eps
        log_likelihood = -np.sum((data_x-mu_x) ** 2,axis=1)/(2 * sigma_square_x) - np.log(sigma_square_x)*self.params['x_dim']/2
        log_prior = -np.sum(data_z ** 2,axis=1)/2
        log_posterior = log_likelihood + log_prior
        return log_posterior


    def metropolis_hastings_sampler(self, data_x, q_sd = 1., burn_in = 500, n_keep = 5000):
        """
        Samples from the posterior distribution P(Z|X=x) using the Metropolis-Hastings algorithm.

        Args:
            x (np.ndarray): Input data with shape (n, p), where p is the dimension of X.
            burn_in (int): Number of samples for burn-in.
            n_keep (int): Number of samples retained after burn-in.

        Returns:
            np.ndarray: Posterior samples with shape (n_keep, n, q), where q is the dimension of Z.
        """
        # Initialize the state of n chains
        current_state = np.random.normal(0, 1, size = (len(data_x), self.params['z_dim'])).astype('float32')

        # Initialize the list to store the samples
        samples = []
        counter = 0
        # Run the Metropolis-Hastings algorithm
        while len(samples) < n_keep:
            # Propose a new state by sampling from a multivariate normal distribution
            proposed_state = current_state + np.random.normal(0, q_sd, size = (len(data_x), self.params['z_dim'])).astype('float32')

            # Compute the acceptance ratio
            proposed_log_posterior = self.get_log_posterior(data_x, proposed_state)
            current_log_posterior  = self.get_log_posterior(data_x, current_state)
            acceptance_ratio = np.exp(proposed_log_posterior-current_log_posterior)
            # Accept or reject the proposed state
            indices = np.random.rand(len(data_x)) < acceptance_ratio
            current_state[indices] = proposed_state[indices]

            # Append the current state to the list of samples
            if counter >= burn_in:
                samples.append(current_state.copy())
            
            counter += 1
        return np.array(samples)

    def save(self, fname, data):
        """Save the data to the specified path."""
        if fname[-3:] == 'npy':
            np.save(fname, data)
        elif fname[-3:] == 'txt' or 'csv':
            np.savetxt(fname, data, fmt='%.6f')
        else:
            print('Wrong saving format, please specify either .npy, .txt, or .csv')
            sys.exit()



class BayesPredGM_Partition(object):
    """ Bayesian Prediction with latent space partition
    """
    def __init__(self, params, timestamp=None, random_seed=None):
        super(BayesPredGM_Partition, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        self.gx_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][2],output_dim = params['x_dim']+1, 
                                        model_name='gx_net', nb_units=params['gx_units'])

        self.gy_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][1],output_dim = params['y_dim']+1, 
                                        model_name='gy_net', nb_units=params['gy_units'])
                                        
        self.gx_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.gy_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.posterior_optimizer = tf.keras.optimizers.legacy.Adam(params['lr_z'], beta_1=0.9, beta_2=0.99)

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

        self.ckpt = tf.train.Checkpoint(gx_net = self.gx_net,
                                    gy_net = self.gy_net,
                                    gx_optimizer = self.gx_optimizer,
                                    gy_optimizer = self.gy_optimizer,
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

        self.gx_net(np.zeros((1, self.params['z_dims'][0] + self.params['z_dims'][2])))
        self.gy_net(np.zeros((1, self.params['z_dims'][0] + self.params['z_dims'][1])))
        if print_summary:
            print(self.gx_net.summary())
            print(self.gy_net.summary())

    #update network for x
    @tf.function
    def update_gx_net(self, data_z, data_x, eps=1e-6):
        with tf.GradientTape(persistent=True) as gen_tape_x:
            data_z0 = data_z[:,:self.params['z_dims'][0]]
            data_z2 = data_z[:,-self.params['z_dims'][-1]:]
            mu_x = self.gx_net(tf.concat([data_z0, data_z2], axis=-1))[:,:self.params['x_dim']]
            if 'sigma_x' in self.params:
                sigma_square_x = self.params['sigma_x']**2
            else:
                sigma_square_x = tf.nn.relu(self.gx_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1])+eps

            #loss = -log(p(x|z))
            loss_mse = tf.reduce_mean((data_x - mu_x)**2)
            loss_x = tf.reduce_sum((data_x - mu_x)**2, axis=1)/(2*sigma_square_x) + \
                    self.params['x_dim'] * tf.math.log(sigma_square_x)/2
            loss_x = tf.reduce_mean(loss_x)

        # Calculate the gradients for generator
        gx_gradients = gen_tape_x.gradient(loss_x, self.gx_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.gx_optimizer.apply_gradients(zip(gx_gradients, self.gx_net.trainable_variables))
        return loss_x, loss_mse

    #update network for y
    @tf.function
    def update_gy_net(self, data_z, data_y, eps=1e-6):
        with tf.GradientTape(persistent=True) as gen_tape_y:
            data_z0 = data_z[:,:self.params['z_dims'][0]]
            data_z1 = data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
            mu_y = self.gy_net(tf.concat([data_z0, data_z1], axis=-1))[:,:self.params['y_dim']]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                sigma_square_y = tf.nn.relu(self.gy_net(tf.concat([data_z0, data_z1], axis=-1))[:,-1])+eps

            #loss = -log(p(y|z))
            loss_mse = tf.reduce_mean((data_y - mu_y)**2)
            loss_y = tf.reduce_sum((data_y - mu_y)**2, axis=1)/(2*sigma_square_y) + \
                    self.params['y_dim'] * tf.math.log(sigma_square_y)/2
            loss_y = tf.reduce_mean(loss_y)

        # Calculate the gradients for generator
        gy_gradients = gen_tape_y.gradient(loss_y, self.gy_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.gy_optimizer.apply_gradients(zip(gy_gradients, self.gy_net.trainable_variables))
        return loss_y, loss_mse

    # update posterior of latent variables Z
    #@tf.function
    def update_latent_variable_sgd(self, data_z, data_x, data_y, eps=1e-6):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(data_z)
            data_z0 = data_z[:,:self.params['z_dims'][0]]
            data_z1 = data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
            data_z2 = data_z[:,-self.params['z_dims'][-1]:]
            mu_x = self.gx_net(tf.concat([data_z0, data_z2], axis=-1))[:,:self.params['x_dim']]
            if 'sigma_x' in self.params:
                sigma_square_x = self.params['sigma_x']**2
            else:
                sigma_square_x = tf.nn.relu(self.gx_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1])+eps

            mu_y = self.gy_net(tf.concat([data_z0, data_z1], axis=-1))[:,:self.params['y_dim']]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                sigma_square_y = tf.nn.relu(self.gy_net(tf.concat([data_z0, data_z1], axis=-1))[:,-1])+eps
            
            loss_px_z = tf.reduce_sum((data_x - mu_x)**2, axis=1)/(2*sigma_square_x) + \
                    self.params['x_dim'] * tf.math.log(sigma_square_x)/2

            loss_py_z = tf.reduce_sum((data_y - mu_y)**2, axis=1)/(2*sigma_square_y) + \
                    self.params['y_dim'] * tf.math.log(sigma_square_y)/2

            loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2
            loss_postrior_z = tf.reduce_mean(loss_px_z) + \
                                tf.reduce_mean(loss_py_z) + tf.reduce_mean(loss_prior_z)

            loss_postrior_z = loss_postrior_z/(self.params['x_dim']+self.params['y_dim'])

        # self.posterior_optimizer.build(data_z)
        # calculate the gradients for generators and discriminators
        posterior_gradients = tape.gradient(loss_postrior_z, [data_z])
        # apply the gradients to the optimizer
        self.posterior_optimizer.apply_gradients(zip(posterior_gradients, [data_z]))
        return loss_postrior_z, data_z
    
    def train_epoch(self, data_train, data_test, data_z_init=None,
            batch_size=32, epochs=100, epochs_per_eval=5, startoff=0,
            verbose=1, save_format='txt'):
        
        if self.params['save_res']:
            f_params = open('{}/params.txt'.format(self.save_dir),'w')
            f_params.write(str(self.params))
            f_params.close()

        t0 = time.time()
        self.history_z = []
        self.history_loss = []
        self.data_x, self.data_y = data_train

        assert len(self.data_x) == len(self.data_y), "X and Y should be the same length"

        if data_z_init is None:
            data_z_init = np.random.normal(0, 1, size = (len(self.data_x), sum(self.params['z_dims']))).astype('float32')

        self.data_z = tf.Variable(data_z_init, name="Latent Variable")
        self.data_z_init = tf.identity(self.data_z)
        best_loss = np.inf
        for epoch in range(epochs+1):
            sample_idx = np.random.choice(len(self.data_x), len(self.data_x), replace=False)
            loss_total = []
            for i in range(0,len(self.data_x),batch_size):
                # get batch data
                batch_idx = sample_idx[i:i+batch_size]
                batch_z = tf.Variable(tf.gather(self.data_z, batch_idx, axis = 0), name='batch_z')
                batch_x = self.data_x[batch_idx]
                batch_y = self.data_y[batch_idx]

                # update model parameters of G_x with SGD
                loss_x, loss_x_mse = self.update_gx_net(batch_z, batch_x)

                # update model parameters of G_y with SGD
                loss_y, loss_y_mse = self.update_gy_net(batch_z, batch_y)

                # update Z by maximizing a posterior or posterior mean
                loss_postrior_z, batch_z= self.update_latent_variable_sgd(batch_z, batch_x, batch_y)
                self.data_z = tf.compat.v1.scatter_update(self.data_z, batch_idx, batch_z)
                loss_total.append([loss_x_mse, loss_y_mse, loss_postrior_z])
            if epoch % epochs_per_eval == 0:
                loss_aveg = np.mean(loss_total, axis=0)
                y_pred_all, sigma_square_y, mse_y, corr = self.evaluate(data_test)
                self.history_loss.append([loss_aveg[0], loss_aveg[1], loss_aveg[2], mse_y, corr])
                loss_contents = '''Epoch [%d, %.1f]: x_mse [%.4f], y_mse [%.4f], postrior_z [%.4f], test_mse_y [%.4f], corr [%.4f]''' \
                %(epoch, time.time()-t0, loss_aveg[0], loss_aveg[1], loss_aveg[2], mse_y, corr)
                if verbose:
                    print(loss_contents)
                if epoch >= startoff and mse_y < best_loss:
                    best_loss = mse_y
                    self.best_epoch = epoch
                    if self.params['save_model']:
                        ckpt_save_path = self.ckpt_manager.save(epoch)
                        #print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

                self.history_z.append(copy.copy(self.data_z))
                np.savez('%s/data_at_%d.npz'%(self.save_dir, epoch), data_z=self.data_z.numpy(),
                        sigma_square_y = sigma_square_y,
                        y_pred_all = y_pred_all)
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                plt.violinplot(self.data_z.numpy())
                plt.title('Epoch %d'%(epoch))
                plt.savefig('%s/violinplot_%d.png'%(self.save_dir, epoch))
                plt.close()
                plt.scatter(self.data_z.numpy()[:,0],self.data_z.numpy()[:,1])
                plt.title('Epoch %d'%(epoch))
                plt.savefig('%s/scatter_%d.png'%(self.save_dir, epoch))
                plt.close()
        self.save('{}/history_loss.txt'.format(self.save_dir), np.array(self.history_loss))

    def evaluate(self, data, eps=1e-6):
        from scipy.stats import pearsonr
        """Evaluate the model on the test data and give estimation interval of P(Y|X)."""
        data_x_test, data_y_test = data

        #posterior samples of P(Z|X) with shape (n_keep, n_test, q)
        data_pz_x = self.metropolis_hastings_sampler(data_x_test) 

        #extract the components of Z for Y
        data_z0 = data_pz_x[:,:,:self.params['z_dims'][0]]
        data_z1 = data_pz_x[:,:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]

        #mean of P(Y|Z): shape (n_keep, n, y_dim)
        gy_out = np.array(list(map(self.gy_net, tf.concat([data_z0, data_z1], axis=-1))))
        mu_y_all = gy_out[:,:,:self.params['y_dim']]

        #sd of P(Y|Z): a constant or variable with shape (n_keep, n, 1)
        if 'sigma_y' in self.params:
            sigma_square_y = self.params['sigma_y']**2
        else:
            sigma_square_y = tf.nn.relu(gy_out[:,:,-1:])+eps

        #sample Y with a normal distribution N(mu_y, sigma_y**2) with shape (n_keep, n, y_dim)
        y_pred_all = np.random.normal(loc = mu_y_all, scale = np.sqrt(sigma_square_y))
        y_pred_mean = np.mean(y_pred_all, axis=0)
        mse_y = np.mean((data_y_test - y_pred_mean)**2)
        corr = pearsonr(data_y_test[:,0], y_pred_mean[:,0])[0]
        return y_pred_all, sigma_square_y, mse_y, corr
        
    def get_log_posterior(self, data_x, data_z, eps=1e-6):
        """
        Calculate log posterior.
        data_x: (np.ndarray): Input data with shape (n, p), where p is the dimension of X.
        data_z: (np.ndarray): Input data with shape (n, q), where q is the dimension of Z.
        return (np.ndarray): Log posterior with shape (n, ).
        """
        data_z0 = data_z[:,:self.params['z_dims'][0]]
        data_z2 = data_z[:,-self.params['z_dims'][-1]:]
            
        mu_x = self.gx_net(tf.concat([data_z0, data_z2], axis=-1))[:,:self.params['x_dim']]

        if 'sigma_x' in self.params:
            sigma_square_x = self.params['sigma_x']**2
        else:
            sigma_square_x = tf.nn.relu(self.gx_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1])+eps
        log_likelihood = -np.sum((data_x-mu_x) ** 2,axis=1)/(2 * sigma_square_x) - np.log(sigma_square_x)*self.params['x_dim']/2
        log_prior = -np.sum(data_z ** 2,axis=1)/2
        log_posterior = log_likelihood + log_prior
        return log_posterior


    def metropolis_hastings_sampler(self, data_x, q_sd = 1., burn_in = 500, n_keep = 5000):
        """
        Samples from the posterior distribution P(Z|X=x) using the Metropolis-Hastings algorithm.

        Args:
            x (np.ndarray): Input data with shape (n, p), where p is the dimension of X.
            burn_in (int): Number of samples for burn-in.
            n_keep (int): Number of samples retained after burn-in.

        Returns:
            np.ndarray: Posterior samples with shape (n_keep, n, q), where q is the dimension of Z.
        """
        # Initialize the state of n chains
        current_state = np.random.normal(0, 1, size = (len(data_x), sum(self.params['z_dims']))).astype('float32')

        # Initialize the list to store the samples
        samples = []
        counter = 0
        # Run the Metropolis-Hastings algorithm
        while len(samples) < n_keep:
            # Propose a new state by sampling from a multivariate normal distribution
            proposed_state = current_state + np.random.normal(0, q_sd, size = (len(data_x), sum(self.params['z_dims']))).astype('float32')

            # Compute the acceptance ratio
            proposed_log_posterior = self.get_log_posterior(data_x, proposed_state)
            current_log_posterior  = self.get_log_posterior(data_x, current_state)
            acceptance_ratio = np.exp(proposed_log_posterior-current_log_posterior)
            # Accept or reject the proposed state
            indices = np.random.rand(len(data_x)) < acceptance_ratio
            current_state[indices] = proposed_state[indices]

            # Append the current state to the list of samples
            if counter >= burn_in:
                samples.append(current_state.copy())
            
            counter += 1
        return np.array(samples)

    def save(self, fname, data):
        """Save the data to the specified path."""
        if fname[-3:] == 'npy':
            np.save(fname, data)
        elif fname[-3:] == 'txt' or 'csv':
            np.savetxt(fname, data, fmt='%.6f')
        else:
            print('Wrong saving format, please specify either .npy, .txt, or .csv')
            sys.exit()