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
        self.g_net = BaseFullyConnectedNet(input_dim=sum(params['z_dims']),output_dim = params['v_dim'], 
                                        model_name='g_net', nb_units=params['g_units'])
        self.f_net = BaseFullyConnectedNet(input_dim=1+params['z_dims'][0]+params['z_dims'][2],
                                        output_dim = 1, model_name='f_net', nb_units=params['f_units'])
        self.h_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][1],
                                        output_dim = 1, model_name='h_net', nb_units=params['h_units'])
        
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
                                    f_net = self.f_net,
                                    h_net = self.h_net,
                                    g_optimizer = self.g_optimizer,
                                    f_optimizer = self.f_optimizer,
                                    h_optimizer = self.h_optimizer,
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

        self.g_net(np.zeros((1, sum(self.params['z_dims']))))
        self.f_net(np.zeros((1, 1+self.params['z_dims'][0]+self.params['z_dims'][2])))
        self.h_net(np.zeros((1, self.params['z_dims'][0]+self.params['z_dims'][1])))
        if print_summary:
            print(self.g_net.summary())
            print(self.f_net.summary())    
            print(self.h_net.summary()) 

    #update network for covariates V
    #@tf.function
    def update_g_net(self, data_z, data_v, eps=1e-6):
        with tf.GradientTape(persistent=True) as gen_tape:
            mu_v = self.g_net(data_z)[:,:self.params['v_dim']]
            if 'sigma_v' in self.params:
                sigma_square_v = self.params['sigma_v']**2
            else:
                sigma_square_v = tf.nn.relu(self.g_net(data_z)[:,-1:])+eps
            #loss = -log(p(x|z))
            loss_mse = tf.reduce_mean((data_v - mu_v)**2)
            loss_v = tf.reduce_mean((data_v - mu_v)**2/(2*sigma_square_v)) + \
                    tf.reduce_mean(tf.math.log(sigma_square_v))/2

        # Calculate the gradients for generators and discriminators
        g_gradients = gen_tape.gradient(loss_mse, self.g_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_net.trainable_variables))
        return loss_v, loss_mse
    
    #update network for treatment X
    #@tf.function
    def update_h_net(self, data_z, data_x, eps=1e-6):
        with tf.GradientTape(persistent=True) as gen_tape:
            data_z0 = data_z[:,:self.params['z_dims'][0]]
            data_z2 = data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
            mu_x = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
            if 'sigma_x' in self.params:
                sigma_square_x = self.params['sigma_x']**2
            else:
                sigma_square_x = tf.nn.relu(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1:])+eps
            #loss = -log(p(x|z))
            loss_mse = tf.reduce_mean((data_x - mu_x)**2)
            loss_x = tf.reduce_mean((data_x - mu_x)**2/(2*sigma_square_x)) + \
                    tf.reduce_mean(tf.math.log(sigma_square_x))/2
            loss_x = loss_x/self.params['v_dim']
        # Calculate the gradients for generators and discriminators
        h_gradients = gen_tape.gradient(loss_mse, self.h_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.h_optimizer.apply_gradients(zip(h_gradients, self.h_net.trainable_variables))
        return loss_x, loss_mse
    
    #update network for outcome Y
    #@tf.function
    def update_f_net(self, data_z, data_x, data_y, eps=1e-6):
        with tf.GradientTape(persistent=True) as gen_tape:
            data_z0 = data_z[:,:self.params['z_dims'][0]]
            data_z1 = data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
            mu_y = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                sigma_square_y = tf.nn.relu(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1:])+eps
            #loss = -log(p(y|z,x))
            loss_mse = tf.reduce_mean((data_y - mu_y)**2)
            loss_y = tf.reduce_mean((data_y - mu_y)**2/(2*sigma_square_y)) + \
                    tf.reduce_mean(tf.math.log(sigma_square_y))/2
            loss_y = loss_y/self.params['v_dim']
        # Calculate the gradients for generators and discriminators
        f_gradients = gen_tape.gradient(loss_mse, self.f_net.trainable_variables)
        
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

            mu_v = self.g_net(data_z)[:,:self.params['v_dim']]
            if 'sigma_v' in self.params:
                sigma_square_v = self.params['sigma_v']**2
            else:
                sigma_square_v = tf.nn.relu(self.g_net(data_z)[:,-1:])+eps

            mu_x = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
            if 'sigma_x' in self.params:
                sigma_square_x = self.params['sigma_x']**2
            else:
                sigma_square_x = tf.nn.relu(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1:])+eps

            mu_y = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                sigma_square_y = tf.nn.relu(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1:])+eps
            
            loss_pv_z = tf.reduce_mean((data_v - mu_v)**2/(2*sigma_square_v)) + \
                    tf.reduce_mean(tf.math.log(sigma_square_v))/2
            loss_px_z = tf.reduce_mean((data_x - mu_x)**2/(2*sigma_square_x)) + \
                    tf.reduce_mean(tf.math.log(sigma_square_x))/2
            loss_py_zx = tf.reduce_mean((data_y - mu_y)**2/(2*sigma_square_y)) + \
                    tf.reduce_mean(tf.math.log(sigma_square_y))/2
            loss_prior_z =  tf.reduce_mean(data_z**2)/2
            loss_postrior_z = self.params['v_dim']*loss_pv_z + loss_px_z + loss_py_zx + sum(self.params['z_dims'])*loss_prior_z

            loss_postrior_z = loss_postrior_z/self.params['v_dim']

        #self.posterior_optimizer.build(data_z)
        # Calculate the gradients for generators and discriminators
        posterior_gradients = tape.gradient(loss_postrior_z, [data_z])
        #print(posterior_gradients)
        #sys.exit()
        # Apply the gradients to the optimizer
        self.posterior_optimizer.apply_gradients(zip(posterior_gradients, [data_z]))
        #print('2',data_z)
        #print(self.g_net.trainable_variables[0])
        #sys.exit()
        return loss_postrior_z, data_z

    def train_epoch(self, data_obs, data_z_init=None, normalize=False,
            batch_size=32, epochs=1000, epochs_per_eval=10, epochs_per_save=100,
            startoff=0, verbose=1, save_format='txt'):
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
            data_z_init = np.random.normal(0, 1, size = (len(self.data_obs), sum(self.params['z_dims']))).astype('float32')
        else:
            assert len(data_z_init) == len(self.data_obs), "Sample size does not match!"
        #self.data_z = data_z_init
        self.data_z = tf.Variable(data_z_init, name="Latent Variable")
        self.data_z_init = tf.identity(self.data_z)
        best_loss = np.inf
        for epoch in range(epochs+1):
            sample_idx = np.random.choice(len(self.data_obs), len(self.data_obs), replace=False)
            for i in range(0,len(self.data_obs),batch_size):
                batch_idx = sample_idx[i:i+batch_size]
                #update model parameters of G with SGD
                batch_z = tf.Variable(tf.gather(self.data_z, batch_idx, axis = 0), name='batch_z')
                batch_obs = self.data_obs[batch_idx]
                batch_x = batch_obs[:,:1]
                batch_y = batch_obs[:,1:2]
                batch_v = batch_obs[:,2:]
                loss_v, loss_mse_v = self.update_g_net(batch_z, batch_v)
                loss_x, loss_mse_x = self.update_h_net(batch_z, batch_x)
                loss_y, loss_mse_y = self.update_f_net(batch_z, batch_x, batch_y)

                #update Z by maximizing a posterior or posterior mean
                loss_postrior_z, batch_z= self.update_latent_variable_sgd(batch_z, batch_x, batch_y, batch_v)
                self.data_z = tf.compat.v1.scatter_update(self.data_z, batch_idx, batch_z)
            if epoch % epochs_per_eval == 0:
                self.history_loss.append([loss_x, loss_y, loss_v, loss_postrior_z])

                loss_contents = '''Epoch [%d, %.1f]: loss_px_z [%.4f], loss_mse_x [%.4f], loss_py_z [%.4f], loss_mse_y [%.4f], loss_pv_z [%.4f], loss_mse_v [%.4f], loss_postrior_z [%.4f]''' \
                %(epoch, time.time()-t0, loss_x, loss_mse_x, loss_y, loss_mse_y, loss_v, loss_mse_v, loss_postrior_z)
                if verbose:
                    print(loss_contents)
                causal_pre, _, mse_y = self.evaluate()
                if epoch >= startoff and mse_y < best_loss:
                    best_loss = mse_y
                    self.best_causal_pre = causal_pre
                    self.best_batch_idx = batch_idx
                    if self.params['save_model']:
                        ckpt_save_path = self.ckpt_manager.save(epoch)
                        #print('Saving checkpoint for iteration {} at {}'.format(batch_idx, ckpt_save_path))
                if self.params['save_res'] and epoch > 0 and epoch % epochs_per_save == 0:
                    self.save('{}/causal_pre_at_{}.{}'.format(self.save_dir, epoch, save_format), self.best_causal_pre)

                self.history_z.append(copy.copy(self.data_z))
                data_z0 = self.data_z[:,:self.params['z_dims'][0]]
                data_z1 = self.data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
                data_z2 = self.data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
                np.savez('%s/data_at_%d.npz'%(self.save_dir, epoch), data_z=self.data_z.numpy(), 
                        data_x_rec=self.h_net(tf.concat([data_z0, data_z2], axis=-1)).numpy(),
                        data_y_rec=self.f_net(tf.concat([data_z0, data_z1, self.data_x], axis=-1)).numpy(),
                        data_v_rec=self.g_net(self.data_z).numpy())
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

        if self.params['save_res']:
            self.save('{}/causal_pre_final.{}'.format(self.save_dir,save_format), self.best_causal_pre)

        if self.params['binary_treatment']:
            self.ATE = np.mean(self.best_causal_pre)
            print('The average treatment effect (ATE) is', self.ATE)

    def evaluate(self, nb_intervals=200):
        data_z0 = self.data_z[:,:self.params['z_dims'][0]]
        data_z1 = self.data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = self.data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
        data_y_pred = self.f_net(tf.concat([data_z0, data_z1, self.data_x], axis=-1))
        data_x_pred = self.h_net(tf.concat([data_z0, data_z2], axis=-1))
        if self.params['binary_treatment']:
            data_x_pred = tf.sigmoid(data_x_pred)
        mse_x = np.mean((self.data_x-data_x_pred)**2)
        mse_y = np.mean((self.data_y-data_y_pred)**2)
        if self.params['binary_treatment']:
            #individual treatment effect (ITE) && average treatment effect (ATE)
            y_pred_pos = self.f_net(tf.concat([data_z0, data_z1, np.ones((len(data_x),1))], axis=-1))
            y_pred_neg = self.f_net(tf.concat([data_z0, data_z1, np.zeros((len(data_x),1))], axis=-1))
            ite_pre = y_pred_pos-y_pred_neg
            return ite_pre, mse_x, mse_y
        else:
            #average dose response function (ADRF)
            dose_response = []
            for x in np.linspace(self.params['x_min'], self.params['x_max'], nb_intervals):
                data_x = np.tile(x, (len(self.data_x), 1))
                y_pred = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))
                dose_response.append(np.mean(y_pred))
            return np.array(dose_response), mse_x, mse_y
        
    def save(self, fname, data):
        """Save the data to the specified path."""
        if fname[-3:] == 'npy':
            np.save(fname, data)
        elif fname[-3:] == 'txt' or 'csv':
            np.savetxt(fname, data, fmt='%.6f')
        else:
            print('Wrong saving format, please specify either .npy, .txt, or .csv')
            sys.exit()