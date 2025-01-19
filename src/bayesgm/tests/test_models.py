import pytest
import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from bayesgm.models.causalbgm import CausalBGM

params = {'dataset': 'Sim_Hirano_Imbens',
        'output_dir': '.',
        'save_res': True,
        'save_model': False,
        'binary_treatment': False,
        'use_bnn': True,
        'z_dims': [1, 1, 1, 7],
        'v_dim': 200,
        'lr_theta': 0.0001,
        'lr_z': 0.0001,
        'x_min': 0,
        'x_max': 3,
        'g_units': [64, 64, 64, 64, 64],
        'f_units': [64, 32, 8],
        'h_units': [64, 32, 8],
        'kl_weight': 0.0001,
        'lr': 0.0002,
        'g_d_freq': 5,
        'use_z_rec': True,
        'e_units': [64, 64, 64, 64, 64],
        'dz_units': [64, 32, 8]}

def test_causalbgm(num_samples=100):
    num_features = params['v_dim']
    model = CausalBGM(params=params, random_seed=None)
    model.initialize_nets(print_summary = True)
    assert model.get_config()['params']['dataset'] == 'Sim_Hirano_Imbens'

    np.random.seed(42)
    x = np.random.uniform(-1, 1, size=(num_samples, 1))
    v = np.random.normal(0, 1, size=(num_samples, num_features))
    y = 2 * x + np.sum(v, axis=1, keepdims=True) + np.random.normal(0, 0.1, size=(num_samples, 1))
    x = x.astype('float32')
    y = y.astype('float32')
    v = v.astype('float32')
    # Run the egm_init method
    try:
        model.egm_init(data=(x, y, v), n_iter=100, batches_per_eval=10, verbose=1)
    except Exception as e:
        pytest.fail(f"egm_init raised an exception: {e}")

    # Assertions to validate expected behavior
    assert model.params == params, "Model parameters should match the input parameters."

    # Check if the model is able to predict or evaluate after initialization (if applicable)
    if hasattr(model, "fit"):
        model.fit(data=(x,y,v), epochs=5, epochs_per_eval=2, verbose=1)

    # Example of testing evaluation metrics (if supported)
    if hasattr(model, "evaluate"):
        causal_pre, mse_x, mse_y, mse_v = model.evaluate(data=(x, y, v), data_z=None, nb_intervals=200)

        # Assert that causal_pre is a NumPy array
        assert tf.is_tensor(causal_pre), "causal_pre should be a TensorFlow tensor."
        causal_pre = causal_pre.numpy()
        mse_x = mse_x.numpy()
        mse_y = mse_y.numpy()
        mse_v = mse_v.numpy()
        print('test',type(mse_x),mse_x)
        assert isinstance(causal_pre, np.ndarray), "causal_pre should be a NumPy array."
        
        # Assert that mse_x, mse_y, and mse_v are scalars
        assert isinstance(mse_x, np.float32), "mse_x should be a float scalar."
        assert isinstance(mse_x, np.float32), "mse_y should be a float scalar."
        assert isinstance(mse_x, np.float32), "mse_v should be a float scalar."

    if hasattr(model, "predict"):
        causal_pre, pos_intervals = model.predict(data=(x,y,v), alpha=0.01, n_mcmc=10, x_values=1.0, q_sd=1.0)
        causal_pre, pos_intervals = model.predict(data=(x,y,v), alpha=0.01, n_mcmc=10, x_values=[1.0, 2.0], q_sd=1.0)

        # Assert that causal_pre is a NumPy array
        assert isinstance(causal_pre, np.ndarray), "causal_pre should be a NumPy array."
        
        # Assert that pos_intervals is a NumPy array
        assert isinstance(pos_intervals, np.ndarray), "pos_intervals should be a NumPy array."






