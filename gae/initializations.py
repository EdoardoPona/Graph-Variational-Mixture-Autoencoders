import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


kl = tfd.kullback_leibler
NUM_MC_SAMPLES = 20

@kl.RegisterKL(tfd.MixtureSameFamily, tfd.MixtureSameFamily)
def _mc_kl_msf_msf(a, b, seed=None, name='_mc_kl_msf_msf'):
  with tf.name_scope(name):
    s = a.sample(NUM_MC_SAMPLES, seed)
    return tf.reduce_mean(
        a.log_prob(s) - b.log_prob(s), axis=0, name='KL_MSF_MSF')


def to_one_hot_vector(data):
  unique_values = pd.Series(data).unique()
  target = np.zeros(shape=(data.shape[0], len(unique_values)))
  for i, v in enumerate(unique_values):
    target[np.where(data == v ), i] = 1
  return target

