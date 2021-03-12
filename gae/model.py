from gae.layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, Discriminator
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp 

tfkl = tf.keras.layers
tfd = tfp.distributions


# flags = tf.app.flags
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


def shuffle(a):
    permuted_rows = []
    for i in range(a.get_shape()[1]):
        permuted_rows.append(tf.random_shuffle(a[:, i]))
    return tf.stack(permuted_rows, axis=1)


def random_choice(a, num_samples=1, axis=0):
    indices = tf.random.uniform([num_samples], minval=1, maxval=a.shape[axis], dtype=tf.int32)
    return tf.gather(a, indices, axis=axis)


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero,  decoder='linear', **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.decoder = decoder
        self.discriminator_sample_size = 64         
        self.build()


    def _build(self):
      self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)
      if self.decoder  == 'linear':
        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                      output_dim=FLAGS.hidden2,
                                      name='z_',
                                      adj=self.adj,
                                      act=lambda x: x,
                                      dropout=self.dropout,
                                      logging=self.logging)(self.hidden1)
        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          name='std_',
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)
        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)
        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.z)

      if self.decoder == 'mixture':
          # q(y|x)
        self.qy_logit = GraphConvolution(
                                    name='qy_',
                                    input_dim=FLAGS.hidden1,
                                    output_dim=FLAGS.auxiliary_pred_dim,
                                    adj=self.adj,
                                    act=lambda x: x,
                                    dropout=self.dropout,
                                    logging=self.logging)(self.hidden1)
          # p(y|x)
        self.qy = tf.nn.softmax(self.qy_logit)
        # q(z|x,z)
        means = []
        stds = []
        for k in range(FLAGS.auxiliary_pred_dim):
          # (batch, category, z-dim) dimensions
          means.append(GraphConvolution(name='{}_mean'.format(k),
                                              input_dim=FLAGS.hidden1,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.hidden1))
          stds.append(GraphConvolution(name='{}_std'.format(k),
                                                  input_dim=FLAGS.hidden1,
                                                  output_dim=FLAGS.hidden2,
                                                  adj=self.adj,
                                                  act=lambda x: x,
                                                  dropout=self.dropout,
                                                  logging=self.logging)(self.hidden1))
        self.z_mean = tf.reshape(
            tf.stack(means, axis=1, name='stack_1'),
            shape=(self.input_dim, FLAGS.auxiliary_pred_dim, FLAGS.hidden2))
        self.z_log_std  = tf.reshape(
            tf.stack(stds, axis=1, name='stack_2'),
            shape=(self.input_dim, FLAGS.auxiliary_pred_dim, FLAGS.hidden2))

        self.gmm =  tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(
                        probs=self.qy),
                        components_distribution=tfd.MultivariateNormalDiag(
                            loc=self.z_mean,
                            scale_diag=tf.exp(self.z_log_std)))

        self.prior =  tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(
                        probs=tf.fill(tf.shape(self.qy), 1/FLAGS.auxiliary_pred_dim)),
                        components_distribution=tfd.MultivariateNormalDiag(
                            loc=tf.fill(tf.shape(self.z_mean), 0.0),
                            scale_diag=tf.fill(tf.shape(self.z_mean), 1.0)))

        # Total Correlation
        # self.z_sample = tf.reshape(self.gmm.sample(self.discriminator_sample_size), (self.discriminator_sample_size, self.input_dim*hidden2))
        # gmm.sample(1)  should sample the whole graph
        self.z_sample = random_choice(tf.reshape(self.gmm.sample(1), (self.input_dim, FLAGS.hidden2)), num_samples=self.discriminator_sample_size)
        self.z_sample_shuffled =  shuffle(self.z_sample)

        discriminator = Discriminator(logging=self.logging, name='discriminator')

        self.true_tc_logits = discriminator(self.z_sample)
        self.shuffled_tc_logits = discriminator(self.z_sample_shuffled)

        self.true_tc_probs = tf.nn.softmax(self.true_tc_logits)
        self.shuffled_tc_probs = tf.nn.softmax(self.shuffled_tc_logits)

        # reconstructing
        self.z = tf.reshape(self.gmm.sample(1), (self.input_dim, FLAGS.hidden2))
        # self.prio = tf.reshape(self.prior_m.sample(1), (hidden2, self.input_dim,1))
        # p(x|z,y)
        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                    act=lambda x: x,
                                                    logging=self.logging)(self.z)


