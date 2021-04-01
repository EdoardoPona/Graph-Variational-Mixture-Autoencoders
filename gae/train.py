from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerVAE
from gae.input_data import load_graph, load_regions, load_disease_network, load_disease_network_types
from gae.model import GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges


tf.disable_eager_execution()

import os 
print(os.listdir('.'))


# Settings
# flags = tf.app.flags
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-2, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

# TODO use this flag to swtich between gaussian and mixture and factor 
flags.DEFINE_string('model', 'gcn_vae', 'Model string.')   
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model

WORKING_PATH = 'gae/data/'


YEAR = 2014

# adj = load_graph(WORKING_PATH, YEAR)[0:100, 0:100]
adj = load_disease_network() 
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()


adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

adj_norm = preprocess_graph(adj)


# target = np.ones((100, 6))
# target = load_regions(WORKING_PATH, YEAR, one_hot=True)[:100]
target = load_disease_network_types(one_hot=True) 
flags.DEFINE_integer('auxiliary_pred_dim', target.shape[1], 'Number of dimensions for auxiliary prediction')

if FLAGS.features == 0:
    features = sp.identity(adj.shape[0])  # featureless
else: 
    # features = sp.diags(load_regions(WORKING_PATH, YEAR, one_hot=False)[:100])
    features = sp.diags(load_disease_network_types(one_hot=False))

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, decoder='mixture')

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


# Optimizer
with tf.name_scope('optimizer'):
    opt = OptimizerVAE(preds=model.reconstructions,
                       labels=tf.reshape(
                           tf.sparse_tensor_to_dense(
                               placeholders['adj_orig'],
                               validate_indices=False), [-1]),
                       model=model,
                       num_nodes=num_nodes,
                       auxiliary_labels=False,
                       target=tf.reshape(
                           tf.constant(target),
                           (-1, FLAGS.auxiliary_pred_dim)),
                       pos_weight=pos_weight,
                       norm=norm)


def get_embeddings():
	feed_dict.update({placeholders['dropout']: 0})
	emb = sess.run(model.z_mean, feed_dict=feed_dict)
	return emb

def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score



# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []

cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

epochs = 100

# Train model
for epoch in range(epochs + 1):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Run single weight update
    # outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.loss, opt.kl, opt.classification_loss], feed_dict=feed_dict)
    outs = sess.run([opt.vae_opt_op,
                     opt.vae_loss,
                     opt.discriminator_opt_op,
                     opt.discriminator_loss,
                     opt.accuracy,
                     opt.reconstruction_loss,
                     opt.kl,
                     opt.classification_loss], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[4]

    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)

    if epoch % 10 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
            "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
            "val_ap=", "{:.5f}".format(ap_curr),
            "reconstruction loss=","{:.5f}".format(outs[5]),
            "kl loss=","{:.5f}".format(outs[6]),
            "classification loss=","{:.5f}".format(outs[7]),
            "time=", "{:.5f}".format(time.time() - t))

    if epoch == epochs:
      feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
      feed_dict.update({placeholders['dropout']: FLAGS.dropout})

      outs = sess.run([model.z_mean, model.z_log_std, model.z, model.qy_logit ], feed_dict=feed_dict)

print("Optimization Finished!")
roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))

