import tensorflow.compat.v1 as tf

# flags = tf.app.flags
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, target=None, auxiliary_labels=None, vae_scope='vae', discriminator_scope='discriminator'):
        preds_sub = preds
        labels_sub = labels       
        self.target = target 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        # self.discriminator_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=discriminator_scope)
        self.discriminator_variables = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if discriminator_scope in var.name]
        self.vae_variables = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if var not in self.discriminator_variables] 
        # TODO this is a temporary hack. model and discriminators shoud have separate scopes 
        # TODO should have separate models for encoder, decoder, discriminator (not layers)
        # TOOD each of those should have its own scope 
        
        # Gaussian
        if auxiliary_labels == None:
            # reconstruction loss
            self.reconstruction_loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, 
                                                                                                        targets=labels_sub, pos_weight=pos_weight))
            self.kl = (0.5 / num_nodes) * tf.reduce_mean(
                tf.reduce_sum(1 + 
                                2 * model.z_log_std - 
                                tf.square(model.z_mean) -
                                tf.square(tf.exp(model.z_log_std)), 
                                1))        
            # self.vae_loss = self.reconstruction_loss - beta * self.kl      beta loss 
            self.vae_loss = self.reconstruction_loss + self.kl 

        # Mixed gaussian
        if auxiliary_labels != None: 
            self.reconstruction_loss = norm * tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(
                    logits=preds_sub, 
                    targets=labels_sub, 
                    pos_weight=pos_weight))
            
    
            self.kl = (0.5 / num_nodes) * tf.reduce_mean(model.gmm.kl_divergence(model.prior))
            self.classification_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=model.qy_logit, labels=target))

            ## self.vae_loss = self.loss + self.classification_loss + beta*self.kl  beta loss 
            self.vae_loss = self.reconstruction_loss + self.classification_loss + self.kl            

        self.total_correlation = tf.log(tf.sigmoid(model.true_tc_logits) / (1 - tf.sigmoid(model.true_tc_logits)))
        self.factor_vae_loss = self.vae_loss + self.total_correlation

        self.discriminator_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(model.true_tc_logits), logits=model.true_tc_logits) +  \
                                  tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(model.shuffled_tc_logits), logits=model.shuffled_tc_logits)
            
        self.vae_grads_and_vars = self.optimizer.compute_gradients(self.factor_vae_loss, var_list=self.vae_variables)  
        self.vae_opt_op = self.optimizer.apply_gradients(self.vae_grads_and_vars)

        self.discriminator_grads_and_vars = self.optimizer.compute_gradients(self.discriminator_loss, var_list=self.discriminator_variables)
        self.discriminator_opt_op = self.optimizer.apply_gradients(self.discriminator_grads_and_vars)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))        
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

