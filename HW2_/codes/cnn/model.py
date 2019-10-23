# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class Model:
    def __init__(self,
                 learning_rate=3e-4,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.y_ = tf.placeholder(tf.int32, [None])

        # TODO:  fill the blank of the arguments
        self.loss, self.pred, self.acc = self.forward(is_train=True, reuse=False)
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_train=False, reuse=True)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        update_params = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_params):
        # TODO:  maybe you need to update the parameter of batch_normalization?
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=None):
    
        with tf.variable_scope("model", reuse=reuse):
            # TODO: implement input -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Conv Layer
            temp = tf.layers.conv2d(inputs=self.x_, filters=32, kernel_size=8)
            # Your BN Layer: use batch_normalization_layer function
            temp = batch_normalization_layer(incoming=temp, is_train=is_train)
            # Your Relu Layer
            temp = tf.nn.relu(features=temp)
            # Your Dropout Layer: use dropout_layer function
            temp = dropout_layer(incoming=temp, drop_rate=0.2, is_train=is_train)
            # Your MaxPool
            temp = tf.layers.max_pooling2d(inputs=temp, pool_size=4, strides=1)
            # Your Conv Layer
            temp = tf.layers.conv2d(inputs=temp, filters=32, kernel_size=8)
            # Your BN Layer: use batch_normalization_layer function
            temp = batch_normalization_layer(incoming=temp, is_train=is_train)
            # Your Relu Layer
            temp = tf.nn.relu(features=temp)
            # Your Dropout Layer: use dropout_layer function
            temp = dropout_layer(incoming=temp, drop_rate=0.2, is_train=is_train)
            # Your MaxPool
            temp = tf.layers.max_pooling2d(inputs=temp, pool_size=4, strides=1)
            # Your Linear Layer
            temp = tf.reshape(tensor=temp, shape=[-1, 12 * 12 * 32])
            logits = tf.layers.dense(inputs=temp, units=10)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        
        return loss, pred, acc


def batch_normalization_layer(incoming, is_train):
    # TODO: implement the batch normalization function and applied it on fully-connected layers
    # NOTE:  If isTrain is True, you should use mu and sigma calculated based on mini-batch
    #       If isTrain is False, you must use mu and sigma estimated from training data
    return tf.layers.batch_normalization(inputs=incoming, training=is_train)


    
def dropout_layer(incoming, drop_rate, is_train):
    # TODO: implement the dropout function and applied it on fully-connected layers
    # Note: When drop_rate=0, it means drop no values
    #       If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
    #       If isTrain is False, remain all values not changed
    return tf.layers.dropout(inputs=incoming, rate=drop_rate, training=is_train)

