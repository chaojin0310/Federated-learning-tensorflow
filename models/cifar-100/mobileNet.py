import tensorflow as tf
import numpy as np
import os

from model import Model
from baseline_constants import ACCURACY_KEY
from utils.model_utils import batch_data


IMAGE_SIZE = 32
# IMAGE_SIZE = 224

class ClientModel(Model):

    def __init__(self, seed, lr, num_classes, width_multiplier=1):
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        # width_multiplier could be modified in baseline_constants.py
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        input_ph = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
        label_ph = tf.placeholder(tf.int64, shape=(None,))
        self.is_training = tf.placeholder(tf.bool)

        def conv_block(inputs, filters, strides, name, bn=True):
            net = tf.layers.conv2d(inputs, filters, kernel_size=(3, 3), strides=(strides, strides), padding='same', name=name+'_conv2d')
            if bn:
                net = tf.layers.batch_normalization(net, training=self.is_training, name=name + '_bn')
            net = tf.nn.relu(net, name=name+'_relu')
            return net

        def separable_conv_block(inputs, input_channel, output_channel, strides, name, bn=True):
            net = tf.layers.separable_conv2d(
                inputs, input_channel, kernel_size=(3, 3), strides=(strides, strides), padding='same', depth_multiplier=1, name=name+'_depthwise')
            if bn:
                net = tf.layers.batch_normalization(net, training=self.is_training, name=name+'_bn1')
            net = tf.nn.relu(net, name=name+'_relu1')
            net = tf.layers.conv2d(net, output_channel, kernel_size=(1, 1), strides=(1, 1), padding='same', name=name+'_pointwise')
            if bn:
                net = tf.layers.batch_normalization(net, training=self.is_training, name=name+'_bn2')
            net = tf.nn.relu(net, name=name+'_relu2')
            return net

        net = conv_block(input_ph, 32, 2, 'conv1')
        
        net = separable_conv_block(net, 32, 64, 1, 'separable1')
        net = separable_conv_block(net, 64, 128, 2, 'separable2')
        net = separable_conv_block(net, 128, 128, 1, 'separable3')
        net = separable_conv_block(net, 128, 256, 2, 'separable4')
        net = separable_conv_block(net, 256, 256, 1, 'separable5')
        net = separable_conv_block(net, 256, 512, 2, 'separable6')
        net = separable_conv_block(net, 512, 512, 1, 'separable7')
        net = separable_conv_block(net, 512, 512, 1, 'separable8')
        net = separable_conv_block(net, 512, 512, 1, 'separable9')
        net = separable_conv_block(net, 512, 512, 1, 'separable10')
        net = separable_conv_block(net, 512, 512, 1, 'separable11')
        net = separable_conv_block(net, 512, 1024, 2, 'separable12')
        net = separable_conv_block(net, 1024, 1024, 1, 'separable13')

        net_shape = net.get_shape().as_list()
        pooling_size = [net_shape[1], net_shape[2]]
        net = tf.layers.average_pooling2d(net, pooling_size, strides=(1, 1), padding='valid')

        net = tf.layers.flatten(net)  # change dim to 1
        logits = tf.layers.dense(net, self.num_classes)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph, logits=logits)
        predictions = tf.argmax(logits, axis=1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # ensure that update_ops executes before train_op
            train_op = self.optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        eval_metric_ops = tf.count_nonzero(tf.equal(label_ph, predictions))

        return input_ph, label_ph, train_op, eval_metric_ops, tf.math.reduce_mean(loss)

    def process_x(self, raw_x_batch):
        x_batch = np.array(raw_x_batch, dtype=np.float32)
        batch_size = x_batch.size // (IMAGE_SIZE * IMAGE_SIZE * 3)
        x_batch = x_batch.reshape((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)).transpose(0, 2, 3, 1)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = np.array(raw_y_batch, dtype=np.int64)
        return y_batch

    def run_epoch(self, data, batch_size):
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            with self.graph.as_default():
                self.sess.run(self.train_op,
                              feed_dict={
                                  self.features: input_data,
                                  self.labels: target_data,
                                  self.is_training: True
                              })

    def test(self, data):
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        with self.graph.as_default():
            tot_acc, loss = self.sess.run(
                [self.eval_metric_ops, self.loss],
                feed_dict={self.features: x_vecs, self.labels: labels, self.is_training: False}
            )
        acc = float(tot_acc) / x_vecs.shape[0]
        return {ACCURACY_KEY: acc, 'loss': loss}

