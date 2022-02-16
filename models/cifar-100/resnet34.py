import tensorflow as tf
import numpy as np
import os

from model import Model
from baseline_constants import ACCURACY_KEY
from utils.model_utils import batch_data


IMAGE_SIZE = 32
# IMAGE_SIZE = 224

class ClientModel(Model):

    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        input_ph = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
        label_ph = tf.placeholder(tf.int64, shape=(None,))
        self.is_training = tf.placeholder(tf.bool)

        def conv2d(inputs, filters, kernel_size, stride, activation=True, name='', bn=False):
            net = tf.layers.conv2d(inputs, filters, 
                                   kernel_size=(kernel_size, kernel_size), 
                                   strides=(stride, stride), 
                                   padding='same', name=name+'_conv2d')
            if bn:
                net = tf.layers.batch_normalization(net, training=self.is_training, name=name+'_bn')
            if activation:
                net = tf.nn.relu(net, name=name+'_relu')
            return net

        def residual_block(inputs, output_channel, block_stride=1, name='', bn=False):
            if block_stride == 2:
                shortcut_conv = conv2d(inputs, output_channel, kernel_size=1, stride=block_stride, activation=False, name=name+'shortcut', bn=bn)
            else:
                shortcut_conv = inputs
            net = conv2d(inputs, output_channel, kernel_size=3, stride=block_stride, name=name+'_1', bn=bn)
            net = conv2d(net, output_channel, kernel_size=3, stride=1, activation=False, name=name+'_2', bn=bn)
            net = tf.add(net, shortcut_conv, name=name+'_add')
            net = tf.nn.relu(net, name=name+'_relu')
            return net

        net = conv2d(input_ph, filters=64, kernel_size=3, stride=2, name='conv1')
        net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=(2, 2), padding='same', name='max_pool')

        net = residual_block(net, 64, block_stride=1, name='conv2_1')
        net = residual_block(net, 64, block_stride=1, name='conv2_2')
        net = residual_block(net, 64, block_stride=1, name='conv2_3')

        net = residual_block(net, 128, block_stride=2, name='conv3_1', bn=True)
        net = residual_block(net, 128, block_stride=1, name='conv3_2')
        net = residual_block(net, 128, block_stride=1, name='conv3_3')
        net = residual_block(net, 128, block_stride=1, name='conv3_4')

        net = residual_block(net, 256, block_stride=2, name='conv4_1', bn=True)
        net = residual_block(net, 256, block_stride=1, name='conv4_2')
        net = residual_block(net, 256, block_stride=1, name='conv4_3')
        net = residual_block(net, 256, block_stride=1, name='conv4_4')
        net = residual_block(net, 256, block_stride=1, name='conv4_5')
        net = residual_block(net, 256, block_stride=1, name='conv4_6')

        net = residual_block(net, 512, block_stride=2, name='conv5_1', bn=True)
        net = residual_block(net, 512, block_stride=1, name='conv5_2')
        net = residual_block(net, 512, block_stride=1, name='conv5_3')

        net_shape = net.get_shape().as_list()
        pooling_size = [net_shape[1], net_shape[2]]
        net = tf.layers.average_pooling2d(net, pooling_size, strides=(1, 1), padding='valid', name='avg_pool')

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
