import tensorflow as tf
import numpy as np

from model import Model
from baseline_constants import ACCURACY_KEY
from utils.model_utils import batch_data


# IMAGE_SIZE = 224
IMAGE_SIZE = 32


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr, tf.train.AdamOptimizer(learning_rate=lr))

    def create_model(self):
        input_ph = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
        label_ph = tf.placeholder(tf.int64, shape=(None,))
        self.is_training = tf.placeholder(tf.bool)

        def conv2d(inputs, output_channels, kernel_size, stride, activation=True, name=''):
            net = tf.layers.conv2d(inputs,
                                   filters=output_channels,
                                   kernel_size=(kernel_size, kernel_size),
                                   strides=(stride, stride),
                                   padding='same',
                                   name=name)
            net = tf.layers.batch_normalization(net, training=self.is_training, name=name+'_bn')
            if activation:
                net = tf.nn.relu(net, name=name+'_relu')
            return net
        
        def separable_conv2d(inputs, output_channels, kernel_size, stride, activation=True, name=''):
            net = tf.layers.separable_conv2d(inputs,
                                             filters=output_channels,
                                             kernel_size=(kernel_size, kernel_size),
                                             strides=(stride, stride),
                                             padding='same',
                                             depth_multiplier=1,
                                             name=name)
            net = tf.layers.batch_normalization(net, training=self.is_training, name=name+'_bn')
            if activation:
                net = tf.nn.relu(net, name=name+'_relu')
            return net

        def maxpooling2d(inputs, kernel_size, stride, name=''):
            net = tf.layers.max_pooling2d(inputs,
                                          pool_size=(kernel_size, kernel_size),
                                          strides=(stride, stride),
                                          padding='same',
                                          name=name)
            return net
        
        # Entry Flow
        # block1
        net = conv2d(input_ph, output_channels=32, kernel_size=3, stride=2, activation=True, name='block1_conv1')
        net = conv2d(net, output_channels=64, kernel_size=3, stride=1, activation=True, name='block1_conv2')
        residual = conv2d(net, output_channels=128, kernel_size=1, stride=2, activation=False, name='block1_res')
        # block2
        net = separable_conv2d(net, output_channels=128, kernel_size=3, stride=1, activation=True, name='block2_dws_conv1')
        net = separable_conv2d(net, output_channels=128, kernel_size=3, stride=1, activation=False, name='block2_dws_conv2')
        net = maxpooling2d(net, kernel_size=3, stride=2, name='block2_max_pool')
        net = tf.add(net, residual, name='block2_add')
        residual = conv2d(net, output_channels=256, kernel_size=1, stride=2, activation=False, name='block2_res')
        # block3
        net = tf.nn.relu(net, name='block3_relu1')
        net = separable_conv2d(net, output_channels=256, kernel_size=3, stride=1, activation=True, name='block3_dws_conv1')
        net = separable_conv2d(net, output_channels=256, kernel_size=3, stride=1, activation=False, name='block3_dws_conv2')
        net = maxpooling2d(net, kernel_size=3, stride=2, name='block3_max_pool')
        net = tf.add(net, residual, name='block3_add')
        residual = conv2d(net, output_channels=728, kernel_size=1, stride=2, activation=False, name='block3_res')
        # block4
        net = tf.nn.relu(net, name='block4_relu1')
        net = separable_conv2d(net, output_channels=728, kernel_size=3, stride=1, activation=True, name='block4_dws_conv1')
        net = separable_conv2d(net, output_channels=728, kernel_size=3, stride=1, activation=False, name='block4_dws_conv2')
        net = maxpooling2d(net, kernel_size=3, stride=2, name='block4_max_pool')
        net = tf.add(net, residual, name='block4_add')
        # Middle Flow
        # block5 to block12
        for i in range(8):
            block_prefix = 'block%s_' % (str(i + 5))
            residual = net
            net = tf.nn.relu(net, name=block_prefix+'relu1')
            net = separable_conv2d(net, output_channels=728, kernel_size=3, stride=1, activation=True, name=block_prefix+'dws_conv1')
            net = separable_conv2d(net, output_channels=728, kernel_size=3, stride=1, activation=True, name=block_prefix+'dws_conv2')
            net = separable_conv2d(net, output_channels=728, kernel_size=3, stride=1, activation=False, name=block_prefix+'dws_conv3')
            net = tf.add(net, residual, name=block_prefix+'add')
        # Exit Flow
        residual = conv2d(net, output_channels=1024, kernel_size=1, stride=2, activation=False, name='block12_res')
        # block13
        net = tf.nn.relu(net, name='block13_relu1')
        net = separable_conv2d(net, output_channels=728, kernel_size=3, stride=1, activation=True, name='block13_dws_conv1')
        net = separable_conv2d(net, output_channels=1024, kernel_size=3, stride=1, activation=False, name='block13_dws_conv2')
        net = maxpooling2d(net, kernel_size=3, stride=2, name='block13_max_pool')
        net = tf.add(net, residual, name='block13_add')
        # block14
        net = separable_conv2d(net, output_channels=1536, kernel_size=3, stride=1, activation=True, name='block14_dws_conv1')
        net = separable_conv2d(net, output_channels=2048, kernel_size=3, stride=1, activation=True, name='block14_dws_conv2')

        net_shape = net.get_shape().as_list()
        pooling_size = [net_shape[1], net_shape[2]]
        net = tf.layers.average_pooling2d(net, pooling_size, strides=(1, 1), padding='valid', name='avg_pool')

        net = tf.layers.flatten(net)
        # net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name='dropout')
        logits = tf.layers.dense(net, self.num_classes, name='fc')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph, logits=logits)
        predictions = tf.argmax(logits, axis=1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # ensure that update_ops executes before train_op
            train_op = self.optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        # train_op = self.optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
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
