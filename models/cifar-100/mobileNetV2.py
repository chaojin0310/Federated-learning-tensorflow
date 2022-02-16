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

        def conv2d(inputs, filters, stride, scope=""):
            with tf.variable_scope(scope):
                with tf.variable_scope("conv2d"):
                    net = tf.layers.conv2d(inputs, filters, kernel_size=(3, 3), strides=(stride, stride), padding='same')
                    net = tf.layers.batch_normalization(net, training=self.is_training)
                    net = tf.nn.relu6(net)
                return net

        def conv2d_1(inputs, filters, stride):
            with tf.variable_scope("conv2d_1"):
                net = tf.layers.conv2d(inputs, filters, kernel_size=(1, 1), strides=(stride, stride), padding='same')
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.relu6(net)
            return net

        def pointwise_expansion(inputs, expansion, stride):
            input_shape = inputs.get_shape().as_list()
            expansion_channel = expansion * input_shape[3]
            with tf.variable_scope("pointwise_expansion"):
                net = tf.layers.conv2d(
                    inputs, filters=expansion_channel, kernel_size=(1, 1), strides=(stride, stride), padding='same')
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.relu6(net)
            return net

        def depthwise_conv(inputs, filters, stride):
            with tf.variable_scope("depthwise_conv"):
                net = tf.layers.separable_conv2d(
                    inputs, filters, kernel_size=(3, 3), strides=(stride, stride), padding='same', depth_multiplier=1)
                # depth_multiplier could be modified if needed
                net = tf.layers.batch_normalization(net, training=self.is_training)
                net = tf.nn.relu6(net)
            return net

        def pointwise_conv(inputs, filters, stride):
            with tf.variable_scope("pointwise_conv"):
                net = tf.layers.conv2d(inputs, filters, kernel_size=(1, 1), strides=(stride, stride), padding='same')
                net = tf.layers.batch_normalization(net, training=self.is_training)
            return net

        def inverted_residual_block(inputs, filters, stride, expansion=6, scope=""):
            with tf.variable_scope(scope):
                net = pointwise_expansion(inputs, expansion, stride=1)
                depthwise_filters = net.get_shape().as_list()[3]
                net = depthwise_conv(net, depthwise_filters, stride)
                net = pointwise_conv(net, filters, stride=1)

                if stride == 1 and inputs.get_shape().as_list()[3] == net.get_shape().as_list()[3]:
                    net = tf.add(net, inputs)
            return net

        output_channels = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        for i in range(len(output_channels)):
            output_channels[i] = int(output_channels[i] * self.width_multiplier)

        net = input_ph
        net = conv2d(net, filters=output_channels[0], stride=2, scope="block0")

        net = inverted_residual_block(net, filters=output_channels[1], stride=1, expansion=1, scope="block1")

        net = inverted_residual_block(net, filters=output_channels[2], stride=2, scope="block2_0")
        net = inverted_residual_block(net, filters=output_channels[2], stride=1, scope="block2_1")

        net = inverted_residual_block(net, filters=output_channels[3], stride=2, scope="block3_0")
        net = inverted_residual_block(net, filters=output_channels[3], stride=1, scope="block3_1")
        net = inverted_residual_block(net, filters=output_channels[3], stride=1, scope="block3_2")

        net = inverted_residual_block(net, filters=output_channels[4], stride=2, scope="block4_0")
        net = inverted_residual_block(net, filters=output_channels[4], stride=1, scope="block4_1")
        net = inverted_residual_block(net, filters=output_channels[4], stride=1, scope="block4_2")
        net = inverted_residual_block(net, filters=output_channels[4], stride=1, scope="block4_3")

        net = inverted_residual_block(net, filters=output_channels[5], stride=1, scope="block5_0")
        net = inverted_residual_block(net, filters=output_channels[5], stride=1, scope="block5_1")
        net = inverted_residual_block(net, filters=output_channels[5], stride=1, scope="block5_2")

        net = inverted_residual_block(net, filters=output_channels[6], stride=2, scope="block6_0")
        net = inverted_residual_block(net, filters=output_channels[6], stride=1, scope="block6_1")
        net = inverted_residual_block(net, filters=output_channels[6], stride=1, scope="block6_2")

        net = inverted_residual_block(net, filters=output_channels[7], stride=1, scope="block7_0")

        net = conv2d_1(net, filters=output_channels[8], stride=1)

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

