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
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        input_ph = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
        label_ph = tf.placeholder(tf.int64, shape=(None,))
        self.is_training = tf.placeholder(tf.bool)

        def fire(inputs, squeeze_channels, expand1_channels, expand3_channels, name):
            with tf.variable_scope(name):
                with tf.variable_scope('squeeze_layer'):
                    squeeze_layer = tf.layers.conv2d(inputs, squeeze_channels, kernel_size=(1, 1), strides=(1, 1),
                                                     padding='same')
                    squeeze_layer = tf.layers.batch_normalization(squeeze_layer, training=self.is_training)
                    squeeze_layer = tf.nn.relu(squeeze_layer)

                with tf.variable_scope('expand1_layer'):
                    expand1_layer = tf.layers.conv2d(squeeze_layer, expand1_channels, kernel_size=(1, 1), strides=(1, 1),
                                                     padding='same')
                    expand1_layer = tf.layers.batch_normalization(expand1_layer, training=self.is_training)
                    expand1_layer = tf.nn.relu(expand1_layer)
                
                with tf.variable_scope('expand3_layer'):
                    expand3_layer = tf.layers.conv2d(squeeze_layer, expand3_channels, kernel_size=(3, 3), strides=(1, 1),
                                                     padding='same')
                    expand3_layer = tf.layers.batch_normalization(expand3_layer, training=self.is_training)
                    expand3_layer = tf.nn.relu(expand3_layer)

                net = tf.concat([expand1_layer, expand3_layer], axis=3)
            return net

        net = tf.layers.conv2d(input_ph, 96, kernel_size=(7, 7), strides=(2, 2), padding='same', name='conv1')
        net = tf.layers.batch_normalization(net, training=self.is_training)
        net = tf.nn.relu(net)

        net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1')

        net = fire(net, 16, 64, 64, 'fire2')
        net = fire(net, 16, 64, 64, 'fire3')
        net = fire(net, 32, 128, 128, 'fire4')

        net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')

        net = fire(net, 32, 128, 128, 'fire5')
        net = fire(net, 48, 192, 192, 'fire6')
        net = fire(net, 48, 192, 192, 'fire7')
        net = fire(net, 64, 256, 256, 'fire8')

        net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool3')

        net = fire(net, 64, 256, 256, 'fire9')
        net = tf.layers.dropout(net, rate=0.5, training=self.is_training)

        net = tf.layers.conv2d(net, filters=self.num_classes,
                               kernel_size=(1, 1), strides=(1, 1), padding='same', name='conv10')

        net_shape = net.get_shape().as_list()
        pooling_size = [net_shape[1], net_shape[2]]
        net = tf.layers.average_pooling2d(net, pooling_size, strides=(1, 1), padding='valid')

        logits = tf.layers.flatten(net)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph, logits=logits)
        predictions = tf.argmax(logits, axis=1)
        # train_op = self.optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
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
