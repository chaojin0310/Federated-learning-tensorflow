import tensorflow as tf
import numpy as np
import pickle

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

        # 1
        net = tf.layers.conv2d(input_ph, filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same', name='conv1')
        net = tf.nn.relu(net, name='relu1')
        net = tf.nn.lrn(net, 4, bias=1, alpha=0.001 / 9, beta=0.75, name='lrn1')
        net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1')

        # 2
        net = tf.layers.conv2d(net, filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', name='conv2')
        net = tf.nn.relu(net, name='relu2')
        net = tf.nn.lrn(net, 4, bias=1, alpha=0.001 / 9, beta=0.75, name='lrn2')
        net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')

        # 3
        net = tf.layers.conv2d(net, filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3')
        net = tf.nn.relu(net, name='relu3')

        # 4
        net = tf.layers.conv2d(net, filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv4')
        net = tf.nn.relu(net, name='relu4')

        # 5
        net = tf.layers.conv2d(net, filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv5')
        net = tf.nn.relu(net, name='relu5')
        # net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool5')

        # 6
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 4096, name='fc1')
        net = tf.nn.relu(net, name='relu6')
        net = tf.layers.dropout(net, 0.5, training=self.is_training, name='dropout1')

        # 7
        net = tf.layers.dense(net, 4096, name='fc2')
        net = tf.nn.relu(net, name='relu7')
        net = tf.layers.dropout(net, 0.5, training=self.is_training, name='dropout2')

        # 8
        logits = tf.layers.dense(net, self.num_classes, name='fc3')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph, logits=logits)
        predictions = tf.argmax(logits, axis=1)
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
