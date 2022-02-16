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

        def conv2d(inputs, output_channels, kernel_size, stride, name=''):
            net = tf.layers.conv2d(inputs,
                                   filters=output_channels,
                                   kernel_size=(kernel_size, kernel_size),
                                   strides=(stride, stride),
                                   padding='same',
                                   name=name)
            net = tf.layers.batch_normalization(net, training=self.is_training, name=name + '_bn')
            net = tf.nn.relu(net, name=name + '_relu')
            return net

        def maxpooling2d(inputs, kernel_size, stride, name=''):
            net = tf.layers.max_pooling2d(inputs,
                                          pool_size=(kernel_size, kernel_size),
                                          strides=(stride, stride),
                                          padding='same',
                                          name=name)
            return net

        # ch(x) represents number of output channels of x*x conv2d layer
        # ch(x)_red represents number of output channels of 1*1 layer before x*x layer
        def inception_module(inputs, ch1, ch3_red, ch3, ch5_red, ch5, pool_proj, name=''):
            branch1 = conv2d(inputs, ch1, 1, 1, name=name + '_branch1')

            branch2 = conv2d(inputs, ch3_red, 1, 1, name=name + '_branch2_0')
            branch2 = conv2d(branch2, ch3, 3, 1, name=name + '_branch2_1')

            branch3 = conv2d(inputs, ch5_red, 1, 1, name=name + '_branch3_0')
            branch3 = conv2d(branch3, ch5, 5, 1, name=name + '_branch3_1')

            branch4 = maxpooling2d(inputs, 3, stride=1, name=name + '_branch4_0')
            branch4 = conv2d(branch4, pool_proj, 1, 1, name=name + '_branch4_1')

            outputs = tf.concat([branch1, branch2, branch3, branch4], axis=3)
            return outputs

        def aux_classifier(inputs, name=''):
            outputs = inputs
            # outputs = tf.layers.average_pooling2d(outputs, pool_size=(5, 5), strides=(1, 1), name=name + 'pool')
            outputs = conv2d(outputs, 128, 1, 1, name=name + 'conv')
            outputs = tf.layers.flatten(outputs)
            outputs = tf.layers.dense(outputs, 1024, activation='relu', name=name + 'fc1')
            outputs = tf.layers.dense(outputs, self.num_classes, name=name + 'fc2')
            return outputs

        def aux_loss(aux_1, aux_2, labels, loss):
            aux_logits_1 = aux_classifier(aux_1, 'aux1')
            aux_logits_2 = aux_classifier(aux_2, 'aux2')
            aux_loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=aux_logits_1)
            aux_loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=aux_logits_2)
            total_loss = loss + 0.3 * (aux_loss_1 + aux_loss_2)
            return total_loss

        net = conv2d(input_ph, 64, 7, 2, name='conv2d_bn1')
        net = maxpooling2d(net, 3, 2, name='pool1')

        net = conv2d(net, 64, 1, 1, name='conv2d_bn2')
        net = conv2d(net, 192, 3, 1, name='conv2d_bn3')
        net = maxpooling2d(net, 3, 2, name='pool2')

        net = inception_module(net, 64, 96, 128, 16, 32, 32, 'inception_3a')
        net = inception_module(net, 128, 128, 192, 32, 96, 64, 'inception_3b')

        net = maxpooling2d(net, 3, 2, name='pool3')

        aux_branch1 = inception_module(net, 192, 96, 208, 16, 48, 64, 'inception_4a')
        # aux_classifier branch 1
        net = inception_module(aux_branch1, 160, 112, 224, 24, 64, 64, 'inception_4b')
        net = inception_module(net, 128, 128, 256, 24, 64, 64, 'inception_4c')
        aux_branch2 = inception_module(net, 112, 144, 288, 32, 64, 64, 'inception_4d')
        # aux_classifier branch 2
        net = inception_module(aux_branch2, 256, 160, 320, 32, 128, 128, 'inception_4e')

        net = maxpooling2d(net, 3, 2, name='pool4')

        net = inception_module(net, 256, 160, 320, 32, 128, 128, 'inception_5a')
        net = inception_module(net, 384, 192, 384, 48, 128, 128, 'inception_5b')

        net_shape = net.get_shape().as_list()
        pooling_size = [net_shape[1], net_shape[2]]
        net = tf.layers.average_pooling2d(net, pooling_size, strides=(1, 1), padding='valid', name='avg_pool')

        net = tf.layers.flatten(net)
        net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name='dropout')
        logits = tf.layers.dense(net, self.num_classes, name='fc')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph, logits=logits)
        loss = tf.cond(self.is_training,
                       true_fn=lambda: aux_loss(aux_branch1, aux_branch2, label_ph, loss),
                       false_fn=lambda: loss)
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
