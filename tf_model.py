# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from enum import Enum

def load_data(data_path="MNIST_data/", one_hot=True):
    dataset = input_data.read_data_sets(data_path, one_hot=one_hot)
    return dataset

# input_size = 784
# num_classes = 10

layer_type = Enum('layer_type', 'FC CNN LSTM')
optimizer_type = Enum('optimizer_type', 'SGD Adam RMSProp')


class Model:

    def __init__(self, sess, model_params, train_params):
        self.sess = sess
        self.model_params = model_params
        self.train_params = train_params


    # def build(self, num_layers, layer_type_list, output_size_list, use_pooling, use_dropout, keep_prob):
    def build(self):
        with tf.variable_scope(self.model_params.name):
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, self.model_params.input_size])
            self.input = self.X
            self.Y = tf.placeholder(tf.float32, [None, self.model_params.num_classes])
            self.training = tf.placeholder(tf.bool)

            for i in range(self.model_params.num_layers):
                # Fully-connected Layer
                if self.model_params.layer_type_list[i] == layer_type.FC.value:
                    print("Hidden layer #%d: Adding FC layer..." % (i + 1))
                    if i == 0:
                        self.input = self.X
                    elif self.model_params.layer_type_list[i - 1] == layer_type.FC.value:
                        self.input = out
                    elif self.model_params.layer_type_list[i-1] == layer_type.CNN.value:
                        flatten_size = int(out.shape[1]*out.shape[2]*out.shape[3])
                        self.input = tf.reshape(out, [-1, flatten_size])
                    else:
                        pass

                    out = tf.layers.dense(inputs=self.input,
                                     units=self.model_params.output_size_list[i], activation=tf.nn.relu)
                    # Dropout
                    if self.model_params.use_dropout[i]:
                        out = tf.layers.dropout(inputs=out, rate=self.model_params.keep_prob[i], training=self.training)

                # Convolutional Layer
                elif self.model_params.layer_type_list[i] == layer_type.CNN.value:
                    print("Hidden layer #%d: Adding CNN layer..." % (i+1))
                    if i == 0:
                        self.input = tf.reshape(self.X, [-1, 28, 28, 1])
                    else:
                        self.input = out

                    out = tf.layers.conv2d(inputs=self.input, filters=self.model_params.output_size_list[i], kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
                    # Pooling Layer #1
                    if self.model_params.use_pooling[i]:
                        out = tf.layers.max_pooling2d(inputs=out, pool_size=[2, 2],
                                                padding="SAME", strides=2)
                    # Dropout
                    if self.model_params.use_dropout[i]:
                        out = tf.layers.dropout(inputs=out, rate=self.model_params.keep_prob[i], training=self.training)

            # Output (no activation) Layer
            print("Output layer: Adding FC layer...")
            if self.model_params.layer_type_list[i] == layer_type.FC.value:
                self.input = out
            elif self.model_params.layer_type_list[i] == layer_type.CNN.value:
                flatten_size = int(out.shape[1] * out.shape[2] * out.shape[3])
                self.input = tf.reshape(out, [-1, flatten_size])
            else:
                pass

            self.logits = tf.layers.dense(inputs=self.input, units=self.model_params.num_classes)
        return True

    def set_optimizer(self, optimizer):
        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))

        if optimizer == optimizer_type.SGD.value:
            print("SGD optimizer is selected.")
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.train_params.learning_rate).minimize(self.cost)
        elif optimizer == optimizer_type.Adam.value:
            print("Adam optimizer is selected.")
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.train_params.learning_rate).minimize(self.cost)
        elif optimizer == optimizer_type.RMSProp.value:
            print("RMSProp optimizer is selected.")
            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.train_params.learning_rate).minimize(self.cost)
        else:
            pass

    def predict(self, image, training=False):
        image = np.asarray(image, dtype="float32")
        # normalize image 0 to 1
        if image.max() == 255:
            image = image / 255

        image = 1.0 - image
        image = image.reshape((1, self.model_params.input_size))
        prediction = tf.argmax(tf.nn.softmax(self.logits), 1)
        return self.sess.run(prediction,
                      feed_dict={self.X: image, self.training: training})

    def evaluate(self, test_data_set, training=False):
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return self.sess.run(self.accuracy,
                             feed_dict={self.X: test_data_set.images, self.Y: test_data_set.labels, self.training: training})

    def train(self, train_data_set, training=True):
        # initialize variables
        self.sess.run(tf.global_variables_initializer())
        print('Training started.')

        # train my model
        for epoch in range(self.train_params.training_epochs):
            avg_cost = 0
            total_batch = int(train_data_set.num_examples / self.train_params.batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = train_data_set.next_batch(self.train_params.batch_size)
                c, _ = self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: batch_xs, self.Y: batch_ys, self.training: training})
                avg_cost += c / total_batch

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        print('Training finished.')

        # save session
        self.saver = tf.train.Saver()
        self.restore_dir = self.train_params.train_dir + self.model_params.name
        tf.gfile.MakeDirs(self.restore_dir)
        self.saver.save(self.sess, self.restore_dir + "model")
        print('Model saved to', self.restore_dir)



