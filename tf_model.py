# DNN model builder using TensorFlow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from enum import Enum


def load_data(data_path="MNIST_data/", one_hot=True):
    dataset = input_data.read_data_sets(data_path, one_hot=one_hot)
    return dataset


def load_batch_data(dataset, batch_size):
    batch_xs, batch_ys = dataset.next_batch(batch_size)
    return batch_xs, batch_ys

# input_size = 784
# num_classes = 10

# enum parameters
layer_type = Enum('layer_type', 'FC CNN')
activate_fn_type = Enum('activate_fn_type', 'No_act Sigmoid tanh ReLU')
init_fn_type = Enum('init_fn_type', 'No_init Normal Xavier')
loss_fn_type = Enum('loss_fn_type', 'Cross_Entropy')
optimizer_type = Enum('optimizer_type', 'SGD Adam RMSProp')


class TFModel:

    def __init__(self, model_params):
        # TF session
        self.sess = tf.Session()
        self.model_params = model_params


    def build(self):
        with tf.variable_scope(self.model_params.name):
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, self.model_params.input_size])
            self.input = self.X
            self.Y = tf.placeholder(tf.float32, [None, self.model_params.num_classes])
            self.training = tf.placeholder(tf.bool)

            num_layers = self.model_params.num_layers
            for i in range(num_layers):
                # Activation function
                if self.model_params.activate_fn[i] == activate_fn_type.No_act.value:
                    act_fn = None
                elif self.model_params.activate_fn[i] == activate_fn_type.Sigmoid.value:
                    act_fn = tf.nn.sigmoid
                elif self.model_params.activate_fn[i] == activate_fn_type.tanh.value:
                    act_fn = tf.nn.tanh
                elif self.model_params.activate_fn[i] == activate_fn_type.ReLU.value:
                    act_fn = tf.nn.relu
                else:
                    pass

                # Initializer
                if self.model_params.init_fn[i] == init_fn_type.No_init.value:
                    init_fn = None
                elif self.model_params.init_fn[i] == init_fn_type.Normal.value:
                    init_fn = tf.random_normal_initializer()
                elif self.model_params.init_fn[i] == init_fn_type.Xavier.value:
                    init_fn = tf.contrib.layers.xavier_initializer()
                else:
                    pass

                # Fully-connected Layer
                if self.model_params.layer_type[i] == layer_type.FC.value:
                    print("Hidden layer #%d: " % (i + 1))
                    if i == 0:
                        self.input = self.X
                    elif self.model_params.layer_type[i - 1] == layer_type.FC.value:
                        self.input = out
                    elif self.model_params.layer_type[i-1] == layer_type.CNN.value:
                        flatten_size = int(out.shape[1]*out.shape[2]*out.shape[3])
                        self.input = tf.reshape(out, [-1, flatten_size])
                    else:
                        pass

                    num_neurons = self.model_params.output_size[i]
                    out = tf.layers.dense(inputs=self.input,
                                          units=num_neurons,
                                          activation=act_fn,
                                          kernel_initializer=init_fn)

                    print("\tAdding FC layer...")
                    print("\t\t# of neurons = %d" % num_neurons)
                    print("\t\tInput:", self.input.shape, "-> Output:", out.shape)



                # Convolutional Layer
                elif self.model_params.layer_type[i] == layer_type.CNN.value:
                    print("Hidden layer #%d:" % (i+1))
                    if i == 0:
                        self.input = tf.reshape(self.X, [-1, 28, 28, 1])
                    else:
                        self.input = out

                    num_filters = self.model_params.output_size[i]
                    k_size = self.model_params.kernel_size[i]
                    s_size = self.model_params.kernel_stride[i]

                    out = tf.layers.conv2d(inputs=self.input,
                                           filters=num_filters,
                                           kernel_size=[k_size, k_size],
                                           strides=(s_size, s_size),
                                           padding="SAME",
                                           activation=act_fn,
                                           kernel_initializer=init_fn)

                    print("\tAdding Conv2D layer...")
                    print("\t\tKernel size = %dx%d, Stride = (%d, %d)" %(k_size, k_size, s_size, s_size))
                    print("\t\tInput:", self.input.shape, "-> Output:", out.shape)

                    # Pooling Layer
                    self.input = out
                    p_size = self.model_params.pool_size[i]
                    pool_s_size = self.model_params.pool_stride[i]

                    if self.model_params.use_pooling[i]:
                        out = tf.layers.max_pooling2d(inputs=self.input,
                                                      pool_size=[p_size, p_size],
                                                      padding="SAME",
                                                      strides=pool_s_size)

                        print("\tAdding MaxPooling layer...")
                        print("\t\tKernel size = %dx%d, Stride = (%d, %d)" %(p_size, p_size, pool_s_size, pool_s_size))
                        print("\t\tInput:", self.input.shape, "-> Output:", out.shape)

                # Dropout
                keep_prob = self.model_params.keep_prob[i]
                if self.model_params.use_dropout[i]:
                    out = tf.layers.dropout(inputs=out,
                                            rate=keep_prob,
                                            training=self.training)

                    print("\tAdding Dropout layer...")
                    print("\t\tkeep_prob = %0.1f" % keep_prob)

            # Output (no activation) Layer
            print("Output layer: ")

            if self.model_params.layer_type[i] == layer_type.FC.value:
                self.input = out
            elif self.model_params.layer_type[i] == layer_type.CNN.value:
                flatten_size = int(out.shape[1] * out.shape[2] * out.shape[3])
                self.input = tf.reshape(out, [-1, flatten_size])
            else:
                pass

            self.logits = tf.layers.dense(inputs=self.input,
                                          units=self.model_params.num_classes,
                                          activation=None,
                                          kernel_initializer=init_fn)
            print("\tAdding FC layer...")
            print("\t\tInput:", self.input.shape, "-> Output:", self.logits.shape)

        return True

    def set_optimizer(self, lossFn, optimizer, learning_rate):
        # define cost/loss & optimizer
        if lossFn == loss_fn_type.Cross_Entropy.value:
            print("\nLoss function: Cross_Entropy.")
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y))

        if optimizer == optimizer_type.SGD.value:
            print("\nSGD optimizer is selected.")
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate).minimize(self.cost)
        elif optimizer == optimizer_type.Adam.value:
            print("\nAdam optimizer is selected.")
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.cost)
        elif optimizer == optimizer_type.RMSProp.value:
            print("\nRMSProp optimizer is selected.")
            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate).minimize(self.cost)
        else:
            pass

        # initialize variables
        self.sess.run(tf.global_variables_initializer())
        print("\nGlobal variable initialized.")

    def predict(self, image, training=False):
        image = np.asarray(image, dtype="float32")
        # normalize image 0 to 1
        if image.max() == 255:
            image = image / 255

        image = 1.0 - image
        image = image.reshape((1, self.model_params.input_size))
        score = tf.nn.softmax(self.logits)
        prediction = tf.argmax(score, 1)[0]
        return self.sess.run([score, prediction],
                             feed_dict={self.X: image, self.training: training})

    def evaluate(self, test_data_set, training=False):
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return self.sess.run(self.accuracy,
                             feed_dict={self.X: test_data_set.images, self.Y: test_data_set.labels, self.training: training})

    def train(self, train_data_set, train_params, training=True):
        self.train_params = train_params

        # optimizer
        self.set_optimizer(self.train_params.loss_fn, self.train_params.optimizer, self.train_params.learning_rate)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())
        print('\nTraining started...')

        # train my model
        num_epochs = self.train_params.training_epochs
        batch_size = self.train_params.batch_size
        learning_rate = self.train_params.learning_rate
        print('\t# of Epochs: %d, Batch size: %d, Learning rate: %f'
              % (num_epochs, batch_size, learning_rate))
        for epoch in range(num_epochs):
            avg_cost = 0
            total_batch = int(train_data_set.num_examples / batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = train_data_set.next_batch(batch_size)
                c, _ = self.sess.run([self.cost, self.optimizer],
                                     feed_dict={self.X: batch_xs, self.Y: batch_ys, self.training: training})
                self.avg_cost += c / total_batch

            print('\t\tEpoch:', '%04d/%04d' % (epoch + 1, num_epochs),
                  'cost =', '{:.9f}'.format(self.avg_cost))

        print('\nTraining finished.')

        # save session
        self.saver = tf.train.Saver()
        self.restore_dir = self.train_params.train_dir + self.model_params.name
        tf.gfile.MakeDirs(self.restore_dir)
        self.saver.save(self.sess, self.restore_dir + "model")
        print('\nModel saved to', self.restore_dir)

    def train_batch(self, batch_xs, batch_ys, training=True):
        c, _ = self.sess.run([self.cost, self.optimizer], feed_dict={self.X: batch_xs, self.Y: batch_ys, self.training: training})
        return c

    def load_model(self, restore_dir):
        # restore saved session
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, restore_dir + "model")

    def save_model(self, restore_dir):
        # save session
        self.saver = tf.train.Saver()
        tf.gfile.MakeDirs(restore_dir)
        self.saver.save(self.sess, restore_dir + "model")
