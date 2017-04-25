import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import initializers
from keras import optimizers
from keras.utils import np_utils
import numpy as np
from enum import Enum

# input image dimensions
img_rows, img_cols = 28, 28
nb_classes = 10


class mnist_train:
    def __init__(self):
        self.images = []
        self.labels = []


class mnist_test:
    def __init__(self):
        self.images = []
        self.labels = []


def load_data(one_hot=True, dtype='float32'):
    # MNIST dataset

    # the data, shuffled and split between train and test sets
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    if dtype == 'float32':
        X_train = X_train.astype(dtype)
        X_test = X_test.astype(dtype)
        X_train /= 255
        X_test /= 255

    # if reshape:
    #     X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
    #     X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)

    # convert class vectors to binary class matrices
    if one_hot:
        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        Y_test = np_utils.to_categorical(Y_test, nb_classes)
    # else:
    #         Y_train = Y_train.reshape(Y_train.shape[0], 1)

    mnist_train.images = X_train
    mnist_train.labels = Y_train
    mnist_test.images = X_test
    mnist_test.labels = Y_test

    return mnist_train, mnist_test


# enum parameters
layer_type = Enum('layer_type', 'FC CNN')
activate_fn_type = Enum('activate_fn_type', 'No_act Sigmoid tanh ReLU')
init_fn_type = Enum('init_fn_type', 'No_init Normal Xavier')
loss_fn_type = Enum('loss_fn_type', 'Cross_Entropy')
optimizer_type = Enum('optimizer_type', 'SGD Adam RMSProp')


class KerasModel:
    def __init__(self, model_params):
        self.model_params = model_params

    def build(self):

        num_layers = self.model_params.num_layers
        self.model = Sequential(name=self.model_params.name)
        for i in range(num_layers):
            # Fully-connected Layer
            layer_name = 'Hidden%d' % (i + 1)
            print(layer_name)

            if self.model_params.layer_type[i] == layer_type.FC.value:
                # Initializer
                if self.model_params.init_fn[i] == init_fn_type.No_init.value:
                    init_fn = initializers.random_normal(stddev=1.0)
                elif self.model_params.init_fn[i] == init_fn_type.Normal.value:
                    init_fn = initializers.random_normal(stddev=1.0)
                elif self.model_params.init_fn[i] == init_fn_type.Xavier.value:
                    init_fn = initializers.glorot_uniform()
                else:
                    pass

                num_neurons = self.model_params.output_size[i]
                if i == 0:
                    prev_num_neurons = self.model_params.input_size
                    self.model.add(Dense(units=num_neurons, input_dim=prev_num_neurons,
                                         kernel_initializer=init_fn,
                                         use_bias=True,
                                         name=layer_name + '_Dense'))
                elif self.model_params.layer_type[i - 1] == layer_type.FC.value:
                    self.model.add(Dense(units=num_neurons,
                                         kernel_initializer=init_fn,
                                         use_bias=True,
                                         name=layer_name + '_Dense'))
                elif self.model_params.layer_type[i - 1] == layer_type.CNN.value:
                    self.model.add(Flatten(name=layer_name + '_Flatten'))
                    self.model.add(Dense(units=num_neurons,
                                         kernel_initializer=init_fn,
                                         use_bias=True,
                                         name=layer_name + '_Dense'))

                # Activation function
                if self.model_params.activate_fn[i] == activate_fn_type.Sigmoid.value:
                    self.model.add(Activation('sigmoid', name=layer_name + '_Act'))
                elif self.model_params.activate_fn[i] == activate_fn_type.tanh.value:
                    self.model.add(Activation('tanh', name=layer_name + '_Act'))
                elif self.model_params.activate_fn[i] == activate_fn_type.ReLU.value:
                    self.model.add(Activation('relu', name=layer_name + '_Act'))
                else:
                    pass

                print("\tAdding FC layer...")
                print("\t\t# of neurons = %d" % num_neurons)

            # Convolutional Layer
            elif self.model_params.layer_type[i] == layer_type.CNN.value:
                # Initializer
                if self.model_params.init_fn[i] == init_fn_type.No_init.value:
                    init_fn = initializers.random_normal(stddev=1.0)
                elif self.model_params.init_fn[i] == init_fn_type.Normal.value:
                    init_fn = initializers.random_normal(stddev=1.0)
                elif self.model_params.init_fn[i] == init_fn_type.Xavier.value:
                    init_fn = initializers.glorot_uniform()
                else:
                    pass

                num_neurons = self.model_params.output_size[i]
                k_size = self.model_params.kernel_size[i]
                s_size = self.model_params.kernel_stride[i]

                if i == 0:
                    self.model.add(Conv2D(filters=num_neurons,
                                          input_shape=(img_rows, img_cols, 1),
                                          kernel_size=(k_size, k_size),
                                          strides=(s_size, s_size),
                                          padding="same",
                                          kernel_initializer=init_fn,
                                          use_bias=True,
                                          name=layer_name + '_Conv2D'))
                else:
                    self.model.add(Conv2D(filters=num_neurons,
                                          kernel_size=(k_size, k_size),
                                          strides=(s_size, s_size),
                                          padding="same",
                                          kernel_initializer=init_fn,
                                          use_bias=True,
                                          name=layer_name + '_Conv2D'))

                # Activation function
                if self.model_params.activate_fn[i] == activate_fn_type.Sigmoid.value:
                    self.model.add(Activation('sigmoid', name=layer_name + '_Act'))
                elif self.model_params.activate_fn[i] == activate_fn_type.tanh.value:
                    self.model.add(Activation('tanh', name=layer_name + '_Act'))
                elif self.model_params.activate_fn[i] == activate_fn_type.ReLU.value:
                    self.model.add(Activation('relu', name=layer_name + '_Act'))
                else:
                    pass

                print("\tAdding Conv2D layer...")
                print("\t\tKernel size = %dx%d, Stride = (%d, %d)" % (k_size, k_size, s_size, s_size))
                # print("\t\tInput:", self.input.size(), "-> Output:", out.size())

                # Pooling Layer
                p_size = self.model_params.pool_size[i]
                pool_s_size = self.model_params.pool_stride[i]

                if self.model_params.use_pooling[i]:
                    self.model.add(MaxPooling2D(pool_size=(p_size, p_size),
                                                strides=(pool_s_size, pool_s_size),
                                                padding="same",
                                                name=layer_name + '_MaxPool'))

                    print("\tAdding MaxPooling layer...")
                    print("\t\tKernel size = %dx%d, Stride = (%d, %d)" %(p_size, p_size, pool_s_size, pool_s_size))

            # Dropout
            keep_prob = self.model_params.keep_prob[i]
            if self.model_params.use_dropout[i]:
                self.model.add(Dropout(rate=1-keep_prob, name=layer_name + '_DropOut'))

                print("\tAdding Dropout layer...")
                print("\t\tkeep_prob = %0.1f" % keep_prob)

        # Output (no activation) Layer
        print("Output layer: ")
        if self.model_params.layer_type[num_layers-1] == layer_type.CNN.value:
            self.model.add(Flatten(name='Output_Flatten'))

        self.model.add(Dense(units=self.model_params.num_classes,
                             use_bias=True,
                             name='Output_Dense'))

        self.model.add(Activation('softmax', name='Output_Act'))

        print("\t\tAdding FC layer...")

        # model info
        self.model.summary()

        # model viewer
        from keras.utils.vis_utils import plot_model
        import os
        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

        if not os.path.exists(self.model_params.init_model_dir):
            os.makedirs('./' + self.model_params.init_model_dir)

        # model = self.model.model
        # self.model.model.layers[0].name = 'Input'   # change name of input layer
        plot_model(self.model, to_file=self.model_params.init_model_dir + self.model_params.name + '.png', show_shapes=True)

        return True

    def set_optimizer(self, train_params):
        self.train_params = train_params
        lossFn = self.train_params.loss_fn
        optimizer = self.train_params.optimizer
        learning_rate = self.train_params.learning_rate

        # define cost/loss & optimizer
        if lossFn == loss_fn_type.Cross_Entropy.value:
            print("\nLoss function: Cross_Entropy.")
            self.criterion = 'categorical_crossentropy'

        if optimizer == optimizer_type.SGD.value:
            print("\nSGD optimizer is selected.")
            self.optimizer = optimizers.SGD(lr=learning_rate)
        elif optimizer == optimizer_type.Adam.value:
            print("\nAdam optimizer is selected.")
            self.optimizer = optimizers.Adam(lr=learning_rate)
        elif optimizer == optimizer_type.RMSProp.value:
            print("\nRMSProp optimizer is selected.")
            self.optimizer = optimizers.RMSprop(lr=learning_rate)
        else:
            pass

        self.model.compile(loss=self.criterion,
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

    def predict(self, image):
        image = np.asarray(image, dtype="float32")
        # normalize image 0 to 1
        if image.max() == 255:
            image = image / 255

        image = 1.0 - image

        if self.model_params.layer_type[0] == layer_type.FC.value:
            image = image.reshape(1, self.model_params.input_size)
        elif self.model_params.layer_type[0] == layer_type.CNN.value:
            image = image.reshape(1, img_rows, img_cols, 1)

        score = self.model.predict(image, batch_size=1, verbose=1)
        prediction = score.argmax(axis=-1)

        return score, prediction

    def evaluate(self, test_data_set):
        if self.model_params.layer_type[0] == layer_type.FC.value:
            test_data_set.images = test_data_set.images.reshape(test_data_set.images.shape[0], img_rows*img_cols)
        self.score = self.model.evaluate(test_data_set.images, test_data_set.labels, verbose=0)
        self.accuracy = self.score[1]

        return self.accuracy

    def train_batch(self, train_params, dataset):
        self.train_params = train_params
        if self.model_params.layer_type[0] == layer_type.FC.value:
            dataset.images = dataset.images.reshape(dataset.images.shape[0], img_rows*img_cols)

        tensorboard = keras.callbacks.TensorBoard(log_dir=self.train_params.train_dir, histogram_freq=1,
                                                  write_graph=True, write_images=False)

        self.model.fit(dataset.images, dataset.labels,
                       batch_size=self.train_params.batch_size, epochs=self.train_params.training_epochs,
                       verbose=1, validation_data=None, callbacks=[tensorboard])
