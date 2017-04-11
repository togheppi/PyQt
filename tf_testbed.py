import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import ui_tf_testbed
from PIL import Image
import tf_model
import tensorflow as tf
from enum import Enum

# enum parameters
layer_type = Enum('layer_type', 'FC CNN LSTM')
activate_fn_type = Enum('activate_fn_type', 'Sigmoid tanh ReLU')
init_fn_type = Enum('init_fn_type', 'No_init Normal Xavier')
optimizer_type = Enum('optimizer_type', 'SGD Adam RMSProp')

class ModelParams:
    def __init__(self):
        # initial parameters
        self.name = "model/"
        self.input_size = 784
        self.num_classes = 10
        self.num_layers = 1

        self.layer_type = [1]
        self.activate_fn = [3]
        self.init_fn = [1]
        self.output_size = [128]
        self.kernel_size = [None]
        self.stride = [None]
        self.pool_size = [None]
        self.pool_stride = [None]
        self.use_pooling = [False]
        self.use_dropout = [False]
        self.keep_prob = [1.0]

    def dnn(self):
        self.name = "dnn_model/"
        self.num_layers = 4
        self.layer_type = [1, 1, 1, 1]
        self.activate_fn = [3, 3, 3, 3]
        self.init_fn = [3, 3, 3, 3]
        self.output_size = [512, 512, 512, 512]
        self.kernel_size = [None, None, None, None]
        self.stride = [None, None, None, None]
        self.pool_size = [None, None, None, None]
        self.pool_stride = [None, None, None, None]
        self.use_pooling = [False, False, False, False]
        self.use_dropout = [True, True, True, True]
        self.keep_prob = [0.7, 0.7, 0.7, 0.7]

    def cnn(self):
        self.name = "cnn_model/"
        self.num_layers = 3
        self.layer_type = [2, 2, 1]
        self.activate_fn = [3, 3, 3]
        self.init_fn = [3, 3, 3]
        self.output_size = [4, 8, 50]
        self.kernel_size = [3, 3, None]
        self.stride = [1, 1, None]
        self.pool_size = [2, 2, None]
        self.pool_stride = [2, 2, None]
        self.use_pooling = [True, True, False]
        self.use_dropout = [True, True, True]
        self.keep_prob = [0.7, 0.7, 0.7]
        # for i in range(self.num_layers):
        #     if self.layer_type[i] == layer_type.FC.value:
        #         self.use_pooling.append(False)
        #     else:
        #         self.use_pooling.append(True)
        #     self.use_dropout.append(True)
        #     self.keep_prob.append(0.7)


class TrainParams:
    def __init__(self):
        # initial parameters
        self.train_dir = "MNIST_train/"
        self.learning_rate = 0.001
        self.training_epochs = 5
        self.batch_size = 100


class MyWindow(QMainWindow, ui_tf_testbed.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # TF session
        self.sess = tf.Session()
        # parameters
        self.model_params = ModelParams()
        self.model_params.dnn()  # DNN model
        # self.model_params.cnn()   # CNN model
        self.train_params = TrainParams()
        self.model_built = False
        self.model_trained = False

        # load data
        self.mnist = tf_model.load_data()

        # ui control
        self.BuildModel.clicked.connect(self.btn_BuildModel_clicked)
        self.TrainModel.clicked.connect(self.btn_TrainModel_clicked)
        self.LoadModel.clicked.connect(self.btn_LoadModel_clicked)
        self.Evaluation.clicked.connect(self.btn_Evaluation_clicked)
        self.Prediction.clicked.connect(self.btn_Prediction_clicked)

    def btn_BuildModel_clicked(self):
        if self.model_built == False:

            # initialize a model
            print("\nBuilding a model...")
            self.model = tf_model.Model(self.sess, self.model_params, self.train_params)

            # build a model
            if self.model.build() == True:
                self.model_built = True
                print("\nModel built.")
            else:
                print("\nFailed to build a model!")
        else:
            print("\nModel already exists.")

    def btn_TrainModel_clicked(self):
        if self.model_built == True:
            # select optimizer
            self.model.set_optimizer(2)
            # training
            self.model.train(self.mnist.train)
        else:
            print('\nModel is not built!')

    def btn_LoadModel_clicked(self):
        # build a model
        self.btn_BuildModel_clicked()
        # restore saved session
        self.saver = tf.train.Saver()
        self.restore_dir = self.train_params.train_dir + self.model_params.name
        self.saver.restore(self.sess, self.restore_dir + "model")
        self.model_built = True
        print('\nModel restored from', self.restore_dir)

    def btn_Evaluation_clicked(self):
        # restore model
        if self.model_built == True:
            # Test model and check accuracy
            print('\nEvaluating model with test data set...')
            print('\tAccuracy:', self.model.evaluate(self.mnist.test))
        else:
            print('\nNo model is trained or restored!')

    def btn_Prediction_clicked(self):
        # restore model
        if self.model_built == True:
            img_fn = 'test_image.bmp'
            test_image = Image.open(img_fn, "r")
            pixmap = QPixmap(img_fn)
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)

            result = self.model.predict(test_image)
            print("Prediction:", result[0])
        else:
            print('\nNo model is trained or restored!')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()