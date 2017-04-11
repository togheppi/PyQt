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
optimizer_type = Enum('optimizer_type', 'SGD Adam RMSProp')


class ModelParams:
    def __init__(self):
        # initial parameters
        self.name = "model/"
        self.input_size = 784
        self.num_classes = 10
        self.num_layers = 1

        self.layer_type_list = []
        self.output_size_list = []
        self.use_pooling = []
        self.use_dropout = []
        self.keep_prob = []
        for _ in range(self.num_layers):
            self.layer_type_list.append(layer_type.FC.value)
            self.output_size_list.append(128)
            self.use_pooling.append(False)
            self.use_dropout.append(False)
            self.keep_prob.append(1.0)

    def cnn(self):
        self.name = "tf_model/"
        self.num_layers = 3
        self.layer_type_list = []
        self.output_size_list = []
        self.use_pooling = []
        self.use_dropout = []
        self.keep_prob = []
        self.layer_type_list = [layer_type.CNN.value, layer_type.CNN.value, layer_type.FC.value]
        self.output_size_list = [32, 64, 128]
        for i in range(self.num_layers):
            if self.layer_type_list[i] == layer_type.FC.value:
                self.use_pooling.append(False)
            else:
                self.use_pooling.append(True)
            self.use_dropout.append(True)
            self.keep_prob.append(0.7)


class TrainParams:
    def __init__(self):
        # initial parameters
        self.train_dir = "MNIST_train/"
        self.learning_rate = 0.001
        self.training_epochs = 15
        self.batch_size = 100




class MyWindow(QMainWindow, ui_tf_testbed.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # TF session
        self.sess = tf.Session()
        # parameters
        self.model_params = ModelParams()
        self.model_params.cnn()   # CNN model
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
            self.model = tf_model.Model(self.sess, self.model_params, self.train_params)

            # build a model
            if self.model.build() == True:
                self.model_built = True
                print("Model built.")
            else:
                print("Failed to build a model!")
        else:
            print("Model already exists.")

    def btn_TrainModel_clicked(self):
        if self.model_built == True:
            # select optimizer
            self.model.set_optimizer(2)
            # training
            self.model.train(self.mnist.train)
        else:
            print('Model is not built!')

    def btn_LoadModel_clicked(self):
        # build a model
        self.btn_BuildModel_clicked()
        # restore saved session
        self.saver = tf.train.Saver()
        self.restore_dir = self.train_params.train_dir + self.model_params.name
        self.saver.restore(self.sess, self.restore_dir + "model")
        self.model_built = True
        print('Model restored from', self.restore_dir)

    def btn_Evaluation_clicked(self):
        # restore model
        if self.model_built == True:
            # Test model and check accuracy
            print('Evaluating model with test dataset...')
            print('Accuracy:', self.model.evaluate(self.mnist.test))
        else:
            print('No model is trained or restored!')

    def btn_Prediction_clicked(self):
        # restore model
        if self.model_built == True:
            img_fn = 'test_image8.bmp'
            test_image = Image.open(img_fn, "r")
            pixmap = QPixmap(img_fn)
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)

            result = self.model.predict(test_image)
            print("Prediction:", result[0])
        else:
            print('No model is trained or restored!')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()