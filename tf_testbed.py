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
layer_type = Enum('layer_type', 'FC CNN')
activate_fn_type = Enum('activate_fn_type', 'No_act Sigmoid tanh ReLU')
init_fn_type = Enum('init_fn_type', 'No_init Normal Xavier')
loss_fn_type = Enum('loss_fn_type', 'Cross_Entropy')
optimizer_type = Enum('optimizer_type', 'SGD Adam RMSProp')

class ModelParams:
    def __init__(self):
        # initial parameters
        self.name = "model/"
        self.input_size = 784
        self.num_classes = 10
        self.num_layers = 0

        self.layer_type = []
        self.activate_fn = []
        self.init_fn = []
        self.output_size = []
        self.kernel_size = []
        self.kernel_stride = []
        self.pool_size = []
        self.pool_stride = []
        self.use_pooling = []
        self.use_dropout = []
        self.keep_prob = []

    def dnn(self):
        self.name = "dnn_model/"
        self.num_layers = 4
        self.layer_type = [1, 1, 1, 1]
        self.activate_fn = [3, 3, 3, 3]
        self.init_fn = [3, 3, 3, 3]
        self.output_size = [512, 512, 512, 512]
        self.kernel_size = [None, None, None, None]
        self.kernel_stride = [None, None, None, None]
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
        self.kernel_stride = [1, 1, None]
        self.pool_size = [2, 2, None]
        self.pool_stride = [2, 2, None]
        self.use_pooling = [True, True, False]
        self.use_dropout = [True, True, True]
        self.keep_prob = [0.7, 0.7, 0.7]



class TrainParams:
    def __init__(self):
        # initial parameters
        self.train_dir = "MNIST_train/"
        self.loss_fn = loss_fn_type.Cross_Entropy.value
        self.optimizer = optimizer_type.Adam.value
        self.learning_rate = 0.001
        self.training_epochs = 10
        self.batch_size = 100


class MyWindow(QMainWindow, ui_tf_testbed.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # ui control
        # combo boxes
        self.comboBox_actFn.addItems(['No_act', 'Sigmoid', 'tanh', 'ReLU'])
        self.comboBox_initFn.addItems(['No_init', 'Normal', 'Xavier'])
        self.comboBox_lossFn.addItems(['Cross Entropy'])
        self.comboBox_optimizer.addItems(['SGD', 'Adam', 'RMSProp'])

        self.pushButton_addLayer.clicked.connect(self.btn_AddLayer_clicked)
        self.pushButton_buildModel.clicked.connect(self.btn_BuildModel_clicked)
        self.pushButton_trainModel.clicked.connect(self.btn_TrainModel_clicked)
        self.pushButton_loadModel.clicked.connect(self.btn_LoadModel_clicked)
        self.pushButton_evaluation.clicked.connect(self.btn_Evaluation_clicked)
        self.pushButton_prediction.clicked.connect(self.btn_Prediction_clicked)
        self.pushButton_Exit.clicked.connect(app.quit)

        self.radioButton_tensorflow.clicked.connect(self.tensorflow_selected)
        self.radioButton_keras.clicked.connect(self.keras_selected)
        self.radioButton_pytorch.clicked.connect(self.pytorch_selected)

        # parameters
        self.model_params = ModelParams()
        # self.model_params.dnn()  # DNN model
        # self.model_params.cnn()   # CNN model
        self.train_params = TrainParams()
        self.model_built = False
        self.model_trained = False
        self.num_layers = 0

        if self.radioButton_tensorflow.isChecked():
            self.tensorflow_selected()
        elif self.radioButton_keras.isChecked():
            self.keras_selected()
        elif self.radioButton_pytorch.isChecked():
            self.pytorch_selected()

    def tensorflow_selected(self):
        print("TensorFlow library is selected.")
        # TF session
        self.sess = tf.Session()

        # load data
        self.mnist = tf_model.load_data()

        # model name
        self.model_params.name = 'tf_model/'

    def keras_selected(self):
        print("Keras (TF backend) library is selected.")

    def pytorch_selected(self):
        print("PyTorch library is selected.")

    def btn_AddLayer_clicked(self):

        # layer type
        if self.radioButton_FC.isChecked():
            self.model_params.layer_type.append(layer_type.FC.value)
            self.model_params.output_size.append(self.spinBox_numNeurons.value())
        elif self.radioButton_CNN.isChecked():
            self.model_params.layer_type.append(layer_type.CNN.value)
            self.model_params.kernel_size.append(self.spinBox_kernelSize.value())
            self.model_params.kernel_stride.append(self.spinBox_kernelStride.value())
            self.model_params.output_size.append(self.spinBox_numFilters.value())
        # activation function
        if self.checkBox_actFn.isChecked():
            self.model_params.activate_fn.append(self.comboBox_actFn.currentIndex() + 1)
        else:
            self.model_params.activate_fn.append(activate_fn_type.No_act.value)
            self.comboBox_actFn.setCurrentIndex(activate_fn_type.No_act.value - 1)
        # initializer
        if self.checkBox_initFn.isChecked():
            self.model_params.init_fn.append(self.comboBox_initFn.currentIndex() + 1)
        else:
            self.model_params.init_fn.append(init_fn_type.No_init.value)
            self.comboBox_initFn.setCurrentIndex(init_fn_type.No_init.value - 1)
        # max pooling
        if self.checkBox_maxPooling.isChecked():
            self.model_params.use_pooling.append(True)
            self.model_params.pool_size.append(self.spinBox_poolSize.value())
            self.model_params.pool_stride.append(self.spinBox_poolStride.value())
        else:
            self.model_params.use_pooling.append(False)
            self.model_params.pool_size.append(None)
            self.model_params.pool_stride.append(None)
        # dropout
        if self.checkBox_dropOut.isChecked():
            self.model_params.use_dropout.append(True)
            self.model_params.keep_prob.append(self.doubleSpinBox_keepProb.value())
        else:
            self.model_params.use_dropout.append(False)
            self.model_params.keep_prob.append(1.0)

        # increase number of hidden layers
        self.model_params.num_layers += 1
        self.lineEdit_numLayers.setText('%d layer(s)' % self.model_params.num_layers)
        print("Hidden layer #%d is added to model name, '%s'."
              % (self.model_params.num_layers, self.model_params.name))

    def btn_BuildModel_clicked(self):
        if self.model_built == False:

            # input / output size
            self.model_params.input_size = self.spinBox_inputSize.value()
            self.model_params.num_classes = self.spinBox_numClasses.value()

            # initialize a model
            print("\nBuilding a model...")
            self.model = tf_model.Model(self.sess, self.model_params)

            # build a model
            if self.model.build() == True:
                self.model_built = True
                print("\nModel built.")
            else:
                print("\nFailed to build a model!")
                return False
        else:
            print("\nModel already exists.")
            return False

        return True

    def btn_TrainModel_clicked(self):
        if self.model_built == True:
            # train parameters
            self.train_params.loss_fn = self.comboBox_lossFn.currentIndex() + 1
            self.train_params.optimizer = self.comboBox_optimizer.currentIndex() + 1
            self.train_params.training_epochs = self.spinBox_numEpochs.value()
            self.train_params.batch_size = self.spinBox_batchSize.value()
            self.train_params.learning_rate = float(self.lineEdit_learningRate.text())

            # training
            self.model.train(self.mnist.train, self.train_params)
        else:
            print('\nModel is not built!')

    def btn_LoadModel_clicked(self):
        # build a model
        if self.btn_BuildModel_clicked():
            # restore saved session
            self.saver = tf.train.Saver()
            self.restore_dir = self.train_params.train_dir + self.model_params.name
            self.saver.restore(self.sess, self.restore_dir + "model")
            self.model_built = True
            print('\nModel restored from', self.restore_dir)
        else:
            print('\nNo model restored.')

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
            self.label_inputImage.setPixmap(pixmap)
            self.label_inputImage.setScaledContents(True)

            result = self.model.predict(test_image)
            print("Prediction:", result[0])
        else:
            print('\nNo model is trained or restored!')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()