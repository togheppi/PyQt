import os
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import ui_tf_testbed
from PIL import Image
import pickle

# import torch_model
# import tf_model

from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

# enum parameters
layer_type = Enum('layer_type', 'FC CNN')
activate_fn_type = Enum('activate_fn_type', 'No_act Sigmoid tanh ReLU')
init_fn_type = Enum('init_fn_type', 'No_init Normal Xavier')
loss_fn_type = Enum('loss_fn_type', 'Cross_Entropy')
optimizer_type = Enum('optimizer_type', 'SGD Adam RMSProp')

class ModelParams:
    def __init__(self):
        # initial parameters
        self.name = "model"
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
        self.init_model_dir = 'MNIST_train/'


    def dnn(self):
        self.name = "dnn_model"
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
        self.name = "cnn_model"
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

        self.loss_fn = loss_fn_type.Cross_Entropy.value
        self.optimizer = optimizer_type.Adam.value
        self.learning_rate = 0.001
        self.training_epochs = 10
        self.batch_size = 100
        self.num_train_batch = 1
        self.train_dir = "MNIST_train/"

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
        self.pushButton_clearModel.clicked.connect(self.btn_ClearModel_clicked)
        self.pushButton_trainModel.clicked.connect(self.btn_TrainModel_clicked)
        self.pushButton_saveModel.clicked.connect(self.btn_SaveModel_clicked)
        self.pushButton_loadModel.clicked.connect(self.btn_LoadModel_clicked)
        self.pushButton_evaluation.clicked.connect(self.btn_Evaluation_clicked)
        self.pushButton_prediction.clicked.connect(self.btn_Prediction_clicked)
        self.pushButton_loadImage.clicked.connect(self.btn_LoadImage_clicked)
        self.pushButton_tensorBoard.clicked.connect(self.btn_TensorBoard_clicked)
        self.pushButton_Exit.clicked.connect(app.quit)

        self.pushButton_buildModel.setDisabled(True)
        self.pushButton_saveModel.setDisabled(True)
        self.pushButton_trainModel.setDisabled(True)
        self.pushButton_clearModel.setDisabled(True)
        self.pushButton_prediction.setDisabled(True)

        self.radioButton_tensorflow.clicked.connect(self.tensorflow_selected)
        self.radioButton_keras.clicked.connect(self.keras_selected)
        self.radioButton_pytorch.clicked.connect(self.pytorch_selected)

        # initial parameters
        self.model_params = ModelParams()
        # self.model_params.dnn()  # DNN model
        # self.model_params.cnn()   # CNN model
        self.train_params = TrainParams()
        self.train_dir = "MNIST_train/"

        self.model_built = False
        self.model_trained = False

        self.comboBox_actFn.setCurrentIndex(3)  # ReLU
        self.comboBox_initFn.setCurrentIndex(0)  # No initializer
        self.comboBox_lossFn.setCurrentIndex(0)  # Cross-Entropy
        self.comboBox_optimizer.setCurrentIndex(1)  # Adam optimizer

        # load library module
        if self.radioButton_tensorflow.isChecked():
            self.tensorflow_selected()
        elif self.radioButton_keras.isChecked():
            self.keras_selected()
        elif self.radioButton_pytorch.isChecked():
            self.pytorch_selected()

        # show score
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.horizontalLayout_plotPrediction.addWidget(self.canvas)
        self.subplot = self.fig.add_subplot(1, 1, 1)
        self.subplot.set_xticks(range(10))
        self.subplot.set_xlim(-0.5, 9.5)
        self.subplot.set_ylim(0, 1)
        self.subplot.tick_params(axis='both', labelsize=8)


    def tensorflow_selected(self):
        if 'torch_model' in sys.modules:
            del sys.modules['torch_model']
            print('torch_model deleted')
        if 'torch' in sys.modules:
            del sys.modules['torch']
            print('torch deleted')
        if 'Variable' in sys.modules:
            del sys.modules['Variable']
            print('Variable deleted')
        globals()['tf_model'] = __import__('tf_model')
        print("TensorFlow library is selected.")

        if not os.path.exists(self.train_dir + 'tf_model'):
            os.mkdir(self.train_dir + 'tf_model')

    def keras_selected(self):
        if 'torch_model' in sys.modules:
            del sys.modules['torch_model']
            print('torch_model deleted')
        if 'torch' in sys.modules:
            del sys.modules['torch']
            print('torch deleted')
        if 'Variable' in sys.modules:
            del sys.modules['Variable']
            print('Variable deleted')
        globals()['keras_model'] = __import__('keras_model')
        print("Keras (TF backend) library is selected.")

        if not os.path.exists(self.train_dir + 'keras_model'):
            os.mkdir(self.train_dir + 'keras_model')

    def pytorch_selected(self):
        if 'tf_model' in sys.modules:
            del sys.modules['tf_model']
            print('tf_model deleted')
        if 'tf' in sys.modules:
            del sys.modules['tf']
            print('tf deleted')
        globals()['torch_model'] = __import__('torch_model')
        print("PyTorch library is selected.")

        if not os.path.exists(self.train_dir + 'torch_model'):
            os.mkdir(self.train_dir + 'torch_model')

    def loadData(self):
        # load data
        if self.radioButton_tensorflow.isChecked():
            mnist_data_set = tf_model.load_data()
            self.mnist_train = mnist_data_set.train
            self.mnist_test = mnist_data_set.test
            self.train_params.num_batch_train = int(self.mnist_train.num_examples / self.train_params.batch_size)
            self.train_params.num_batch_test = int(self.mnist_test.num_examples / self.train_params.batch_size)
        elif self.radioButton_keras.isChecked():
            self.mnist_train, self.mnist_test = keras_model.load_data()
            self.train_params.num_batch_train = (self.mnist_train.images.shape[0]) // self.train_params.batch_size
            self.train_params.num_batch_test = (self.mnist_test.images.shape[0]) // self.train_params.batch_size
            # pass
        elif self.radioButton_pytorch.isChecked():
            # self.mnist_train, self.mnist_test = torch_model.load_data()
            # self.train_params.num_batch_train = len(self.mnist_train) // self.train_params.batch_size
            # self.train_params.num_batch_test = len(self.mnist_test) // self.train_params.batch_size
            pass

    def save_model_params(self, model_params):
        if not os.path.exists(model_params.init_model_dir):
            os.mkdir(model_params.init_model_dir)

        model_params_fn = model_params.init_model_dir + 'model_params.dat'
        f = open(model_params_fn, "wb")
        pickle.dump(model_params, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def load_model_params(self, model_dir):
        model_params_fn = model_dir + 'model_params.dat'
        f = open(model_params_fn, "rb")
        model_params = pickle.load(f)
        f.close()

        return model_params

    def btn_ClearModel_clicked(self):
        # # delete model
        # del self.model
        del self.model_params
        del self.train_params

        # reset parameters
        self.model_params = ModelParams()
        # self.model_params.dnn()  # DNN model
        # self.model_params.cnn()   # CNN model
        self.train_params = TrainParams()
        self.train_dir = "MNIST_train/"

        self.model_built = False
        self.model_trained = False

        # reset number of layer
        self.lineEdit_numLayers.setText('%d layer(s)' % self.model_params.num_layers)

        print('Clear model.')
        self.textEdit_log.append('Clear model.')

        self.pushButton_addLayer.setEnabled(True)
        # self.pushButton_buildModel.setEnabled(True)
        self.pushButton_trainModel.setDisabled(True)
        self.pushButton_saveModel.setDisabled(True)

    def btn_AddLayer_clicked(self):
        # model name
        self.model_params.name = self.lineEdit_modelName.text()
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
        self.textEdit_log.append("Hidden layer #%d is added to model name, '%s'."
                                 % (self.model_params.num_layers, self.model_params.name))

        self.pushButton_buildModel.setEnabled(True)
        self.pushButton_clearModel.setEnabled(True)

    def btn_BuildModel_clicked(self):
        if self.model_built:
            print("\nModel already exists.")
            return False
        else:
            # input / output size
            self.model_params.input_size = self.spinBox_inputSize.value()
            self.model_params.num_classes = self.spinBox_numClasses.value()

            # save model dir
            if self.radioButton_tensorflow.isChecked():
                self.model_params.init_model_dir = self.train_dir + 'tf_model/' + self.model_params.name + '/'
            elif self.radioButton_keras.isChecked():
                self.model_params.init_model_dir = self.train_dir + 'keras_model/' + self.model_params.name + '/'
                # pass
            elif self.radioButton_pytorch.isChecked():
                # init_model_dir = self.train_dir + 'torch_model/' + self.model_params.name + '/'
                pass

            # save model params to file
            self.save_model_params(self.model_params)

            # initialize a model with respective library
            print("\nBuilding a model...")
            self.textEdit_log.append("Building a model...")
            if self.radioButton_tensorflow.isChecked():
                self.model = tf_model.TFModel(self.model_params)
            elif self.radioButton_keras.isChecked():
                self.model = keras_model.KerasModel(self.model_params)
                # pass
            elif self.radioButton_pytorch.isChecked():
                # self.model = torch_model.TorchModel(self.model_params)
                pass

            # build a model
            if self.model.build():
                self.model_built = True
                self.model_params.num_layers = 0
                print("\nModel built.")
                self.textEdit_log.append("Model built.")

                if self.radioButton_keras.isChecked():
                    img_fn = self.model_params.init_model_dir + self.model_params.name + '.png'
                    pixmap = QPixmap(img_fn)
                    label_modelViewer = QLabel()

                    width = self.scrollArea_modelViewer.width()
                    label_modelViewer.setPixmap(pixmap.scaledToWidth(width, mode=Qt.SmoothTransformation))
                    self.scrollArea_modelViewer.setWidget(label_modelViewer)

                    self.scrollArea_modelViewer.setMinimumWidth(
                        width + self.scrollArea_modelViewer.verticalScrollBar().sizeHint().width())

                self.pushButton_buildModel.setDisabled(True)
                self.pushButton_addLayer.setDisabled(True)
                self.pushButton_trainModel.setEnabled(True)
                self.pushButton_saveModel.setEnabled(True)
            else:
                print("\nFailed to build a model!")
                return False

        return True

    def btn_TrainModel_clicked(self):
        if self.model_built:
            # save train dir
            run = 1
            train_dir = self.model_params.init_model_dir + 'run_%d/' % run
            # increase number of runs
            while os.path.exists(train_dir):
                run += 1
                train_dir = self.model_params.init_model_dir + 'run_%d/' % run

            self.train_params.train_dir = train_dir

            # train parameters
            self.train_params.loss_fn = self.comboBox_lossFn.currentIndex() + 1
            self.train_params.optimizer = self.comboBox_optimizer.currentIndex() + 1
            self.train_params.training_epochs = self.spinBox_numEpochs.value()
            self.train_params.batch_size = self.spinBox_batchSize.value()
            self.train_params.learning_rate = float(self.lineEdit_learningRate.text())

            num_epochs = self.train_params.training_epochs
            batch_size = self.train_params.batch_size
            learning_rate = self.train_params.learning_rate

            # load data
            self.loadData()

            # optimizer
            self.model.set_optimizer(self.train_params)

            print('\nTraining started...')
            self.textEdit_log.append("Training started...")

            # train my model
            print('\t# of Epochs: %d, Batch size: %d, Learning rate: %0.4f'
                  % (num_epochs, batch_size, learning_rate))
            self.textEdit_log.append('# of Epochs: %d, Batch size: %d, Learning rate: %0.4f'
                                     % (num_epochs, batch_size, learning_rate))

            if self.radioButton_keras.isChecked():
                self.model.train_batch(self.train_params, self.mnist_train)
            else:
                for epoch in range(num_epochs):
                    if self.radioButton_tensorflow.isChecked():
                        global_step = epoch * self.train_params.num_batch_train
                        avg_cost, avg_accu = self.model.train_batch(self.train_params, self.mnist_train, global_step)
                    elif self.radioButton_pytorch.isChecked():
                        # avg_cost, avg_accu = self.model.train_batch(self.train_params, self.mnist_train)
                        pass

                    print('\t\tEpoch:', '%04d/%04d' % (epoch + 1, num_epochs),
                          'cost =', '{:.9f}'.format(avg_cost),
                          'accu =', '{:.4f}'.format(avg_accu))

            print('\nTraining finished.')
            self.textEdit_log.append("Training finished.")

            self.pushButton_saveModel.setEnabled(True)
            self.pushButton_evaluation.setEnabled(True)
            self.pushButton_prediction.setEnabled(True)

        else:
            print('\nModel is not built!')

    def btn_SaveModel_clicked(self):
        if self.model_built:
            # save model
            # if self.radioButton_tensorflow.isChecked():
            self.model.save_model(self.model_params.init_model_dir)
            print('\nModel saved to', self.model_params.init_model_dir)
            self.textEdit_log.append('Model saved to ' + self.model_params.init_model_dir)
        else:
            print('\nModel is not built!')

    def btn_LoadModel_clicked(self):
        # deletes the existing model
        if self.model_built:
            del self.model
            self.model_built = False

        # model name to load
        model_name = self.lineEdit_modelName.text()


        if self.radioButton_tensorflow.isChecked():
            # load model dir
            self.model_params.init_model_dir = self.train_dir + 'tf_model/' + model_name + '/'

            # rebuild and load model with saved model params
            self.model_params = self.load_model_params(self.model_params.init_model_dir)
            if self.btn_BuildModel_clicked():
                self.model.load_model(self.model_params.init_model_dir)
                self.model_built = True
                print('\nModel restored from', self.model_params.init_model_dir)
            else:
                print('\nNo model restored.')

        elif self.radioButton_keras.isChecked():
            # load model dir
            self.model_params.init_model_dir = self.train_dir + 'keras_model/' + model_name + '/'

            # instantiate new model with saved model params
            self.model_params = self.load_model_params(self.model_params.init_model_dir)
            self.model = keras_model.KerasModel(self.model_params)

            # load model
            self.model.load_model(self.model_params.init_model_dir)
            self.model_built = True
            print('\nModel restored from', self.model_params.init_model_dir)
            self.textEdit_log.append('Model restored from ' + self.model_params.init_model_dir)

            # show load model info
            img_fn = self.model_params.init_model_dir + self.model_params.name + '.png'
            pixmap = QPixmap(img_fn)
            label_modelViewer = QLabel()

            width = self.scrollArea_modelViewer.width()
            label_modelViewer.setPixmap(pixmap.scaledToWidth(width, mode=Qt.SmoothTransformation))
            self.scrollArea_modelViewer.setWidget(label_modelViewer)

            self.scrollArea_modelViewer.setMinimumWidth(
                width + self.scrollArea_modelViewer.verticalScrollBar().sizeHint().width())

        elif self.radioButton_pytorch.isChecked():
            # init_model_dir = self.train_dir + 'torch_model/' + self.model_params.name + '/'
            pass

        self.pushButton_buildModel.setDisabled(True)
        self.pushButton_addLayer.setDisabled(True)
        self.pushButton_trainModel.setEnabled(True)
        self.pushButton_saveModel.setEnabled(True)

    def btn_Evaluation_clicked(self):
        if self.model_built:
            # load data
            self.loadData()

            # Test model and check accuracy
            print('\nEvaluating model with test data set...')
            self.textEdit_log.append('Evaluating model with test data set...')
            print('\tAccuracy:', self.model.evaluate(self.mnist_test))
            self.textEdit_log.append('Accuracy: %0.4f' % self.model.evaluate(self.mnist_test))
        else:
            print('\nNo model is trained or loaded!')

    def btn_LoadImage_clicked(self):
        # Load test image
        img_fn = QFileDialog.getOpenFileName(self, 'Open file')[0]
        pixmap = QPixmap(img_fn)
        self.label_inputImage.setPixmap(pixmap)
        self.label_inputImage.setScaledContents(True)

        self.test_image = Image.open(img_fn, "r")
        self.subplot.clear()
        print('\nTest image: ', img_fn, 'is loaded.')

        self.pushButton_prediction.setEnabled(True)

        return self.test_image

    def btn_Prediction_clicked(self):
        if self.model_built:
            # prediction
            score, result = self.model.predict(self.test_image)
            print("Prediction:", result)
            self.textEdit_log.append("Prediction: %d" % result)

            # show score
            self.subplot.set_xticks(range(10))
            self.subplot.set_xlim(-0.5, 9.5)
            self.subplot.set_ylim(0, 1)
            self.subplot.bar(range(10), score[0], align='center')
            self.canvas.draw()

        else:
            print('\nNo model is trained or restored!')

    def btn_TensorBoard_clicked(self):
        import subprocess

        subprocess.Popen("tensorboard --logdir=%s" % self.train_dir, shell=True)

        print('\nTensorBoard is running: logdir =', self.train_dir)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()