import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import ui_tf_testbed
from PIL import Image

# import torch_model
# import tf_model



from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

# enum parameters
layer_type = Enum('layer_type', 'FC CNN')
activate_fn_type = Enum('activate_fn_type', 'No_act Sigmoid tanh ReLU')
init_fn_type = Enum('init_fn_type', 'Uniform Normal Xavier')
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
        self.train_dir = "MNIST_train/"

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


class MyWindow(QMainWindow, ui_tf_testbed.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # ui control
        # combo boxes
        self.comboBox_actFn.addItems(['No_act', 'Sigmoid', 'tanh', 'ReLU'])
        self.comboBox_initFn.addItems(['Uniform', 'Normal', 'Xavier'])
        self.comboBox_lossFn.addItems(['Cross Entropy'])
        self.comboBox_optimizer.addItems(['SGD', 'Adam', 'RMSProp'])

        self.pushButton_addLayer.clicked.connect(self.btn_AddLayer_clicked)
        self.pushButton_buildModel.clicked.connect(self.btn_BuildModel_clicked)
        self.pushButton_clearModel.clicked.connect(self.btn_ClearModel_clicked)
        self.pushButton_trainModel.clicked.connect(self.btn_TrainModel_clicked)
        self.pushButton_loadModel.clicked.connect(self.btn_LoadModel_clicked)
        self.pushButton_evaluation.clicked.connect(self.btn_Evaluation_clicked)
        self.pushButton_prediction.clicked.connect(self.btn_Prediction_clicked)
        self.pushButton_Exit.clicked.connect(app.quit)

        self.radioButton_tensorflow.clicked.connect(self.tensorflow_selected)
        self.radioButton_keras.clicked.connect(self.keras_selected)
        self.radioButton_pytorch.clicked.connect(self.pytorch_selected)

        # load library module
        if self.radioButton_tensorflow.isChecked():
            self.tensorflow_selected()
        elif self.radioButton_keras.isChecked():
            self.keras_selected()
        elif self.radioButton_pytorch.isChecked():
            self.pytorch_selected()

        # initial parameters
        self.model_params = ModelParams()
        # self.model_params.dnn()  # DNN model
        # self.model_params.cnn()   # CNN model
        self.train_params = TrainParams()
        self.train_dir = "MNIST_train/"

        self.model_built = False
        self.model_trained = False

        self.comboBox_actFn.setCurrentIndex(3)      # ReLU
        self.comboBox_initFn.setCurrentIndex(0)     # No initializer
        self.comboBox_lossFn.setCurrentIndex(0)     # Cross-Entropy
        self.comboBox_optimizer.setCurrentIndex(1)  # Adam optimizer


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

    def pytorch_selected(self):
        # if 'tf_model' in sys.modules:
        #     del sys.modules['tf_model']
        #     print('tf_model deleted')
        # if 'tf' in sys.modules:
        #     del sys.modules['tf']
        #     print('tf deleted')
        # globals()['torch_model'] = __import__('torch_model')

        print("PyTorch library is selected.")

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

    # def loadTrainBatch(self, batch_size):
    #     # load batch data
    #     if self.radioButton_tensorflow.isChecked():
    #         batch_xs, batch_ys = tf_model.load_batch_data(self.mnist_train, batch_size)
    #         return batch_xs, batch_ys
    #     elif self.radioButton_keras.isChecked():
    #         pass
    #     elif self.radioButton_pytorch.isChecked():
    #         batch_x_list, batch_y_list = torch_model.load_batch_data(self.mnist_train, batch_size)
    #         return batch_x_list, batch_y_list

    def btn_ClearModel_clicked(self):
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
            self.model_params.init_fn.append(init_fn_type.Uniform.value)
            self.comboBox_initFn.setCurrentIndex(init_fn_type.Uniform.value - 1)
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

    def btn_BuildModel_clicked(self):
        if self.model_built:
            print("\nModel already exists.")
            return False
        else:
            # input / output size
            self.model_params.input_size = self.spinBox_inputSize.value()
            self.model_params.num_classes = self.spinBox_numClasses.value()

            # initialize a model with respective library
            print("\nBuilding a model...")
            self.textEdit_log.append("Building a model...")
            if self.radioButton_tensorflow.isChecked():
                self.model_params.train_dir = self.train_dir + 'tf_model/' + self.model_params.name + '/'
                self.model = tf_model.TFModel(self.model_params)
            elif self.radioButton_keras.isChecked():
                self.model_params.train_dir = self.train_dir + 'keras_model/' + self.model_params.name + '/'
                self.model = keras_model.KerasModel(self.model_params)
                # pass
            elif self.radioButton_pytorch.isChecked():
                # self.model_params.train_dir = self.train_dir + 'torch_model/' + self.model_params.name + '/'
                # self.model = torch_model.TorchModel(self.model_params)
                pass

            # build a model
            if self.model.build():
                self.model_built = True
                print("\nModel built.")
                self.textEdit_log.append("Model built.")
            else:
                print("\nFailed to build a model!")
                return False

        return True

    def btn_TrainModel_clicked(self):
        if self.model_built:
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
            self.model.set_optimizer(self.train_params.loss_fn,
                                     self.train_params.optimizer,
                                     self.train_params.learning_rate)

            print('\nTraining started...')
            self.textEdit_log.append("Training started...")

            # train my model
            print('\t# of Epochs: %d, Batch size: %d, Learning rate: %f'
                  % (num_epochs, batch_size, learning_rate))
            self.textEdit_log.append('# of Epochs: %d, Batch size: %d, Learning rate: %f'
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
                          'cost =', '{:.9f}'.format(avg_cost))
                    self.textEdit_training.append('Epoch #%d\n' % (epoch + 1) + 'cost={:.3f}'.format(avg_cost))

            print('\nTraining finished.')
            self.textEdit_log.append("Training finished.")

            # save model
            # self.model.save_model(self.model_params.train_dir)
            # print('\nModel saved to', self.model_params.train_dir)
            # self.textEdit_log.append('Model saved to' + self.model_params.train_dir)
        else:
            print('\nModel is not built!')

    def btn_LoadModel_clicked(self):
        # build a model
        if self.btn_BuildModel_clicked():
            self.model.load_model(self.model_params.train_dir)
            self.model_built = True
            print('\nModel restored from', self.model_params.train_dir)
        else:
            print('\nNo model restored.')

    def btn_Evaluation_clicked(self):
        # restore model
        if self.model_built:
            # load data
            self.loadData()
            # Test model and check accuracy
            print('\nEvaluating model with test data set...')
            self.textEdit_log.append('Evaluating model with test data set...')
            print('\tAccuracy:', self.model.evaluate(self.mnist_test))
            # self.textEdit_log.append('Accuracy:' + self.model.evaluate(self.mnist_test))
        else:
            print('\nNo model is trained or restored!')

    def btn_Prediction_clicked(self):
        # restore model
        if self.model_built:
            img_fn = 'test_image.bmp'
            test_image = Image.open(img_fn, "r")
            pixmap = QPixmap(img_fn)
            self.label_inputImage.setPixmap(pixmap)
            self.label_inputImage.setScaledContents(True)

            score, result = self.model.predict(test_image)
            print("Prediction:", result)
            # self.textEdit_log.append("Prediction:" + result)

            # show score
            fig = plt.Figure()
            self.canvas = FigureCanvas(fig)
            self.horizontalLayout_plotPrediction.addWidget(self.canvas)
            subplot = fig.add_subplot(1, 1, 1)
            subplot.set_xticks(range(10))
            subplot.set_xlim(-0.5, 9.5)
            subplot.set_ylim(0, 1)
            subplot.bar(range(10), score[0], align='center')
            subplot.tick_params(axis='both', labelsize=8)

            # plt.show()
            self.canvas.draw()

        else:
            print('\nNo model is trained or restored!')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()