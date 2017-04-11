# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tf_testbed.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(790, 545)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 301, 301))
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 30, 281, 261))
        self.label.setObjectName("label")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(430, 30, 131, 231))
        self.groupBox_2.setObjectName("groupBox_2")
        self.TrainModel = QtWidgets.QPushButton(self.groupBox_2)
        self.TrainModel.setGeometry(QtCore.QRect(10, 71, 111, 31))
        self.TrainModel.setObjectName("TrainModel")
        self.Evaluation = QtWidgets.QPushButton(self.groupBox_2)
        self.Evaluation.setGeometry(QtCore.QRect(10, 151, 111, 31))
        self.Evaluation.setObjectName("Evaluation")
        self.Prediction = QtWidgets.QPushButton(self.groupBox_2)
        self.Prediction.setGeometry(QtCore.QRect(10, 190, 111, 31))
        self.Prediction.setObjectName("Prediction")
        self.BuildModel = QtWidgets.QPushButton(self.groupBox_2)
        self.BuildModel.setGeometry(QtCore.QRect(10, 30, 111, 31))
        self.BuildModel.setObjectName("BuildModel")
        self.LoadModel = QtWidgets.QPushButton(self.groupBox_2)
        self.LoadModel.setGeometry(QtCore.QRect(10, 110, 111, 31))
        self.LoadModel.setObjectName("LoadModel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "TensorFlow Testbed"))
        self.groupBox.setTitle(_translate("MainWindow", "Input image"))
        self.label.setText(_translate("MainWindow", "MNIST image"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Model Builder"))
        self.TrainModel.setText(_translate("MainWindow", "Train model"))
        self.Evaluation.setText(_translate("MainWindow", "Evaluation"))
        self.Prediction.setText(_translate("MainWindow", "Prediction"))
        self.BuildModel.setText(_translate("MainWindow", "Build model"))
        self.LoadModel.setText(_translate("MainWindow", "Load model"))

