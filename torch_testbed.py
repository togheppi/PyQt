import sys
import torch
import torch_model
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


def loadData(batch_size):
    # load data
    mnist_train, mnist_test = torch_model.load_data()
    num_batch_train = len(mnist_train) // batch_size
    num_batch_test = len(mnist_test) // batch_size

    # dataset loader
    data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size=batch_size,
                                              shuffle=True)

    return mnist_train, mnist_test, num_batch_train, num_batch_test, data_loader


def loadTrainBatch(data_loader, index):
    # load batch data
    batch_xs, batch_ys = data_loader[index]

    return batch_xs, batch_ys

def main():
    # initial parameters
    model_params = ModelParams()
    model_params.dnn()  # DNN model
    # self.model_params.cnn()   # CNN model
    train_params = TrainParams()
    model_built = False
    model_trained = False

    # initialize a model
    print("\nBuilding a model...")
    model = torch_model.TorchModel(model_params)

    # build a model
    if model.build():
        model_built = True
        print("\nModel built.")
    else:
        print("\nFailed to build a model!")

    num_epochs = train_params.training_epochs
    batch_size = train_params.batch_size
    learning_rate = train_params.learning_rate

    # load data
    mnist_train, mnist_test, num_batch_train, num_batch_test, data_loader = loadData(batch_size)

    # optimizer
    model.set_optimizer(train_params.loss_fn,
                        train_params.optimizer,
                        train_params.learning_rate)

    print('\nTraining started...')

    # train my model
    print('\t# of Epochs: %d, Batch size: %d, Learning rate: %f'
          % (num_epochs, batch_size, learning_rate))
    for epoch in range(num_epochs):
        avg_cost = 0

        for i, (batch_xs, batch_ys) in enumerate(data_loader):
            # batch_xs, batch_ys = loadTrainBatch(data_loader, i)
            c = model.train_batch(batch_xs, batch_ys)
            avg_cost += c / num_batch_train

        print('\t\tEpoch:', '%04d/%04d' % (epoch + 1, num_epochs),
              'cost =', '{:.9f}'.format(avg_cost))

    print('\nTraining finished.')

if __name__ == '__main__':
    main()
