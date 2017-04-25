# DNN model builder using PyTorch
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import numpy as np
from enum import Enum
from visualize import make_dot


def load_data(data_path="MNIST_data/"):
    # MNIST dataset
    mnist_train = dsets.MNIST(root=data_path,
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    mnist_test = dsets.MNIST(root=data_path,
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)
    return mnist_train, mnist_test


def load_batch_data(dataset, batch_size):
    # dataset loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    # batch_x_list = []
    # batch_y_list = []
    # for i, (batch_xs, batch_ys) in enumerate(data_loader):
    #     batch_x_list.append(batch_xs)
    #     batch_y_list.append(batch_ys)
    return data_loader

# input_size = 784
# num_classes = 10

# enum parameters
layer_type = Enum('layer_type', 'FC CNN')
activate_fn_type = Enum('activate_fn_type', 'No_act Sigmoid tanh ReLU')
init_fn_type = Enum('init_fn_type', 'No_init Normal Xavier')
loss_fn_type = Enum('loss_fn_type', 'Cross_Entropy')
optimizer_type = Enum('optimizer_type', 'SGD Adam RMSProp')


class TorchModel(torch.nn.Module):

    def __init__(self, model_params):
        super(TorchModel, self).__init__()
        self.model_params = model_params

    def build(self):

        num_layers = self.model_params.num_layers
        self.layers = []
        for i in range(num_layers):
            self.layer = torch.nn.Sequential()
            # Fully-connected Layer
            if self.model_params.layer_type[i] == layer_type.FC.value:
                print("Hidden layer #%d: " % (i + 1))
                if self.model_params.layer_type[i - 1] == layer_type.FC.value:
                    if i == 0:
                        prev_num_neurons = self.model_params.input_size
                    else:
                        prev_num_neurons = self.model_params.output_size[i - 1]
                elif self.model_params.layer_type[i - 1] == layer_type.CNN.value:
                    if i == 0:
                        prev_num_neurons = self.model_params.input_size
                    else:
                        prev_num_neurons = self.model_params.output_size[i - 1] * (self.model_params.input_size // 16)

                num_neurons = self.model_params.output_size[i]
                fc = torch.nn.Linear(prev_num_neurons,
                                     num_neurons,
                                     bias=True)

                self.layer.add_module('FC', fc)

                # Initializer
                if self.model_params.init_fn[i] == init_fn_type.Normal.value:
                    torch.nn.init.normal(fc.weight)
                elif self.model_params.init_fn[i] == init_fn_type.Xavier.value:
                    torch.nn.init.xavier_uniform(fc.weight)
                else:
                    pass

                # Activation function
                if self.model_params.activate_fn[i] != activate_fn_type.No_act.value:
                    if self.model_params.activate_fn[i] == activate_fn_type.Sigmoid.value:
                        act_fn = torch.nn.Sigmoid()
                    elif self.model_params.activate_fn[i] == activate_fn_type.tanh.value:
                        act_fn = torch.nn.Tanh()
                    elif self.model_params.activate_fn[i] == activate_fn_type.ReLU.value:
                        act_fn = torch.nn.ReLU()
                    else:
                        pass
                    self.layer.add_module('activation', act_fn)
                else:
                    pass

                print("\tAdding FC layer...")
                print("\t\t# of neurons = %d" % num_neurons)
                # print("\t\tInput:", self.input.size(), "-> Output:", out.size())

            # Convolutional Layer
            elif self.model_params.layer_type[i] == layer_type.CNN.value:
                print("Hidden layer #%d:" % (i+1))
                if i == 0:
                    prev_num_filters = 1
                else:
                    prev_num_filters = self.model_params.output_size[i - 1]

                num_filters = self.model_params.output_size[i]
                k_size = self.model_params.kernel_size[i]
                s_size = self.model_params.kernel_stride[i]
                k_padding = (k_size-1)/2

                conv = torch.nn.Conv2d(prev_num_filters,
                                       num_filters,
                                       kernel_size=k_size,
                                       stride=s_size,
                                       padding=1)

                self.layer.add_module('CNN', conv)

                # Initializer
                # Initializer
                if self.model_params.init_fn[i] == init_fn_type.Uniform.value:
                    torch.nn.init.uniform(fc.weight)
                elif self.model_params.init_fn[i] == init_fn_type.Normal.value:
                    torch.nn.init.normal(conv.weight)
                elif self.model_params.init_fn[i] == init_fn_type.Xavier.value:
                    torch.nn.init.xavier_uniform(conv.weight)
                else:
                    pass

                # Activation function
                if self.model_params.activate_fn[i] != activate_fn_type.No_act.value:
                    if self.model_params.activate_fn[i] == activate_fn_type.Sigmoid.value:
                        act_fn = torch.nn.Sigmoid()
                    elif self.model_params.activate_fn[i] == activate_fn_type.tanh.value:
                        act_fn = torch.nn.Tanh()
                    elif self.model_params.activate_fn[i] == activate_fn_type.ReLU.value:
                        act_fn = torch.nn.ReLU()
                    else:
                        pass
                    self.layer.add_module('activation', act_fn)
                else:
                    pass

                print("\tAdding Conv2D layer...")
                print("\t\tKernel size = %dx%d, Stride = (%d, %d)" %(k_size, k_size, s_size, s_size))
                # print("\t\tInput:", self.input.size(), "-> Output:", out.size())

                # Pooling Layer
                p_size = self.model_params.pool_size[i]
                pool_s_size = self.model_params.pool_stride[i]

                if self.model_params.use_pooling[i]:
                    pooling = torch.nn.MaxPool2d(kernel_size=p_size,
                                                 stride=pool_s_size)

                    self.layer.add_module('pooling', pooling)

                    print("\tAdding MaxPooling layer...")
                    print("\t\tKernel size = %dx%d, Stride = (%d, %d)" %(p_size, p_size, pool_s_size, pool_s_size))
                    # print("\t\tInput:", self.input.size(), "-> Output:", out.size())

            # Dropout
            keep_prob = self.model_params.keep_prob[i]
            if self.model_params.use_dropout[i]:
                dropout = torch.nn.Dropout(p=1-keep_prob)

                self.layer.add_module('dropout', dropout)

                print("\tAdding Dropout layer...")
                print("\t\tkeep_prob = %0.1f" % keep_prob)

            # Append hidden layer
            self.layers.append(self.layer)

        # Output (no activation) Layer
        print("Output layer: ")
        if self.model_params.layer_type[num_layers-1] == layer_type.FC.value:
            prev_num_neurons = self.model_params.output_size[num_layers-1]
        elif self.model_params.layer_type[num_layers-1] == layer_type.CNN.value:
            prev_num_neurons = self.model_params.output_size[num_layers-1] * (self.model_params.input_size / 16)

        self.logits = torch.nn.Linear(prev_num_neurons,
                                      self.model_params.num_classes,
                                      bias=True)

        print("\t\tAdding FC layer...")
        # print("\t\tInput:", self.input.size(), "-> Output:", self.logits.size())

        return True

    def forward(self, x):
        # x = Variable(x)
        for i in range(self.model_params.num_layers):
            if self.model_params.layer_type[i] == layer_type.FC.value:
                if i == 0:
                    out = x.view(-1, self.model_params.input_size)
                elif self.model_params.layer_type[i - 1] == layer_type.CNN.value:
                    out = out.view(out.size(0), -1)  # Flatten them for FC

            elif self.model_params.layer_type[i] == layer_type.CNN.value:
                if i == 0:
                    out = x

            out = self.layers[i](out)

        if self.model_params.layer_type[i] == layer_type.CNN.value:
            out = out.view(out.size(0), -1)

        out = self.logits(out)
        return out

    def set_optimizer(self, train_params):
        self.train_params = train_params
        lossFn = self.train_params.loss_fn
        optimizer = self.train_params.optimizer
        learning_rate = self.train_params.learning_rate

        # define cost/loss & optimizer
        if lossFn == loss_fn_type.Cross_Entropy.value:
            print("\nLoss function: Cross_Entropy.")
            self.criterion = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.

        if optimizer == optimizer_type.SGD.value:
            print("\nSGD optimizer is selected.")
            self.optimizer = torch.optim.SGD(self.layer.parameters(), lr=learning_rate)
        elif optimizer == optimizer_type.Adam.value:
            print("\nAdam optimizer is selected.")
            self.optimizer = torch.optim.Adam(self.layer.parameters(), lr=learning_rate)
        elif optimizer == optimizer_type.RMSProp.value:
            print("\nRMSProp optimizer is selected.")
            self.optimizer = torch.optim.RMSprop(self.layer.parameters(), lr=learning_rate)
        else:
            pass

    def predict(self, image):
        self.eval()
        image = np.asarray(image, dtype="float32")
        # normalize image 0 to 1
        if image.max() == 255:
            image = image / 255

        image = 1.0 - image
        image = Variable(torch.Tensor(image))
        prediction = self.forward(image)
        return torch.max(prediction.data, 1)[1]

    def evaluate(self, test_data_set):
        self.eval()
        if self.model_params.layer_type[0] == layer_type.FC.value:
            X_test = Variable(test_data_set.test_data.view(-1, self.model_params.input_size).float())
        elif self.model_params.layer_type[0] == layer_type.CNN.value:
            X_test = Variable(test_data_set.test_data.view(len(test_data_set), 1, 28, 28).float())
        Y_test = Variable(test_data_set.test_labels)
        prediction = self.forward(X_test)
        correct_prediction = (torch.max(prediction.data, 1)[1] == Y_test.data)
        self.accuracy = correct_prediction.float().mean()
        return self.accuracy

    def train_batch(self, train_params, dataset):
        self.train_params = train_params
        self.avg_cost = 0
        self.avg_accu = 0
        # data_loader = self.load_batch_data(dataset, self.train_params.batch_size)
        # dataset loader
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.train_params.batch_size,
                                                  shuffle=True)

        for step, (batch_xs, batch_ys) in enumerate(data_loader):
            self.train()
            X = Variable(batch_xs)
            Y = Variable(batch_ys)
            self.optimizer.zero_grad()
            hypothesis = self.forward(X)
            self.cost = self.criterion(hypothesis, Y)
            self.cost.backward()
            self.optimizer.step()

            # average cost and accuracy
            self.avg_cost += self.cost / self.train_params.num_batch_train

            correct_prediction = (torch.max(hypothesis.data, 1)[1] == Y.data)
            accu = correct_prediction.float().mean()
            self.avg_accu += accu / self.train_params.num_batch_train

        return self.avg_cost.data[0], self.avg_accu
    
    def visualize_model(self, X_test):
        out = self.forward(X_test)
        make_dot(out)



