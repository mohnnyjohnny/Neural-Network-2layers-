# coding=utf-8
import numpy as np
import struct
import os
import time
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully conneted layer with input %d, output %d.' % (self.num_input, self.num_output))

    def init_param(self, std=0.01):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, input):  # 前向传播
        start_time = time.time()
        self.input = input
        self.output = np.matmul(input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):  # 反向传播计算偏导，top_diff是损失函数对z(l)的偏导
        self.d_weight = np.dot(self.input.T, top_diff)  # W(l)的偏导
        self.d_bias = np.sum(top_diff, axis=0)  # b(l)的偏导
        bottom_diff = np.dot(top_diff, self.weight.T)  # a(l-1)的偏导
        return bottom_diff

    def update_param(self, lr, beta):  # 更新参数W,b; lr是学习率(步长); beta是权重衰减系数(一般取值较小，0.0005); 标准随机梯度下降中，权重衰减正则化与l2正则化效果相同
        self.weight = self.weight * (1 - beta) - lr * self.d_weight
        self.bias = self.bias * (1 - beta) - lr * self.d_bias

    def load_param(self, weight, bias):  # 加载参数
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def save_param(self):  # 参数保存
        return self.weight, self.bias


class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')

    def forward(self, input):
        start_time = time.time()
        self.input = input
        output = np.maximum(0, input)
        return output

    def backward(self, top_diff):
        bottom_diff = top_diff
        bottom_diff[self.input < 0] = 0
        return bottom_diff


class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')

    def forward(self, input):
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff


MNIST_DIR = "D:/孙墨瀚/硕士阶段/课程/研一下/神经网络和深度学习/mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


class MNIST_MLP(object):
    def __init__(self, batch_size=30, input_size=784, hidden1=256, out_classes=10, lr=0.001, beta=0.00001,
                 beta_decay=0.96, max_epoch=30, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.out_classes = out_classes
        self.lr = lr
        self.beta = beta
        self.beta_decay = beta_decay
        self.max_epoch = max_epoch
        self.print_iter = print_iter

    def load_mnist(self, file_dir, is_images='True'):
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        if is_images:
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
        return mat_data

    def load_data(self):  # 调用load_mnist 读取和预处理MNIST中的训练和测试数据的图像和标记
        print('Loading MNIST data from files...')
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)

    def shuffle_data(self):
        print('Randomly shuffle MNIST data...')
        np.random.shuffle(self.train_data)

    def build_model(self):
        print('Building two-layer perception model...')
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2]

    def init_model(self):
        print('Initializing parameters of each layer in MLP...')
        for layer in self.update_layer_list:
            layer.init_param()

    def load_model(self, param_dir):
        print('Loading parameters from file' + param_dir)
        params = np.load(param_dir).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])

    def save_model(self, param_dir):
        print('Saving parameters to file' + param_dir)
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        np.save(param_dir, params)
        return params

    def forward(self, input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        prob = self.softmax.forward(h2)
        return prob

    def backward(self):
        dloss = self.softmax.backward()
        dh2 = self.fc2.backward(dloss)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr, beta):
        for layer in self.update_layer_list:
            layer.update_param(lr, beta)

    def train(self):
        max_batch = self.train_data.shape[0] / self.batch_size
        print('Start training...')
        idx_batch_list = []
        loss_list = []
        accuracy_list = []
        val_loss_list = []
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            lr = self.lr
            for idx_batch in range(int(max_batch)):
                batch_images = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(lr, self.beta)
                # lr = lr * self.beta_decay ** (idx_batch + 1) # 学习率衰减，可以变换公式
                if idx_epoch == 0:
                    accuracy, val_loss, _ = self.evaluate()
                    idx_batch_list = np.append(idx_batch_list, idx_batch)
                    loss_list = np.append(loss_list, loss)
                    accuracy_list = np.append(accuracy_list, accuracy)
                    val_loss_list = np.append(val_loss_list, val_loss)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f, lr: %.6f' % (idx_epoch, idx_batch, loss, lr))
            if idx_epoch == 0:
                fig_loss = plt.figure(figsize=(7, 5))
                p1 = pl.plot(idx_batch_list, loss_list, 'r-', label=u'tra_loss')
                pl.legend()
                p2 = pl.plot(idx_batch_list, val_loss_list, 'b-', label=u'val_loss')
                pl.legend()
                pl.xlabel(u'iters')
                pl.ylabel(u'loss')
                plt.title('Compare loss between training set and validation set')
                fig_accuracy = plt.figure(figsize=(7, 5))
                pl.plot(idx_batch_list, accuracy_list, 'g-', label=u'val_acc')
                pl.legend()
                pl.xlabel(u'iters')
                pl.ylabel(u'accuracy')
                plt.title('the accuracy in validation set')

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        start_time = time.time()
        batch_images = self.test_data[:, :-1]
        batch_labels = self.test_data[:, -1]
        prob = self.forward(batch_images)
        val_loss = self.softmax.get_loss(batch_labels)
        end = time.time()
        pred_labels = np.argmax(prob, axis=1)
        pred_results[:] = pred_labels
        # print("All evaluate time: %f" % (time.time() - start_time))
        evaluate_time = end - start_time
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        # print('Accuracy in test set: %f' % accuracy)
        return accuracy, val_loss, evaluate_time


def build_mnist_mlp(param_dir='weight.npy'):
    h1, e = 256, 30
    mlp = MNIST_MLP(hidden1=h1, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    mlp.train()
    params = mlp.save_model(os.path.join(MNIST_DIR, 'weight.npy'))
    return mlp, params


if __name__ == '__main__':
    mlp, params = build_mnist_mlp()
    accuracy, _, evaluate_time = mlp.evaluate()
    print("All evaluate time: %f" % evaluate_time)
    print('Accuracy in test set: %f' % accuracy)

    # 可视化网络参数
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']
    fig_w1 = plt.figure(figsize=(5, 5))
    plt.imshow(w1, cmap='Greys', interpolation=None)
    plt.title('w1')
    fig_w2 = plt.figure(figsize=(5, 5))
    plt.imshow(w2, cmap='Greys', interpolation=None)
    plt.title('w2')
    fig_b1 = plt.figure(figsize=(10, 10))
    plt.imshow(b1, cmap='Greys', interpolation=None)
    plt.title('b1')
    fig_b2 = plt.figure(figsize=(5, 5))
    plt.imshow(b2, cmap='Greys', interpolation=None)
    plt.title('b2')
