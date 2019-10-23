import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)
        self.output = None

    def forward(self, input):
        '''Your codes here'''
        out = np.maximum(input, 0)
        self.output = out
        '''np.maximum, Compare two arrays and returns a new array 
        containing the element-wise maxima. '''
        return out

    def backward(self, grad_output):
        '''Your codes here'''
        grad_output[self.output <= 0] = 0
        print(self.name)
        print (grad_output.shape)
        return grad_output



class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)
        self.output = 0

    def forward(self, input):
        '''Your codes here'''
        out = 1 / (1 + np.exp(-input))
        self.output = out
        return out

    def backward(self, grad_output):
        '''Your codes here'''
        print(self.name)
        print(grad_output.shape)
        out = grad_output * self.output * (1 - self.output)
        print(out.shape)
        return out


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.input_stored = 0
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)
        self.batch_size = 1

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        '''Your codes here'''
        self.input_stored = input
        out = np.dot(input, self.W) + self.b
        self.batch_size = input.shape[0]
        return out

    def backward(self, grad_output):
        '''Your codes here'''
        print(self.name)
        # if len(self.input_stored.shape) != 3:
        #     self.input_stored = self.input_stored.reshape(self.batch_size, self.in_num, 1)
        # input = np.broadcast_to(self.input_stored, (self.batch_size, self.in_num, self.out_num))
        # #input = self.input_stored[:, :, np.newaxis]
        # #grad_prev = grad_output[:, np.newaxis, :]
        # if len(grad_output.shape) != 3:
        #     grad_output = grad_output.reshape(self.batch_size, 1, self.out_num)
        # grad_prev = np.broadcast_to(grad_output, (self.batch_size, self.in_num, self.out_num))
        # print(self.name)
        # print("input")
        # print(input.shape)
        # print("grad_prev")
        # print(grad_prev.shape)
        # z = np.dot(grad_prev, self.W.transpose())
        # print(z.shape)
        # grad_W = np.sum(input * grad_prev, 0)
        # grad_b = np.sum(grad_output, 0)
        # self.grad_b = grad_b / self.batch_size
        # self.grad_W = grad_W / self.batch_size
        # return z

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
