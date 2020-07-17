import torch
import torch.nn as nn
import numpy as np
from meta_module import *
from utils import *

class Optimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
    
    def forward(self, inp, hidden, cell):
        if self.preproc:
            inp = inp.data
            inp2 = to_device(torch.zeros(inp.size()[0], 2))
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()
            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = to_device(Variable(inp2))
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)

class OptimizerWithModulation(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)

        self.m_emb = nn.Embedding(2, 2*hidden_sz)

        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
    
    def forward(self, inp, id, hidden, cell):
        if self.preproc:
            inp = inp.data
            inp2 = to_device(torch.zeros(inp.size()[0], 2))
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()
            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = to_device(Variable(inp2))
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        embedding = self.m_emb(id)
        gammas, betas = torch.split(embedding, hidden1.size(1), dim=-1)
        gammas = gammas + torch.ones_like(gammas)
        hidden1 = hidden1 * gammas + betas
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)

class QuadOptimizee(MetaModule):
    def __init__(self, theta=None):
        super().__init__()
        self.register_buffer('theta', to_var(torch.zeros(10), requires_grad=True))
    
    def forward(self, target):
        return target.get_loss(self.theta)
    
    def all_named_parameters(self):
        return [('theta', self.theta)]

class MNISTNetLinear(MetaModule):
    def __init__(self, layer_size=20, n_layers=1, **kwargs):
        super().__init__()

        inp_size = 28*28
        self.layers = {}
        for i in range(n_layers):
            self.layers[f'mat_{i}'] = MetaLinear(inp_size, layer_size)
            inp_size = layer_size

        self.layers['final_mat'] = MetaLinear(inp_size, 10)
        self.layers = nn.ModuleDict(self.layers)

        self.activation = nn.Sigmoid()
        self.loss = nn.NLLLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]
    
    def forward(self, loss):
        inp, out = loss.sample()
        inp = to_device(Variable(inp.view(inp.size()[0], 28*28)))
        out = to_device(Variable(out))

        cur_layer = 0
        while f'mat_{cur_layer}' in self.layers:
            inp = self.activation(self.layers[f'mat_{cur_layer}'](inp))
            cur_layer += 1

        inp = F.log_softmax(self.layers['final_mat'](inp), dim=1)
        l = self.loss(inp, out)
        return l

class CifarNet(MetaModule):
    def __init__(self, n_out=10):
        super(CifarNet, self).__init__()

        self.layers = {}
        self.layers['conv1'] = MetaConv2d(3, 16, kernel_size=5, padding=2)
        self.layers['bn1'] = MetaBatchNorm2d(16)
        self.layers['conv2'] = MetaConv2d(16, 16, kernel_size=5, padding=2)
        self.layers['bn2'] = MetaBatchNorm2d(16)
        self.layers['conv3'] = MetaConv2d(16, 16, kernel_size=5, padding=2)
        self.layers['bn3'] = MetaBatchNorm2d(16)
        self.layers['fc1'] = MetaLinear(256, 32)
        self.layers['bn4'] = MetaBatchNorm1d(32)
        self.layers['fc2'] = MetaLinear(32, n_out)
        self.layers = nn.ModuleDict(self.layers)

        self.activation = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.loss = nn.NLLLoss()
    
    def all_named_parameters(self):
        return [(k, v) for k,v in self.named_parameters()]
    
    def forward(self, loss):
        inp, out = loss.sample()
        inp, out = to_device(Variable(inp)), to_device(Variable(out))

        inp = self.pooling(self.activation(self.layers['bn1'](self.layers['conv1'](inp))))
        inp = self.pooling(self.activation(self.layers['bn2'](self.layers['conv2'](inp))))
        inp = self.pooling(self.activation(self.layers['bn3'](self.layers['conv3'](inp))))
        inp = inp.view(-1, 256)
        inp = self.activation(self.layers['bn4'](self.layers['fc1'](inp)))
        inp = self.layers['fc2'](inp).squeeze()
        inp = F.log_softmax(inp, dim=1)
        return self.loss(inp, out)
