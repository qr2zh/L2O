import torch
import torchvision
from torchvision import datasets
from utils import *

class QuadraticLoss:
    def __init__(self, **kwargs):
        self.W = to_device(torch.randn(10, 10))
        self.y = to_device(torch.randn(10))
    
    def get_loss(self, theta):
        return torch.sum((self.W.matmul(theta) - self.y)**2)

class MNISTLoss:
    def __init__(self, training=True):
        dataset = datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
        indices = list(range(len(dataset)))
        if training:
            indices = indices[: len(indices) // 2]
        else:
            indices = indices[len(indices) // 2 :]
        
        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, 
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices)
        )

        self.batches = []
        self.cur_batch = 0
    
    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch

class CifarLoss:
    def __init__(self, training=True):
        trainset = datasets.CIFAR10('./data', train=True, download=False, transform=torchvision.transforms.ToTensor())
        testset = datasets.CIFAR10('./data', train=False, download=False, transform=torchvision.transforms.ToTensor())
        if training:
            indices = list(range(len(trainset)))
            self.loader = torch.utils.data.DataLoader(
                trainset, batch_size=128,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices)
            )
        else:
            indices = list(range(len(testset)))
            self.loader = torch.utils.data.DataLoader(
                testset, batch_size=128,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices)
            )

        self.batches = []
        self.cur_batch = 0
    
    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch