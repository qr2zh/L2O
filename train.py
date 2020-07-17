import os
import copy
import torch
import argparse
import numpy as np
import torch.optim as optim
from utils import to_device
from network import *
from problems import *
# from meta_module import *


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='quad', help='Path for saved meta-optimizer')
    parser.add_argument('--problem', type=str, default='quadratic', choices=['quadratic', 'mnist', 'cifar'], help='Type of problem')
    parser.add_argument('--preproc', action='store_false', help='Whether to preprocess the gradient')
    parser.add_argument('--unroll', type=int, default=20, help='Meta-optimizer unroll length')
    parser.add_argument('--num_epochs', type=int, default=600, help='Number of training epochs')
    parser.add_argument('--evaluation_period', type=int, default=20, help='Evaluation period')
    parser.add_argument('--evaluation_epochs', type=int, default=2, help='Number of evaluation epochs')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of optimization steps per epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use_modulation', action='store_true', help='whether to use modulation module')
    parser.add_argument('--outmult', type=float, default=1.0, help='None')
    args = parser.parse_args()
    
    args.save_path = os.path.join('./experiment', args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.problem == 'quadratic':
        target_cls = QuadraticLoss
        target_to_opt = QuadOptimizee

        opt_net = [to_device(Optimizer(preproc=args.preproc)) for _ in range(1)]
        meta_opt = [optim.Adam(opt_net[i].parameters(), lr=args.lr) for i in range(1)]

    elif args.problem == 'mnist':
        target_cls = MNISTLoss
        target_to_opt = MNISTNetLinear

        opt_net = [to_device(Optimizer(preproc=args.preproc)) for _ in range(1)]
        meta_opt = [optim.Adam(opt_net[i].parameters(), lr=args.lr) for i in range(1)]

    elif args.problem == 'cifar':
        target_cls = CifarLoss
        target_to_opt = CifarNet

        if args.use_modulation:
            opt_net = [to_device(OptimizerWithModulation(preproc=args.preproc)) for _ in range(1)]
            meta_opt = [optim.Adam(opt_net[i].parameters(), lr=args.lr) for i in range(1)]
        else:
            opt_net = [to_device(Optimizer(preproc=args.preproc)) for _ in range(2)]
            meta_opt = [optim.Adam(opt_net[i].parameters(), lr=args.lr) for i in range(2)]
    
    best_loss = 1000000
    for epoch in range(1, args.num_epochs+1):
        do_fit(opt_net, meta_opt, target_cls, target_to_opt, args.unroll, args.num_steps, args.outmult, should_train=True)

        if epoch % args.evaluation_period == 0:
            loss = (np.mean([
                np.sum(do_fit(opt_net, meta_opt, target_cls, target_to_opt, args.unroll, args.num_steps, args.outmult, should_train=False))
                for _ in range(args.evaluation_epochs)
                ]))
            print('epoch{:3d} : {}'.format(epoch, loss))
            if loss < best_loss:
                best_loss = loss
                # save model
                print('save model...')
                for i, net in enumerate(opt_net):
                    torch.save(copy.deepcopy(net.state_dict()), os.path.join(args.save_path, 'opt_net_{0}.pth'.format(i)))
