import os
import argparse
import numpy as np
from torch import optim
from utils import to_device
import seaborn as sns
import matplotlib.pyplot as plt
from network import *
from problems import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='cifar', choices=['quadratic', 'mnist', 'cifar'], help='Type of problem')
    parser.add_argument('--preproc', action='store_false', help='Whether to preprocess the gradient')
    parser.add_argument('--unroll', type=int, default=20, help='Meta-optimizer unroll length')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of optimization steps per epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use_modulation', action='store_true', help='whether to use modulation module')
    parser.add_argument('--outmult', type=float, default=1.0, help='None')
    parser.add_argument('--n_tests', type=int, default=20, help='test number of each optimizer')
    args = parser.parse_args()

    if args.use_modulation:
        args.path = os.path.join('./experiment', '{}_modulate'.format(args.problem))
    else:
        args.path = os.path.join('./experiment', args.problem)

    if args.problem == 'quadratic':
        pass

    elif args.problem == 'mnist':
        pass

    elif args.problem == 'cifar':
        target_cls = CifarLoss
        target_to_opt = CifarNet

        if args.use_modulation:
            opt_net = [to_device(OptimizerWithModulation(preproc=args.preproc)) for _ in range(1)]
            for i in range(1):
                opt_net[i].load_state_dict(torch.load('{0}/opt_net_{1}.pth'.format(args.path, i)))
        else:
            opt_net = [to_device(Optimizer(preproc=args.preproc)) for _ in range(2)]
            for i in range(2):
                opt_net[i].load_state_dict(torch.load('{0}/opt_net_{1}.pth'.format(args.path, i)))
    
    NORMAL_OPTS = [(optim.Adam, {}), (optim.RMSprop, {}), (optim.SGD, {'momentum': 0.9}), (optim.SGD, {'nesterov': True, 'momentum': 0.9})]
    OPT_NAMES = ['ADAM', 'RMSprop', 'SGD', 'NAG']

    QUAD_LRS = [0.1, 0.03, 0.01, 0.01]
    fit_data = np.zeros((args.n_tests, 100, len(OPT_NAMES) + 1))
    for i, ((opt, extra_kwargs), lr) in enumerate(zip(NORMAL_OPTS, QUAD_LRS)):
        torch.random.manual_seed(0)
        torch.cuda.random.manual_seed(0)
        np.random.seed(0)
        fit_data[:, :, i] = np.array(fit_normal(target_cls, target_to_opt, opt, n_tests=args.n_tests, num_steps=args.num_steps, lr=lr, **extra_kwargs))

    torch.random.manual_seed(0)
    torch.cuda.random.manual_seed(0)
    np.random.seed(0)   
    fit_data[:, :, -1] = np.array([do_fit(opt_net, None, target_cls, target_to_opt, unroll=1, num_steps=args.num_steps, outmult=args.outmult, should_train=False) for _ in range(args.n_tests)])
    
    with open(os.path.join(args.path, '{}.txt'.format(args.problem)), 'w') as file:
        file.write('# Array shape: {0}\n'.format(fit_data.shape))
        for dataslice in fit_data:
            np.savetxt(file, dataslice)
            file.write('# New slice\n')
    
    sns.set(color_codes=True)
    sns.set_style('white')
    ax = sns.tsplot(data=fit_data[:, :, :], condition=OPT_NAMES + ['LSTM'], linestyle='--', color=['r', 'g', 'b', 'k', 'y'])
    ax.lines[-1].set_linestyle('-.')
    ax.legend()
    plt.yscale('log')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title(args.problem)
    plt.savefig(os.path.join(args.path, '{}.png'.format(args.problem)), dpi=600)
    print('done')