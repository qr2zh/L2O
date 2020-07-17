import torch
import functools
import numpy as np
from torch.autograd import Variable

USE_CUDA = False
if torch.cuda.is_available():
    USE_CUDA = True

def to_device(v):
    if USE_CUDA:
        return v.cuda()
    return v

def to_var(v, requires_grad=True):
    if USE_CUDA:
        v = v.cuda()
    return Variable(v, requires_grad=requires_grad)

def detach_var(v):
    var = to_device(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, num_steps, outmult, should_train=True):
    if should_train:
        for net, meta_net in zip(opt_net, meta_opt):
            net.train()
            meta_net.zero_grad()
    else:
        for net in opt_net:
            net.eval()
        unroll = 1

    # initialize dataset and net_to_opt
    target = target_cls(training=should_train)
    optimizee = to_device(target_to_opt())

    len_opt_net = len(opt_net)
    n_params = [0 for _ in range(len_opt_net)]
    if len_opt_net > 1:    # 需要学习多个优化器时，根据网络层的种类判断该层属于哪个元优化器
        for name, p in optimizee.all_named_parameters():
            if 'conv' in name and 'weight' in name:
                n_params[0] += int(np.prod(p.size()))
            else:
                n_params[1] += int(np.prod(p.size()))
    else:
        for name, p in optimizee.all_named_parameters():
            n_params[0] += int(np.prod(p.size()))
    
    hidden_states = [[to_device(Variable(torch.zeros(n_params[i], opt_net[i].hidden_sz))) for _ in range(2)] for i in range(len_opt_net)]
    cell_states = [[to_device(Variable(torch.zeros(n_params[i], opt_net[i].hidden_sz))) for _ in range(2)] for i in range(len_opt_net)]

    all_losses_ever = []
    all_losses = None

    for step in range(1, num_steps+1):
        loss = optimizee(target)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        
        all_losses_ever.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=should_train)

        offset = [0 for _ in range(len_opt_net)]
        result_params = {}
        hidden_states_temp = [[to_device(Variable(torch.zeros(n_params[i], opt_net[i].hidden_sz))) for _ in range(2)] for i in range(len_opt_net)]
        cell_states_temp = [[to_device(Variable(torch.zeros(n_params[i], opt_net[i].hidden_sz))) for _ in range(2)] for i in range(len_opt_net)]

        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            gradients = detach_var(p.grad.view(cur_sz, 1))
            if len_opt_net > 1:
                if 'conv' in name and 'weight' in name:
                    updates, new_hidden, new_cell = opt_net[0](
                        gradients,
                        [h[offset[0]:offset[0]+cur_sz] for h in hidden_states[0]],
                        [c[offset[0]:offset[0]+cur_sz] for c in cell_states[0]]
                    )
                    for i in range(len(new_hidden)):
                        hidden_states_temp[0][i][offset[0]:offset[0]+cur_sz] = new_hidden[i]
                        cell_states_temp[0][i][offset[0]:offset[0]+cur_sz] = new_cell[i]
                    offset[0] += cur_sz
                else:
                    updates, new_hidden, new_cell = opt_net[1](
                        gradients,
                        [h[offset[1]:offset[1]+cur_sz] for h in hidden_states[1]],
                        [c[offset[1]:offset[1]+cur_sz] for c in cell_states[1]]
                    )
                    for i in range(len(new_hidden)):
                        hidden_states_temp[1][i][offset[1]:offset[1]+cur_sz] = new_hidden[i]
                        cell_states_temp[1][i][offset[1]:offset[1]+cur_sz] = new_cell[i]
                    offset[1] += cur_sz
                
            else:
                if opt_net[0].__class__.__name__ == 'Optimizer':
                    updates, new_hidden, new_cell = opt_net[0](
                        gradients,
                        [h[offset[0]:offset[0]+cur_sz] for h in hidden_states[0]],
                        [c[offset[0]:offset[0]+cur_sz] for c in cell_states[0]]
                    )

                else:
                    id = None
                    if 'conv' in name and 'weight' in name:
                        id = to_device(torch.LongTensor([0]))
                    else:
                        id = to_device(torch.LongTensor([1]))
                    
                    updates, new_hidden, new_cell = opt_net[0](
                        gradients,
                        id,
                        [h[offset[0]:offset[0]+cur_sz] for h in hidden_states[0]],
                        [c[offset[0]:offset[0]+cur_sz] for c in cell_states[0]]
                    )
                
                for i in range(len(new_hidden)):
                    hidden_states_temp[0][i][offset[0]:offset[0]+cur_sz] = new_hidden[i]
                    cell_states_temp[0][i][offset[0]:offset[0]+cur_sz] = new_cell[i]
                offset[0] += cur_sz
            result_params[name] = p + updates.view(*p.size()) * outmult
            result_params[name].retain_grad()

        if step % unroll == 0:
            if should_train:
                for meta_net in meta_opt:
                    meta_net.zero_grad()
                all_losses.backward()
                for meta_net in meta_opt:
                    meta_net.step()
            
            for name, p in optimizee.named_buffers():
                if 'running_mean' in name or 'running_var' in name:
                    result_params[name] = p
            
            all_losses = None

            optimizee = to_device(target_to_opt())
            optimizee.load_state_dict(result_params)
            optimizee.zero_grad()
            for i in range(len_opt_net):
                hidden_states[i] = [detach_var(v) for v in hidden_states_temp[i]]
                cell_states[i] = [detach_var(v) for v in cell_states_temp[i]]
        else:
            for name, p in optimizee.all_named_parameters():
                rsetattr(optimizee, name, result_params[name])
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states_temp
            cell_states = cell_states_temp
    
    return all_losses_ever

def fit_normal(target_cls, target_to_opt, opt_class, n_tests=20, num_steps=100, lr=0.01, **kwargs):
    results = []
    for i in range(n_tests):
        target = target_cls(training=False)
        optimizee = to_device(target_to_opt())

        optimizer = opt_class(optimizee.parameters(), lr=lr, **kwargs)
        total_loss = []
        for _ in range(num_steps):
            loss = optimizee(target)
            total_loss.append(loss.data.cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results.append(total_loss)
    return results
