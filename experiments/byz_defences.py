import numpy as np
import torch
import torch.optim as optim 
from torch.nn import functional as F


def defence_update(net,global_update):

    idx = 0
    # optimizer.zero_grad()
    for j, (param) in enumerate(net.parameters()):
        # param.requires_grad = False
        param_size = 1

        for i in range(len(param.data.size())):
            param_size *= param.data.size()[i]
        # param.set_(param.data - lr * global_update[idx: idx + param_size].reshape(param.data.shape))
        # param.set_()

        with torch.no_grad():
            param.grad = global_update[idx: idx + param_size].reshape(param.data.shape)
        idx += param_size

def fltrust_agg(param_list, grad_ag_list=None, add_ag=False, t=0):
    baseline = param_list[-1].squeeze()
    n = len(param_list) - 1
    if add_ag:
        if len(grad_ag_list) == 0:
            grad_ag_list = param_list
        else:
            for i in range(len(grad_ag_list)):
                alpha = 0.9
                # grad_ag_list[i] = grad_ag_list[i] * alpha + param_list[i] * (1 - alpha)
                grad_ag_list[i] = grad_ag_list[i] * (t / (t + 1)) + (param_list[i]) * (1 / (t + 1))
                # grad_ag_list[i] = grad_ag_list[i] * (t / (t + 1)) + (param_list[i]/(torch.norm(param_list[i]) + 1e-8)) * (1 / (t + 1))
                # grad_ag_list[i] = param_list[i]
    if add_ag:
        cos_sim = fltrust_cos_sim(grad_ag_list)
    else:
        cos_sim = fltrust_cos_sim(param_list)
    cos_sim = torch.stack(cos_sim)[:-1]
    cos_sim = F.relu(cos_sim)
    normalized_weights = cos_sim / (torch.sum(cos_sim) + 1e-8)  # weighted trust score

    # normalize the magnitudes and weight by the trust score
    new_param_list = []
    for i in range(n):
        new_param_list.append(
            (param_list[i] * normalized_weights[i] / (torch.norm(param_list[i]) + 1e-8)) * torch.norm(baseline))
    global_update = torch.sum(torch.cat(new_param_list, dim=1), axis=-1)

    return global_update, grad_ag_list

def fltrust_cos_sim(param_list):
    # use the last gradient (server update) as the trusted source
    baseline = param_list[-1].squeeze()
    cos_sim = []

    # compute cos similarity
    for each_param_list in param_list:
        # print("compute cos similarity")
        # print(each_param_list)
        each_param_array = each_param_list.squeeze()
        cos_sim.append(torch.dot(baseline, each_param_array) / (torch.norm(baseline) + 1e-9) / (
                torch.norm(each_param_array) + 1e-9))
    return cos_sim