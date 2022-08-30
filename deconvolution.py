"""
    1 - 4) optimization 1 - 3 ; fine tune
    5 - 8) loss 1 - 3 ; fine tune loss

"""
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim

import info_log

def deconvolution_handler(X_sc, X_bulk, X_dropout, TRS, cluster_lists_of_idx, args, param):
    info_log.print('--------> Starting Deconvolution ...')

    # Transpose all cell * gene matrices to gene * cell matrices
    X_sc = X_sc.T
    X_bulk = X_bulk.T
    X_dropout = X_dropout.T
    TRS = TRS.T
    X_sc[X_dropout != 0] = X_dropout[X_dropout != 0]

    weights_0 = pre_deconvolution_helper(cluster_lists_of_idx, param)

    X_bulk_avg = average_with_mask(X_bulk) # Changed to normal average

    info_log.print('----------------> Starting Optimization 1 ...')
    weights = optimization_1(
        X_sc_ct_avg_TRS = average_by_ct(X_sc, cluster_lists_of_idx, mask=TRS==0), # ? ways to determine high expression gene
        X_bulk_avg = X_bulk_avg, 
        weights_0 = weights_0, 
        args = args, param = param) # ct * 1

    info_log.print('----------------> Starting Optimization 2 ...')
    X_ct_avg = optimization_2(
        X_sc_ct_avg_pos = average_by_ct(X_sc, cluster_lists_of_idx),
        X_bulk_avg = X_bulk_avg, 
        weights = weights, 
        args = args, param = param) # gene * ct

    info_log.print('----------------> Starting Optimization 3 ...')
    X_deconvoluted_unadjusted = optimization_3(
        X_ct_avg = X_ct_avg, 
        X_sc = X_sc, 
        X_dropout = X_dropout,
        clusterIndexList = cluster_lists_of_idx, 
        args = args, param = param) # gene * cell

    info_log.print('----------------> Start Fine-Tunning ...')
    tunning_weights = fine_tune(
        X_bulk_avg = X_bulk_avg, 
        X_deconvoluted_ct_avg = average_by_ct(X_deconvoluted_unadjusted, cluster_lists_of_idx),
        weights = weights, 
        args = args, param = param)  
    
    X_deconvoluted = tunning_weights * X_deconvoluted_unadjusted # gene * cell

    param['X_ct_avg'] = X_ct_avg.T # ct * gene
    param['X_deconvoluted_unadjusted'] = X_deconvoluted_unadjusted.T # cell * gene

    return X_deconvoluted.T # cell * gene

def pre_deconvolution_helper(clusterIndexList, param):

    weights_0 = np.ones((len(clusterIndexList), 1))
    population = 0

    for ct, clust in enumerate(clusterIndexList):
        weights_0[ct] = len(clust)
        population += len(clust)

    weights_0 /= population # weights_0 contains the population proportion of each cell type [ct * ]
    # weights_0 = torch.from_numpy(weights_0).type(torch.FloatTensor).to(param['device'])

    return weights_0

def average_with_mask(X, mask=None, axis = -1, keepdims = True):
    if mask is not None:
        return np.ma.array(X, mask = mask).mean(axis = axis, keepdims = keepdims)
    else:
        return X.mean(axis = axis, keepdims = keepdims)

def average_by_ct(X, clusterIndexList, mask=None):
    X_ct_avg = []
    for ct_idx in clusterIndexList: 
        X_ct_avg_i = X[:,ct_idx].mean(axis=-1,keepdims=True) if mask is None else average_with_mask(X[:,ct_idx], mask = mask[:,ct_idx]) # [gene * cell of `ct` type] -> [gene * 1]
        X_ct_avg.append(X_ct_avg_i) # [gene * total # of cell types (aka # of cell clusters)]
    X_ct_avg = np.hstack(X_ct_avg) # [gene * ct]

    return X_ct_avg

def loss_1(weights, X_sc_ct_avg_TRS, X_bulk_avg, weights_0, eta=1e-2):
    loss_a = X_bulk_avg - X_sc_ct_avg_TRS @ weights # [gene * 1] -  [gene * ct] @ [ct * 1] -> [gene * 1]
    loss_b = weights - weights_0 # [ct * 1]
    loss = torch.norm(loss_a) + eta * torch.norm(loss_b)
    return loss

def optimization_1(X_sc_ct_avg_TRS, X_bulk_avg, weights_0, args, param): 

    learning_rate = args.deconv_opt1_learning_rate
    max_epoch = args.deconv_opt1_epoch
    epsilon = args.deconv_opt1_epsilon
    regu_strength = args.deconv_opt1_regu_strength
    output_dir = args.output_dir

    X_sc_ct_avg_TRS = torch.from_numpy(X_sc_ct_avg_TRS).type(torch.FloatTensor).to(param['device']) # [gene * ct]
    X_bulk_avg = torch.from_numpy(X_bulk_avg).type(torch.FloatTensor).to(param['device']) # [gene * 1]
    
    weights_0 = torch.from_numpy(weights_0).type(torch.FloatTensor).to(param['device']) # [ct * 1]
    weights = torch.randn_like(weights_0).type(torch.FloatTensor).to(param['device']).requires_grad_()  # random initialization [ct * 1]
    
    weights = train(weights, loss_1, 
                    {
                        'X_sc_ct_avg_TRS': X_sc_ct_avg_TRS,
                        'X_bulk_avg': X_bulk_avg,
                        'weights_0': weights_0,
                        'eta': regu_strength
                    },
                    var_min=0, learning_rate=learning_rate, maxepochs=max_epoch, epsilon=epsilon, output_dir=output_dir)
    
    del X_sc_ct_avg_TRS
    del X_bulk_avg
    del weights_0
    torch.cuda.empty_cache()
    
    return weights

def loss_2(X_sc_ct_avg, X_sc_ct_avg_pos, X_bulk_avg, weights, eta=1e-2):

    loss_a = X_bulk_avg - X_sc_ct_avg @ weights # [gene * 1] -  [gene * ct] * [ct * 1] -> [gene * 1]
    loss_b = X_sc_ct_avg - X_sc_ct_avg_pos # [gene * ct]
    loss = torch.norm(loss_a) + eta * torch.sum(torch.norm(loss_b, dim=0)) # maybe just torch.norm(loss_b)
    return loss

def optimization_2(X_sc_ct_avg_pos, X_bulk_avg, weights, args, param): 

    learning_rate = args.deconv_opt2_learning_rate
    max_epoch = args.deconv_opt2_epoch
    epsilon = args.deconv_opt2_epsilon
    regu_strength = args.deconv_opt2_regu_strength
    output_dir = args.output_dir
    
    X_bulk_avg = torch.from_numpy(X_bulk_avg).type(torch.FloatTensor).to(param['device']) # [gene * 1]
    weights = torch.from_numpy(weights).type(torch.FloatTensor).to(param['device']) # [ct * 1]

    X_sc_ct_avg_pos = torch.from_numpy(X_sc_ct_avg_pos).type(torch.FloatTensor).to(param['device']) # [gene * ct]
    X_sc_ct_avg = torch.randn_like(X_sc_ct_avg_pos).type(torch.FloatTensor).to(param['device']).requires_grad_()  # random initialization [gene * ct]
    
    X_sc_ct_avg = train(X_sc_ct_avg, loss_2, 
                    {
                        'X_sc_ct_avg_pos': X_sc_ct_avg_pos,
                        'X_bulk_avg': X_bulk_avg,
                        'weights': weights,
                        'eta': regu_strength
                    },
                    var_min=0, learning_rate=learning_rate, maxepochs=max_epoch, epsilon=epsilon, output_dir=output_dir)
                    
    del X_bulk_avg
    del weights
    del X_sc_ct_avg_pos
    torch.cuda.empty_cache()

    return X_sc_ct_avg

def loss_3(X_deconvoluted_ct, X_sc_ct, X_dropout_ct, X_ct_avg_i, eta_1=1e-2, eta_2=1e-2, eta_3=1e-2):

    loss_a = X_ct_avg_i - X_deconvoluted_ct.mean(dim=-1, keepdim=True) # [gene * 1]
    loss_b = (X_sc_ct - X_deconvoluted_ct)[X_dropout_ct == 0] # ~[gene * cell in ct i]
    # loss_c = (X_dropout_ct - X_deconvoluted_ct)[X_dropout_ct.nonzero()] # ~[gene * cell in ct i]
    loss = torch.norm(loss_a) + \
        eta_1 * torch.sum(torch.norm(loss_b, dim=0))  + \
        eta_2 * torch.norm(X_deconvoluted_ct, p='nuc') #+ \
        # eta_3 * torch.norm(loss_c)
    return loss

def optimization_3(X_ct_avg, X_sc, X_dropout, clusterIndexList, args, param): 

    learning_rate = args.deconv_opt3_learning_rate
    max_epoch = args.deconv_opt3_epoch
    epsilon = args.deconv_opt3_epsilon
    regu_strength_1 = args.deconv_opt3_regu_strength_1
    regu_strength_2 = args.deconv_opt3_regu_strength_2
    regu_strength_3 = args.deconv_opt3_regu_strength_3
    output_dir = args.output_dir
    
    X_ct_avg = torch.from_numpy(X_ct_avg).type(torch.FloatTensor) # [gene * ct]

    X_sc = torch.from_numpy(X_sc).type(torch.FloatTensor) # [gene * cell]
    X_dropout = torch.from_numpy(X_dropout).type(torch.FloatTensor) # [gene * cell]
    X_deconvoluted = torch.randn_like(X_sc).type(torch.FloatTensor)  # random initialization [gene * cell]
    X_deconvoluted[X_dropout != 0] = X_dropout[X_dropout != 0]

    for i, ct_idx in enumerate(clusterIndexList):

        X_deconvoluted_ct = X_deconvoluted[:,ct_idx].to(param['device']).requires_grad_()
        X_sc_ct = X_sc[:,ct_idx].to(param['device'])
        X_dropout_ct = X_dropout[:,ct_idx].to(param['device'])
        X_ct_avg_i = X_ct_avg[:,i].to(param['device'])

        X_deconvoluted_ct = train(X_deconvoluted_ct, loss_3, 
                    {
                        'X_sc_ct': X_sc_ct,
                        'X_dropout_ct': X_dropout_ct,
                        'X_ct_avg_i': X_ct_avg_i,
                        'eta_1': regu_strength_1,
                        'eta_2': regu_strength_2,
                        'eta_3': regu_strength_3
                    },
                    var_min=0, learning_rate=learning_rate, maxepochs=max_epoch, epsilon=epsilon, idx=i, output_dir=output_dir)
        
        with torch.no_grad():
            X_deconvoluted[:,ct_idx] = torch.from_numpy(X_deconvoluted_ct[:,np.arange(len(ct_idx))])
        
        del X_deconvoluted_ct
        del X_sc_ct
        del X_dropout_ct
        del X_ct_avg_i
        torch.cuda.empty_cache()

    return X_deconvoluted.detach().numpy()

def fine_tune_loss(tunning_weights, X_bulk_avg, X_deconvoluted_ct_avg, weights, ones_tensor):

    loss_a = X_bulk_avg - tunning_weights * (X_deconvoluted_ct_avg @ weights)
    loss_b = tunning_weights - ones_tensor # [gene * 1]
    loss = torch.norm(loss_a) + torch.norm(loss_b)
    return loss

def fine_tune(X_bulk_avg, X_deconvoluted_ct_avg, weights, args, param): 

    learning_rate = args.deconv_tune_learning_rate
    max_epoch = args.deconv_tune_epoch
    epsilon = args.deconv_tune_epsilon
    output_dir = args.output_dir

    X_deconvoluted_ct_avg = torch.from_numpy(X_deconvoluted_ct_avg).type(torch.FloatTensor).to(param['device']) # [gene * ct]
    weights = torch.from_numpy(weights).type(torch.FloatTensor).to(param['device']) # [ct * 1]

    X_bulk_avg = torch.from_numpy(X_bulk_avg).type(torch.FloatTensor).to(param['device']) # [gene * 1]
    tunning_weights = torch.randn_like(X_bulk_avg).type(torch.FloatTensor).to(param['device']).requires_grad_() # [gene * 1]
    ones_tensor = torch.ones_like(X_bulk_avg).type(torch.FloatTensor).to(param['device']) # [gene * 1]

    tunning_weights = train(tunning_weights, fine_tune_loss, 
                    {
                        'X_bulk_avg': X_bulk_avg,
                        'X_deconvoluted_ct_avg': X_deconvoluted_ct_avg,
                        'weights': weights,
                        'ones_tensor': ones_tensor
                    },
                    var_min=1, learning_rate=learning_rate, maxepochs=max_epoch, epsilon=epsilon, output_dir=output_dir)

    return tunning_weights

def train(var2opt_init, loss_func, kwargs, var_min=None, learning_rate=1e-3, maxepochs=100, epsilon=1e-4, idx=None, output_dir=None):
    
    var2opt = var2opt_init
    epochs_count = 0
    loss_list = []
    epochs_list = []

    optimizer = optim.Adam([var2opt], lr=learning_rate)

    while epochs_count < maxepochs:
        loss = loss_func(var2opt, **kwargs)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            var2opt = var2opt.clamp_(min=var_min)
        optimizer.zero_grad()

        loss_cpu = loss.item()
        if np.abs(loss_cpu) < epsilon:  # 终止条件
            break

        loss_list.append(loss_cpu)
        epochs_list.append(epochs_count)
        epochs_count += 1
    
    info_log.print(f"------------------------> {'' if idx is None else 'Cell type ' + str(idx+1) + ' '}Ran {epochs_count} epochs")
    plt.plot(epochs_list, loss_list)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(f"{os.path.join(output_dir, loss_func.__name__)}{'' if idx is None else '_ct_' + str(idx+1)}.png")
    plt.clf()

    return var2opt.detach().cpu().numpy()