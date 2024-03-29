"""

"""
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import optim

import util
import auto_encoders.model as model
import auto_encoders.train as train
import info_log

def feature_AE_handler(X, TRS, args, param, model_state=None):
    info_log.print('--------> Starting Feature AE ...')

    batch_size = args.feature_AE_batch_size
    total_epoch = args.feature_AE_epoch[param['epoch_num'] > 0]
    learning_rate = args.feature_AE_learning_rate
    regu_strength = args.feature_AE_regu_strength
    masked_prob = args.feature_AE_dropout_prob
    concat_prev_embed = args.feature_AE_concat_prev_embed

    if concat_prev_embed and param['epoch_num'] > 0:
        if concat_prev_embed == 'graph':
            prev_embed = util.normalizer(param['graph_embed'], base=X, axis=0)
        elif concat_prev_embed == 'feature':
            prev_embed = param['feature_embed']
        else:
            info_log.print('--------> feature_AE_concat_prev_embed argument not recognized, not using any previous embed ...')
            prev_embed = None
        X = np.concatenate((X, prev_embed), axis=1)

    X_dataset = util.ExpressionDataset(X)
    X_loader = DataLoader(X_dataset, batch_size=batch_size, **param['dataloader_kwargs'])
    TRS = torch.from_numpy(TRS).type(torch.FloatTensor)

    feature_AE = model.Feature_AE(dim=X.shape[1]).to(param['device'])
    optimizer = optim.Adam(feature_AE.parameters(), lr=learning_rate)

    impute_regu = None
    if param['epoch_num'] > 0:
        # Load Graph and Celltype Regu
        adjdense, celltypesample = param['impute_regu']
        adjsample = torch.from_numpy(adjdense).type(torch.FloatTensor).to(param['device'])
        celltypesample = torch.from_numpy(celltypesample).type(torch.FloatTensor).to(param['device'])
        # Load x_dropout as regu
        x_dropout = torch.from_numpy(param['x_dropout']).type(torch.FloatTensor).to(param['device'])
        impute_regu = {
            'graph_regu': adjsample,
            'celltype_regu': celltypesample,
            'x_dropout': x_dropout
        }
        
        if concat_prev_embed and param['epoch_num'] > 1:
            feature_AE.load_state_dict(model_state['model_concat'])
            # optimizer.load_state_dict(model_state['optimizer_concat']) # Adam optimizer includes momentum, will turn this off for now
        elif not concat_prev_embed:
            feature_AE.load_state_dict(model_state['model'])
            # optimizer.load_state_dict(model_state['optimizer']) # Adam optimizer includes momentum, will turn this off for now

    _, X_embed, X_recon = train.train_handler(
        model = feature_AE, 
        train_loader = X_loader, 
        optimizer = optimizer, 
        TRS = TRS, 
        total_epoch = total_epoch,
        impute_regu = impute_regu,
        regu_type = ['LTMG', 'noregu'],
        regu_strength = regu_strength,
        masked_prob = masked_prob,
        param = param)

    checkpoint = {
        'model_concat': feature_AE.state_dict(),
        'optimizer_concat': optimizer.state_dict(),
        'model': model_state['model'],
        'optimizer': model_state['optimizer']
    } if concat_prev_embed and param['epoch_num'] > 0 else {
        'model': feature_AE.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    X_recon_out = X_recon.detach().cpu().numpy()
    X_recon_out = X_recon_out[:,:param['n_feature_orig']]

    return X_embed.detach().cpu().numpy(), X_recon_out, checkpoint # cell * {gene, embedding}
