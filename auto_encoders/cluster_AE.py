"""
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import util
import auto_encoders.model as model
import auto_encoders.train as train
import info_log

def cluster_AE_handler(X_recon, TRS, clusterIndexList, args, param, model_state):
    info_log.print('--------> Starting Cluster AE ...')

    batch_size = args.cluster_AE_batch_size
    total_epoch = args.cluster_AE_epoch
    learning_rate = args.cluster_AE_learning_rate
    regu_strength = args.cluster_AE_regu_strength

    # Initialize an empty matrix for storing the results
    reconNew = np.zeros_like(X_recon)
    reconNew = torch.from_numpy(reconNew).type(torch.FloatTensor).to(param['device'])

    TRS = torch.from_numpy(TRS).type(torch.FloatTensor)

    # checkpoint, X_loader = cluster_AE_state_dict[0], cluster_AE_state_dict[1]

    for i, clusterIndex in enumerate(clusterIndexList):
        info_log.print(f'----------------> Training cluster {i+1}/{len(clusterIndexList)} -> size = {len(clusterIndex)}')

        # Define separate models for each cell type, they should not share weights
        cluster_AE = model.Cluster_AE(dim=X_recon.shape[1]).to(param['device'])
        optimizer = optim.Adam(cluster_AE.parameters(), lr=learning_rate)
        
        # Load weights from Feature AE to save training time
        cluster_AE.load_state_dict(model_state['model'])
        # optimizer.load_state_dict(model_state['optimizer']) # Adam optimizer includes momentum, will turn this off for now

        reconUsage = X_recon[clusterIndex]
        scDataInter = util.ClusterDataset(reconUsage)
        X_loader = DataLoader(scDataInter, batch_size=batch_size, **param['dataloader_kwargs'])
        
        Cluster_orig, Cluster_embed, Cluster_recon = train.train_handler(
            model = cluster_AE,                        
            train_loader = X_loader, 
            optimizer = optimizer, 
            TRS = TRS, 
            total_epoch = total_epoch,
            regu_strength = regu_strength,
            param = param)

        for i, row in enumerate(clusterIndex):
            reconNew[row] = Cluster_recon[i, :]
        
        # AE_state = cluster_AE.state_dict()

        # empty cuda cache
        del Cluster_orig
        del Cluster_embed
        torch.cuda.empty_cache()

    # checkpoint = {
    #     'model': cluster_AE.state_dict(),
    #     'optimizer': optimizer.state_dict()
    # }

    return reconNew.detach().cpu().numpy() #, checkpoint # cell * gene
