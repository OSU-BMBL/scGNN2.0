
import torch
from torch.nn import functional as F

import info_log

def train_handler(model, train_loader, optimizer, TRS, total_epoch, regu_strength, masked_prob, param, impute_regu=None):
    '''
    EMFlag indicates whether in EM processes.
        If in EM, use regulized-type parsed from program entrance,
        Otherwise, noregu
        taskType: celltype or imputation
    '''

    for epoch in range(total_epoch):
        model.train()
        train_loss = 0

        for batch_idx, (data, dataindex) in enumerate(train_loader): # data is Tensor of shape [batch * gene]

            # Send data and regulation matrix to device
            data = data.type(torch.FloatTensor).to(param['device'])
            data_masked = F.dropout(data, p=masked_prob)
            regulationMatrixBatch = TRS[dataindex, :].to(param['device'])

            optimizer.zero_grad()
            
            z, recon_batch = model.forward(data_masked) # reconstructed batch and encoding layer as outputs
            
            # Calculate loss
            if param['epoch_num'] == 0: # pre EM Feature AE
                loss = loss_function_graph(
                    recon_batch, 
                    data.view(-1, recon_batch.shape[1]), 
                    regulationMatrix = regulationMatrixBatch, 
                    regu_strength = regu_strength,
                    regularizer_type = 'LTMG')
            elif impute_regu is not None: # EM Imputation AE
                loss = loss_function_graph(
                    recon_batch, 
                    data.view(-1, recon_batch.shape[1]), 
                    regulationMatrix = impute_regu, 
                    regu_strength = regu_strength,
                    regularizer_type = 'Celltype')
            else:
                loss = loss_function_graph( # EM Feature AE
                    recon_batch, 
                    data.view(-1, recon_batch.shape[1]), 
                    regulationMatrix = regulationMatrixBatch, 
                    regu_strength = regu_strength,
                    regularizer_type = 'noregu')
                

            # Backprop and Update
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # Grow recon_batch, data, z at each epoch, while printing train loss
            if batch_idx == 0:
                recon_batch_all = recon_batch
                data_all = data
                z_all = z
            else:
                recon_batch_all = torch.cat((recon_batch_all, recon_batch), 0)
                data_all = torch.cat((data_all, data), 0)
                z_all = torch.cat((z_all, z), 0)

        info_log.interval_print(f'----------------> Epoch: {epoch+1}/{total_epoch}, Average loss: {train_loss / len(train_loader.dataset):.4f}', epoch=epoch, total_epoch=total_epoch)

    return  data_all, z_all, recon_batch_all

def loss_function_graph(recon_x, x, regulationMatrix=None, regularizer_type='noregu', regu_strength=0.9, reduction='sum'):
    '''
    Regularized by the graph information
    Reconstruction + KL divergence losses summed over all elements and batch
    '''
    if regularizer_type in ['LTMG', 'Celltype']:
        x.requires_grad = True
    
    BCE = (1-regu_strength) * F.mse_loss(recon_x, x, reduction=reduction) # [cell batch * gene]
    
    if regularizer_type == 'noregu':
        loss = BCE

    elif regularizer_type == 'LTMG':
        loss = BCE + regu_strength * ( F.mse_loss(recon_x, x, reduction='none') * regulationMatrix ).sum()

    elif regularizer_type == 'Celltype':

        graphregu, celltyperegu = regulationMatrix
        loss = 0.3 * ( graphregu @ F.mse_loss(recon_x, x, reduction='none') ).sum() + \
            0.1 * ( celltyperegu @ F.mse_loss(recon_x, x, reduction='none') ).sum()

    return loss