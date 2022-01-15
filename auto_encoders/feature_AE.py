"""

"""
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

    X_dataset = util.ExpressionDataset(X)
    X_loader = DataLoader(X_dataset, batch_size=batch_size, **param['dataloader_kwargs'])
    TRS = torch.from_numpy(TRS).type(torch.FloatTensor)

    feature_AE = model.Feature_AE(dim=X.shape[1]).to(param['device'])
    optimizer = optim.Adam(feature_AE.parameters(), lr=learning_rate)

    if model_state is not None:
        feature_AE.load_state_dict(model_state['model'])
        # optimizer.load_state_dict(model_state['optimizer']) # Adam optimizer includes momentum, will turn this off for now

    _, X_embed, X_recon = train.train_handler(
        model = feature_AE, 
        train_loader = X_loader, 
        optimizer = optimizer, 
        TRS = TRS, 
        total_epoch = total_epoch,
        regu_strength = regu_strength,
        masked_prob = masked_prob,
        param = param)

    checkpoint = {
        'model': feature_AE.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    return X_embed.detach().cpu().numpy(), X_recon.detach().cpu().numpy(), checkpoint # cell * {gene, embedding}
