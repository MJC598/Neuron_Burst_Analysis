import torch
import numpy as np
from sklearn.metrics import r2_score
import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from config import params


def r2_eval(model, testing_dataloader, filt=None, k=None):
    output_list = []
    labels_list = []

    if filt is not None:
        filt = iter(filt)
    for i, (x, y) in enumerate(testing_dataloader):
        if params.RECURRENT_NET:
            x = torch.transpose(x, 2, 1)
        output = model(x)         
        if filt is None:
            output_list.append(output.detach().cpu().numpy())
            labels_list.append(y.detach().cpu().numpy())
        else:
            xf, yf = next(filt)
            yf = yf.detach().cpu().numpy().reshape((-1,1))
            y = y.detach().cpu().numpy().reshape((-1,1))
            output = output.detach().cpu().numpy().reshape((-1,1))
            pred = yf-y+output
            output_list.append(pred)
            labels_list.append(yf)
        if k != None and i == k-1:
            break
    output_list = np.squeeze(np.concatenate(output_list, axis=0))
    labels_list = np.squeeze(np.concatenate(labels_list, axis=0))
    for j in range(params.LOOK_AHEAD):
        print("{} steps ahead: {}".format(j, 
                                        r2_score(labels_list[:,j], 
                                                output_list[:,j])))
    return output_list, labels_list