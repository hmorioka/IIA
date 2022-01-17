""" Evaluation
    Main script for evaluating the model trained by iia_training.py
"""


import os
import numpy as np
import pickle
import torch

from subfunc.generate_artificial_data import generate_artificial_data
from subfunc.preprocessing import pca
from subfunc.showdata import *
from igcl import igcl, utils
from itcl import itcl
from sklearn.metrics import accuracy_score


# parameters ==================================================
# =============================================================

eval_dir_base = './storage'

eval_dir = os.path.join(eval_dir_base, 'model')

parmpath = os.path.join(eval_dir, 'parm.pkl')
savefile = eval_dir.replace('.tar.gz', '') + '.pkl'

load_ema = True  # recommended unless the number of iterations was not enough

num_data_test = -1  # number of data points for testing (-1: same with training)
# num_data_test = None  # do not generate test data


# =============================================================
# =============================================================

if eval_dir.find('.tar.gz') >= 0:
    unzipfolder = './storage/temp_unzip'
    utils.unzip(eval_dir, unzipfolder)
    eval_dir = unzipfolder
    parmpath = os.path.join(unzipfolder, 'parm.pkl')

modelpath = os.path.join(eval_dir, 'model.pt')

# Load parameter file
with open(parmpath, 'rb') as f:
    model_parm = pickle.load(f)

num_comp = model_parm['num_comp']
num_data = model_parm['num_data']
ar_order = model_parm['ar_order']
modulate_range = model_parm['modulate_range']
modulate_range2 = model_parm['modulate_range2'] if 'modulate_range2' in model_parm else None
num_basis = model_parm['num_basis']
num_layer = model_parm['num_layer']
list_hidden_nodes = model_parm['list_hidden_nodes']
list_hidden_nodes_z = model_parm['list_hidden_nodes_z'] if 'list_hidden_nodes_z' in model_parm else None
moving_average_decay = model_parm['moving_average_decay']
random_seed = model_parm['random_seed']
pca_parm = model_parm['pca_parm']
num_segment = model_parm['num_segment'] if 'num_segment' in model_parm else None
num_segmentdata = model_parm['num_segmentdata'] if 'num_segmentdata' in model_parm else None
net_model = model_parm['net_model'] if 'net_model' in model_parm else 'igcl'
if num_data_test == -1:
    num_data_test = num_data


# Generate sensor signal --------------------------------------
x, s, y, x_te, s_te, y_te, _,_,_ = generate_artificial_data(num_comp=num_comp,
                                                            num_data=num_data,
                                                            num_layer=num_layer,
                                                            num_basis=num_basis,
                                                            modulate_range1=modulate_range,
                                                            modulate_range2=modulate_range2,
                                                            num_data_test=num_data_test,
                                                            random_seed=random_seed)

if net_model == 'itcl':  # remake label for TCL learning
    num_segmentdata = int(np.ceil(num_data / num_segment))
    y = np.tile(np.arange(num_segment), [num_segmentdata, 1]).T.reshape(-1)[:num_data]
    if x_te is not None:
        num_segmentdata_te = int(np.ceil(num_data_test / num_segment))
        y_te = np.tile(np.arange(num_segment), [num_segmentdata_te, 1]).T.reshape(-1)[:num_data_test]

# Preprocessing -----------------------------------------------
x, _ = pca(x, num_comp, params=pca_parm)
if x_te is not None:
    x_te, _ = pca(x_te, num_comp, params=pca_parm)

# Evaluate model ----------------------------------------------
# -------------------------------------------------------------

# transpose for pytorch
x = x.T
if x_te is not None:
    x_te = x_te.T

# define network
if net_model == 'igcl':
    model = igcl.NetGaussScaleMean(h_sizes=list_hidden_nodes,
                                   h_sizes_z=list_hidden_nodes_z,
                                   ar_order=ar_order,
                                   num_dim=x.shape[1],
                                   num_data=num_data,
                                   num_basis=num_basis)
elif net_model == 'itcl':
    model = itcl.Net(h_sizes=list_hidden_nodes,
                     h_sizes_z=list_hidden_nodes_z,
                     ar_order=ar_order,
                     num_dim=x.shape[1],
                     num_class=num_segment)
device = 'cpu'
model = model.to(device)
model.eval()

# load parameters
print('Load trainable parameters from %s...' % modelpath)
checkpoint = torch.load(modelpath, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
if load_ema:
    model.load_state_dict(checkpoint['ema_state_dict'])

# augment data for AR model
t_idx = np.arange(x.shape[0] - ar_order) + ar_order
t_idx = t_idx.reshape([-1, 1]) + np.arange(0, -ar_order - 1, -1).reshape([1, -1])
x = x[t_idx.reshape(-1), :].reshape([-1, ar_order + 1, x.shape[-1]])
y = y[t_idx[:, 0]]

# forward
x_torch = torch.from_numpy(x.astype(np.float32)).to(device)
y_torch = torch.from_numpy(y).type(torch.LongTensor).to(device)
if net_model == 'igcl':
    logits, h, hz, _, _ = model(x_torch, y_torch)
    predicted = (logits > 0.5).float()
    h, hstar = torch.split(h, split_size_or_sections=int(h.size()[0]/2), dim=0)
    hz, hzstar = torch.split(hz, split_size_or_sections=int(hz.size()[0]/2), dim=0)
elif net_model == 'itcl':
    logits, h, hz = model(x_torch)
    _, predicted = torch.max(logits.data, 1)

# convert to numpy
pred_val = predicted.cpu().numpy()
h_val = np.squeeze(h.detach().cpu().numpy())
hz_val = np.squeeze(hz.detach().cpu().numpy()) if 'hz' in locals() else None

# for test data
if x_te is not None:
    t_idx = np.arange(x_te.shape[0] - ar_order) + ar_order
    t_idx = t_idx.reshape([-1, 1]) + np.arange(0, -ar_order - 1, -1).reshape([1, -1])
    x_te = x_te[t_idx.reshape(-1), :].reshape([-1, ar_order + 1, x_te.shape[-1]])
    y_te = y_te[t_idx[:, 0]]

    x_torch = torch.from_numpy(x_te.astype(np.float32)).to(device)
    y_torch = torch.from_numpy(y_te).type(torch.LongTensor).to(device)
    if net_model == 'igcl':
        logits_te,_,_,_,_ = model(x_torch, y_torch)
        predicted_te = (logits_te > 0.5).float()
    elif net_model == 'itcl':
        logits_te, _, _ = model(x_torch)
        _, predicted_te = torch.max(logits_te.data, 1)
    pred_val_te = predicted_te.cpu().numpy()


# Evaluate outputs --------------------------------------------
# -------------------------------------------------------------

# Calculate accuracy
if net_model == 'igcl':
    label_val = np.concatenate([np.ones(y.shape[0]), np.zeros(y.shape[0])])
    accu_tr = accuracy_score(pred_val, label_val.T)
    if x_te is not None:
        label_val_te = np.concatenate([np.ones(y_te.shape[0]), np.zeros(y_te.shape[0])])
        accu_te = accuracy_score(pred_val_te, label_val_te.T)
elif net_model == 'itcl':
    accu_tr = accuracy_score(pred_val, y)
    if x_te is not None:
        accu_te = accuracy_score(pred_val_te, y_te)

# correlation
corrmat_tr, sort_idx, _ = utils.correlation(h_val, s[:, :-1].T, 'Pearson')

meanabscorr_tr = np.mean(np.abs(np.diag(corrmat_tr)))

showmat(corrmat_tr,
        yticklabel=np.arange(num_comp),
        xticklabel=sort_idx,
        ylabel='source',
        xlabel='feature')

# Display results
print('Result...')
print('    accuracy (train) : %7.4f [percent]' % (accu_tr * 100))
if 'accu_te' in locals():
    print('    accuracy (test)  : %7.4f [percent]' % (accu_te * 100))
print('    correlation      : %7.4f' % meanabscorr_tr)


# Save results
result = {'accu_tr': accu_tr,
          'accu_te': accu_te if 'accu_te' in locals() else None,
          'corrmat_tr': corrmat_tr,
          'meanabscorr_tr': meanabscorr_tr,
          'sort_idx': sort_idx,
          'num_comp': num_comp,
          'modelpath': modelpath}

print('Save results...')
with open(savefile, 'wb') as f:
    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


print('done.')
