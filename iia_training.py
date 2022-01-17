""" Training
    Main script for training the model
"""


import os
import pickle
import shutil
import tarfile

from subfunc.generate_artificial_data import generate_artificial_data
from subfunc.preprocessing import pca
from igcl.igcl_train import train as igcl_train
from itcl.itcl_train import train as itcl_train
from subfunc.showdata import *


# Parameters ==================================================
# =============================================================

# Data generation ---------------------------------------------
num_layer = 3  # number of layers of mixing-MLP
num_comp = 20  # number of components (dimension)
num_data = 2**18  # number of data points
num_basis = 64  # number of frequencies of fourier bases
modulate_range = [-2, 2]
modulate_range2 = [-2, 2]
ar_order = 1
random_seed = 0  # random seed

# select learning framework (igcl or itcl)
# net_model = 'igcl'  # learn by IIA-GCL
net_model, num_segment = 'itcl', 256  # learn by IIA-TCL


# MLP ---------------------------------------------------------
list_hidden_nodes = [4 * num_comp] * (num_layer - 1) + [num_comp]
list_hidden_nodes_z = None
# list of the number of nodes of each hidden layer of feature-MLP
# [layer1, layer2, ..., layer(num_layer)]


# Training ----------------------------------------------------
initial_learning_rate = 0.1  # initial learning rate (default:0.1)
momentum = 0.9  # momentum parameter of SGD
max_steps = int(3e6)  # number of iterations (mini-batches)
decay_steps = int(1e6)  # decay steps (tf.train.exponential_decay)
decay_factor = 0.1  # decay factor (tf.train.exponential_decay)
batch_size = 512  # mini-batch size
moving_average_decay = 0.999  # moving average decay of variables to be saved
checkpoint_steps = int(1e7)  # interval to save checkpoint
summary_steps = int(1e4)  # interval to save summary
apply_pca = True  # apply PCA for preprocessing or not
weight_decay = 1e-5  # weight decay


# Other -------------------------------------------------------
# # Note: save folder must be under ./storage
train_dir_base = './storage'

train_dir = os.path.join(train_dir_base, 'model')  # save directory (Caution!! this folder will be removed at first)

saveparmpath = os.path.join(train_dir, 'parm.pkl')  # file name to save parameters


# =============================================================
# =============================================================

# Prepare save folder -----------------------------------------
if train_dir.find('/storage/') > -1:
    if os.path.exists(train_dir):
        print('delete savefolder: %s...' % train_dir)
        shutil.rmtree(train_dir)  # remove folder
    print('make savefolder: %s...' % train_dir)
    os.makedirs(train_dir)  # make folder
else:
    assert False, 'savefolder looks wrong'

# Generate sensor signal --------------------------------------
x, s, y,_,_,_,_,_,_ = generate_artificial_data(num_comp=num_comp,
                                               num_data=num_data,
                                               num_layer=num_layer,
                                               num_basis=num_basis,
                                               modulate_range1=modulate_range,
                                               modulate_range2=modulate_range2,
                                               random_seed=random_seed)

if net_model == 'itcl':  # Remake label for TCL learning
    num_segmentdata = int(np.ceil(num_data / num_segment))
    y = np.tile(np.arange(num_segment), [num_segmentdata, 1]).T.reshape(-1)[:num_data]

# Preprocessing -----------------------------------------------
x, pca_parm = pca(x, num_comp=num_comp)  # PCA

# Train model  ------------------------------------------------
if net_model == 'igcl':
    igcl_train(x.T,
               y,
               list_hidden_nodes=list_hidden_nodes,
               list_hidden_nodes_z=list_hidden_nodes_z,
               num_data=num_data,
               num_basis=num_basis,
               initial_learning_rate=initial_learning_rate,
               momentum=momentum,
               max_steps=max_steps,
               decay_steps=decay_steps,
               decay_factor=decay_factor,
               batch_size=batch_size,
               train_dir=train_dir,
               ar_order=ar_order,
               weight_decay=weight_decay,
               checkpoint_steps=checkpoint_steps,
               moving_average_decay=moving_average_decay,
               summary_steps=summary_steps,
               random_seed=random_seed)
elif net_model == 'itcl':
    itcl_train(x.T,
               y,
               list_hidden_nodes=list_hidden_nodes,
               list_hidden_nodes_z=list_hidden_nodes_z,
               num_segment=num_segment,
               initial_learning_rate=initial_learning_rate,
               momentum=momentum,
               max_steps=max_steps,
               decay_steps=decay_steps,
               decay_factor=decay_factor,
               batch_size=batch_size,
               train_dir=train_dir,
               ar_order=ar_order,
               weight_decay=weight_decay,
               checkpoint_steps=checkpoint_steps,
               moving_average_decay=moving_average_decay,
               summary_steps=summary_steps,
               random_seed=random_seed)

# Save parameters necessary for evaluation --------------------
model_parm = {'random_seed': random_seed,
              'num_comp': num_comp,
              'num_data': num_data,
              'ar_order': ar_order,
              'num_basis': num_basis,
              'modulate_range': modulate_range,
              'modulate_range2': modulate_range2,
              'num_layer': num_layer,
              'list_hidden_nodes': list_hidden_nodes,
              'list_hidden_nodes_z': list_hidden_nodes_z,
              'moving_average_decay': moving_average_decay,
              'pca_parm': pca_parm,
              'num_segment': num_segment if 'num_segment' in locals() else None,
              'num_segmentdata': num_segmentdata if 'num_segmentdata' in locals() else None,
              'net_model': net_model}

print('Save parameters...')
with open(saveparmpath, 'wb') as f:
    pickle.dump(model_parm, f, pickle.HIGHEST_PROTOCOL)

# Save as tarfile
tarname = train_dir + ".tar.gz"
archive = tarfile.open(tarname, mode="w:gz")
archive.add(train_dir, arcname="./")
archive.close()

print('done.')
