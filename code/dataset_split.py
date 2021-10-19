#%% Imports
import pandas as pd
import os.path as osp
import numpy as np
from utils import almost_equal_split
import os


#%% Set up 
random_state = 42
source_data_path = '../data/'
data_file = 'creditcard.csv'
# Column to be used as target value
label_col = 'Class'


#%% Data split definition
# n_agents = 4
# offset = 0.01
# target_data_path = '../data/balanced_split/'

# n_agents = 4
# offset = 0.7
# target_data_path = '../data/unbalanced_split/'

# n_agents = 4
# offset = 0.9
# target_data_path = '../data/very_unbalanced_split/'

# n_agents = 10
# offset = 0.9
# target_data_path = '../data/very_unbalanced_split_10/'

n_agents = 5
offset = 0.99 # degree of unbalanceness
target_data_path = '../data/very_unbalanced_split_5/'


#%% Define the names of the nodes
ID = ['Node%d' % x for x in range(n_agents)]


#%% Load data
data = pd.read_csv(osp.join(source_data_path, data_file))


#%% Shuffle
data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    
#%% Create the dataset splitting
# Set the seed
np.random.seed(random_state)

# Extract the target values
y = data[[label_col]].values.ravel()

# Identify the negative and positive class
neg_idx = np.argwhere(y == 0)
pos_idx = np.argwhere(y == 1)

# Compute the splitting indices
split_idx_pos = almost_equal_split(pos_idx.shape[0], n_agents, offset)
split_idx_neg = almost_equal_split(neg_idx.shape[0], n_agents, offset)

# Store the actual splitting by running the splitting indices
data_split = {}
for idx in range(n_agents):
    index = list(pos_idx[split_idx_pos[idx] : split_idx_pos[idx+1]].ravel())
    index += list(neg_idx[split_idx_neg[idx] : split_idx_neg[idx+1]].ravel())
     
    data_split[ID[idx]] = data.loc[index]
    

#%% Print some stats
print('-' * 42)
print('|    ID | Samples | Frauds | Fraud ratio |')
print('-' * 42)
for id in ID:
    print('| %4s |  %6d |  %5d |      %2.4f |' % (id, 
                                                   data_split[id].shape[0], 
                                                   data_split[id][label_col].sum(), 
                                                   data_split[id][label_col].sum() / data_split[id].shape[0]))
print('-' * 42)


#%% Save the newly created dataset
for id in ID:
    if not os.path.exists(target_data_path):
        os.makedirs(target_data_path)
    data_split[id].to_csv(osp.join(target_data_path, id + '_' + data_file), index=None)



