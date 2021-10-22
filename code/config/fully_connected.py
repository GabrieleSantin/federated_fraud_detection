#%% Imports
import pandas as pd
import numpy as np


#%% Definition of the simulation name
# Here you should define:
#    simulation_id
simulation_id = 'five_agents-very_unbalanced-fully_connected'


#%% Algorithm parameters
# Here you should define:
#    random_state
#    verbose
#    n_new_estimators
#    max_estimators
#    n_share
#    max_depth
#    n_rep
#    val_size
#    test_size

alg_cfg = {}
alg_cfg['random_state'] = 42
alg_cfg['verbose'] = 0
alg_cfg['n_new_estimators'] = 5
alg_cfg['max_estimators'] = 10
alg_cfg['n_share'] = 5
alg_cfg['max_depth'] = 4
alg_cfg['n_rep'] = 10
alg_cfg['val_size'] = 0.3


#%% Agents' names and connections
# Here you should define:
#   n_agents
#   ID
#   network
#   unique_test

net_cfg = {}
n_agents = 5
net_cfg['ID'] = ['Node%d' % x for x in range(n_agents)]

net_cfg['network'] = pd.DataFrame((np.ones((n_agents, n_agents)) - 
                        np.eye(n_agents)).astype('i'), columns=net_cfg['ID'], index=net_cfg['ID'])


#%% Definition of the nodes' data files
# Here you should define:

data_cfg = {}
data_cfg['data_path'] = '../data/very_unbalanced_split_5/'
data_cfg['label_col'] = 'class'
data_cfg['unused_col'] = 'time'

data_cfg['data_files'] = {}
for id in net_cfg['ID']:
    data_cfg['data_files'][id] = id + '_' + 'creditcard.csv'

data_cfg['test_size'] = 0.1
data_cfg['unique_test'] = True


#%% Definition of the agents' actions
# Here you should define:
#    actions

acts = ['fit', 'share', 'get', 
        'fit', 'share', 'get', 
        'fit', 'share', 'get',
        'fit', 'share', 'get',
        ]

# Here every agent perform the same actions
actions = {}
for id in net_cfg['ID']:
    actions[id] = acts.copy()


