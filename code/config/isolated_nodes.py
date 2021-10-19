#%% Imports
import pandas as pd
import numpy as np


#%% Definition of the simulation name
# Here you should define:
#    simulation_id
simulation_id = 'five_agents-very_unbalanced-disconnected'


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

random_state = 42
verbose = 0
n_new_estimators = 5
max_estimators = 25
n_share = 5
max_depth = 10
n_rep = 10
val_size = 0.3
test_size = 0.1


#%% Agents' names and connections
# Here you should define:
#   n_agents
#   ID
#   network

n_agents = 5
ID = ['Node%d' % x for x in range(n_agents)]

network = pd.DataFrame((np.ones((n_agents, n_agents)) - 
                        np.eye(n_agents)).astype('i'), columns=ID, index=ID)


#%% Definition of the nodes' data files
# Here you should define:
#    data_files

data_path = '../data/very_unbalanced_split_5/'
label_col = 'class'
unused_col = 'time'

data_files = {}
for id in ID:
    data_files[id] = id + '_' + 'creditcard.csv'


#%% Definition of the agents' actions
# Here you should define:
#    actions

acts = ['fit', 'share', 'get', 
        'fit', 'share', 'get', 
        'fit', 'share', 'get',
        'fit', 'share', 'get',
        'fit', 'share', 'get',
        'fit', 'share', 'get',
        ]

# Here every agent perform the same actions
actions = {}
for t in range(len(acts)):
    actions[t] = {}
    for id in ID:
        actions[t][id] = acts[t]







