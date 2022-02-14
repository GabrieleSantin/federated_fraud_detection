#%% Imports
from utils import get_data
from simulator import Simulator
import numpy as np


#%% Import the experiment's configuration
from config.example import simulation_id, net_cfg, alg_cfg, data_cfg, actions
print(20 * '-' + 'Running the simulation %s' % simulation_id + 20 * '-')


#%% Load and define the agents' datasets
X_train, X_test, y_test, y_train, input_scaler = {}, {}, {}, {}, {}

for id in net_cfg['ID']:
    #% Load data, preprocess, split 
    X_train[id], X_test[id], y_test[id], y_train[id], input_scaler[id] = get_data(data_cfg['data_path'],
                                                              data_cfg['data_files'][id],
                                                              data_cfg['unused_col'],
                                                              data_cfg['label_col'],
                                                              alg_cfg['random_state'],
                                                              data_cfg['test_size'])

    
#%% Simulate
# Initialize the simulator
s = Simulator(alg_cfg, net_cfg)

# Set the nodes' data and actions
s.init_simulation(actions, 
             X_train, y_train, input_scaler, 
             X_test, y_test)

# Run the simulation
estimators_ids, scores = s.simulate()
   

#%% Fuse the models, save the result

# Create an additional node to store the final result
from simulator import Node
final_id = 'Final'

# Set the capacity of the node so large that it fits all the other nodes' estimators
max_estimators = alg_cfg['max_estimators'] * len(net_cfg['ID'])

# Copy the node's parameter, and modify the capacity
final_alg_cfg = {}
for key in alg_cfg:
    final_alg_cfg[key] = alg_cfg[key]
final_alg_cfg['max_estimators'] = max_estimators

# Actually create the node and extract its model
final_model = Node(id=final_id, alg_cfg=final_alg_cfg).model

# Run over the existing nodes and add their estimators to the final model
for node_id in s.nodes:
    estimators = s.nodes[node_id].model.get_top_estimators([], [], top=max_estimators)
    final_model.add_external_estimators(estimators, [])


# Save the final estimator to a json file
from utils import write_json, read_json
filename = 'final_estimator'
write_json(final_model, final_id, filename)
    
# Try to read it from file
model_reborn = read_json(filename)

# Extract the first 20 samples from one of the datasets
X = X_train[net_cfg['ID'][0]]

# Check if the two estimators give the same prediction on X
final_pred = final_model.predict(X)
reborn_pred = model_reborn.predict(X)
print(np.all)