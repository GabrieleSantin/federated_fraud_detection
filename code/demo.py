#%% Imports
from utils import get_data, merge_test_set, count
import matplotlib.pyplot as plt
import seaborn as sns
from simulator import Simulator
import numpy as np

# How many steps of the algorithm you want to see in the plot
max_steps_plot = 20


#%% Import the experiment's configuration
# from config.isolated_nodes import simulation_id, net_cfg, alg_cfg, data_cfg, actions
# from config.loop import simulation_id, net_cfg, alg_cfg, data_cfg, actions
from config.fully_connected import simulation_id, net_cfg, alg_cfg, data_cfg, actions


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
   
# Merge the test sets if required    
if data_cfg['unique_test']:
    X_test, y_test = merge_test_set(X_test, y_test)

# Print some stats
for id in net_cfg['ID']:
    print('-' * 42)
    print('|       | Samples | Frauds | Fraud ratio |')
    print('-' * 42)
    print('| Train | %6d  | %6d |      %2.4f |' % (y_train[id].shape[0], y_train[id].sum(), 
                                                   y_train[id].sum() / y_train[id].shape[0]))
    print('| Test  | %6d  | %6d |      %2.4f |' % (y_test[id].shape[0], y_test[id].sum(), 
                                                   y_test[id].sum() / y_test[id].shape[0]))
    print('-' * 42)


#%% Simulate
# Initialize the simulator
s = Simulator(alg_cfg, net_cfg)

# Set the nodes' data and actions
s.init_simulation(actions, 
             X_train, y_train, input_scaler, 
             X_test, y_test)

# Run the simulation
estimators_ids, scores = s.simulate()
   

#%% Plot the scoress
n_agents = len(net_cfg['ID'])

# Test scores
metrics = ['test_bacc', 'test_precision', 'test_recall']
fig = plt.figure(1)
fig.clf()
for idx, id in enumerate(net_cfg['ID']):
    ax = fig.add_subplot(1, n_agents, idx+1)
    leg = []
    for metric in metrics:
        values = []
        x_label = []
        for idx_t, t in enumerate(scores[id]):
            if idx_t >= len(scores[id]) - max_steps_plot:
                values.append(t[metric])
                x_label.append(str(idx_t) + ': ' + actions[id][idx_t])
        ax.plot(x_label, values, 'o-')
        leg.append(metric)
        ax.text(max_steps_plot - 0.5, values[-1], '%2.2f' % values[-1])
    ax.legend(leg)
    ax.grid(True)
    ax.set_title(id)
    plt.xticks(rotation=90)
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, np.min([max_steps_plot, len(scores[id])]) + 1)
fig.tight_layout()


# Train scores
metrics = ['train_bacc', 'train_precision', 'train_recall']
fig = plt.figure(2)
fig.clf()
for idx, id in enumerate(net_cfg['ID']):
    ax = fig.add_subplot(1, n_agents, idx+1)
    leg = []
    for metric in metrics:
        values = []
        x_label = []
        for idx_t, t in enumerate(scores[id]):
            if idx_t >= len(scores[id]) - max_steps_plot:
                values.append(t[metric])
                x_label.append(str(idx_t) + ': ' + actions[id][idx_t])
        ax.plot(x_label, values, 'o-')
        leg.append(metric)
        ax.text(max_steps_plot - 0.5, values[-1], '%2.2f' % values[-1])
    ax.legend(leg)
    ax.grid(True)
    ax.set_title(id)
    plt.xticks(rotation=90)
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, np.min([max_steps_plot, len(scores[id])]) + 1)
fig.tight_layout()


#%% Plot the estimators count
counts = count(estimators_ids, actions)

fig = plt.figure(3)
fig.clf()
for idx, id in enumerate(net_cfg['ID']):
    ax = fig.add_subplot(2, np.ceil(n_agents / 2), idx+1)
    sns.barplot(data=counts[id][-max_steps_plot*n_agents:], 
                x='iteration', y='count', hue='node', ax=ax)
    ax.grid(True)
    ax.set_title(id)
    plt.xticks(rotation=90)
fig.tight_layout()

        