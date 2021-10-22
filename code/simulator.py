class Simulator():
    # Set up methods
    def __init__(self, alg_cfg, net_cfg):
        '''
        Construct the simulator by saving some config and initializing the 
        nodes.

        Parameters
        ----------
        alg_cfg : dict
            Configuration parameters for the single node learning model.
        net_cfg : dict
            Configuration parameters for the network of nodes.

        Returns
        -------
        None.

        '''        
        # Save the network configuration
        self.net_cfg = net_cfg
        
        # Initialize the nodes
        self.nodes = {}
        for id in self.net_cfg['ID']:
            self.nodes[id] = Node(id, alg_cfg)

    def init_simulation(self, actions, 
                        X_train, y_train, input_scaler, 
                        X_test, y_test):
        '''
        Initialize the simulation: construct the registry used for communcation
        between the nodes, and assign to each node its dataset and list of actions.

        Parameters
        ----------
        actions : dict
            For each node, it gives a list of actions to be performed.
        X_train : dict
            Input train set for each node.
        y_train : dict
            Output train set set for each node.
        input_scaler : dict
            Scaler of the input set for each node.
        X_test : dict
            Input test set set for each node.
        y_test : dict
            Output test set set for each node.

        Returns
        -------
        None.

        '''
        # Initialize the communication registry 
        registry = Registry(self.net_cfg['ID'], self.net_cfg['network'])

        # Initialize the nodes' simulation 
        for id in self.net_cfg['ID']:
            self.nodes[id].init_simulation(registry, actions[id], 
                                X_train[id], y_train[id], input_scaler[id], 
                                X_test[id], y_test[id]) 
 
    
    # Simulation methods
    def get_active_nodes(self):
        '''
        Find the list of nodes that are still active.

        Returns
        -------
        active : list
            The ids of the active nodes.

        '''
        active = [id for id in self.nodes if self.nodes[id].is_active()]
        return active
    
    def simulate(self):
        '''
        Run the simulation: iterate over each node in the order given by the 
        list of ids self.net_cfg['ID'], and for each node execute its currently
        planned action, or ignore it if no action is left.
        For each iteration and each node, the method saves the ids of the 
        estimators of each node and the test scores.

        Returns
        -------
        estimators_ids : dict
            For each node, a list of the estimators used by the node at each 
            iteration (except for the 'share' iteration, which does not change 
            the state of the node)
        scores : dict
            For each node, a list of the scores obtained on the test set by the 
            node at each iteration  (except for the 'share' iteration, which 
            does not change the state of the node)
        '''
        # Initialize some dicts for logging
        estimators_ids = {id: [] for id in self.net_cfg['ID']}
        scores = {id: [] for id in self.net_cfg['ID']}
            
        # Run over the nodes and execute their actions
        iter_count = 0
        while True:
            # Print the current iteration
            iter_count += 1
            print('[Iteration: %4d]' % iter_count)
            
            # Get the list of active nodes
            active_nodes = self.get_active_nodes()
            
            # Stop is there are no active node
            if not active_nodes:
                print('Simulation terminated: no active nodes left')
                break
            
            # Run the actions of the active nodes
            for id in active_nodes:
                # Run the action of the current node
                estimators_ids_node, scores_node = self.nodes[id].perform_action()
                # Log estimator ids
                if estimators_ids_node:
                    estimators_ids[id].append(estimators_ids_node)
                # Log scores
                if scores_node:
                    scores[id].append(scores_node)
                     
        return estimators_ids, scores
    

#%%
class Registry():
    def __init__(self, IDs, network):
        '''
        Construct the registry, that is represented by a dictionary indexed over
        the nodes, and which stores a dictionary as registry per each node.
        The network structure is assumed to be fixed during the simulation, and 
        it is represented as an adjacency matrix of the connection graph.
        
        Parameters
        ----------
        IDs : list
            The ids of the nodes.
        network : pd.DataFrame
            The adjacency matrix of the connection graph.

        Returns
        -------
        None.

        '''
        self.registry = {id: {} for id in IDs}
        self.network = network
    
        
    def get_connections(self, id):
        '''
        Finds the nodes connected to the input node.

        Parameters
        ----------
        id : str
            ID of the input node.

        Returns
        -------
        connections : list
            IDs of the nodes connected to the input node.
        '''
        connections = list(self.network[id][self.network[id] == 1].index)
        return connections


    def write(self, id, message):
        '''
        Write the message transmitted by the input node to the registry of its
        neighbors.

        Parameters
        ----------
        id : str
            ID of the input node.
        message : dict
            Message to be appended to the registry of the neighbors.

        Returns
        -------
        None.

        '''
        # Get the list of nodes connected to the node id
        connections = self.get_connections(id)
            
        # Write the message on each of the connections' entry
        for c_id in connections:
            # Append if non empty, initialize if empty
            if self.registry[c_id]:
                self.registry[c_id] = {**self.registry[c_id], **message}
            else:
                self.registry[c_id] = message.copy()
            
    def read(self, id):
        '''
        Read the registry of the input node, return its content, and clear the 
        registry.

        Parameters
        ----------
        id : str
            ID of the input node.

        Returns
        -------
        message : dict
            Message contained in the registry of the node.

        '''
        message = self.registry[id]
        self.registry[id] = {}
        return message


#%%
from learning_model import ClassifierLinReg 
from utils import get_scores


class Node():
    # Set up methods
    def __init__(self, id, alg_cfg):
        '''
        Construct the node by saving its id, some parameters, and constructing
        the instance of the nodes' learning model.

        Parameters
        ----------
        id : str
            ID of the node.
        alg_cfg : dict
            Configuration parameters for the single node learning model.

        Returns
        -------
        None.

        '''
        # Save the node's name
        self.id = id
        self.n_share = alg_cfg['n_share']

        # Initialize the classifier            
        self.model = ClassifierLinReg(ID=id, 
                        n_new_estimators=alg_cfg['n_new_estimators'], 
                        verbose=alg_cfg['verbose'], 
                        random_state=alg_cfg['random_state'],
                        max_depth=alg_cfg['max_depth'],
                        max_estimators=alg_cfg['max_estimators'],
                        n_rep=alg_cfg['n_rep'], 
                        val_size=alg_cfg['val_size'])
            
    def init_simulation(self, registry, actions, 
                        X_train, y_train, input_scaler, 
                        X_test, y_test): 
        '''
        Initialize the simulation: save a pointer to the global registry, a copy
        of the list of actions to be performed during the simulation, and the 
        nodes' dataset.

        Parameters
        ----------
        registry : Registry
            The global registry used for communication.
        actions : list
           List of actions to be performed.
        X_train : pd.DataFrame
            Input train set.
        y_train : pd.DataFrame
            Output train set.
        input_scaler : sklearn.preprocessing._data.StandardScaler
            Scaler of the input.
        X_test : pd.DataFrame
            Input test set.
        y_test : pd.DataFrame
            Output test set.

        Returns
        -------
        None.

        '''
        self.registry = registry
        self.actions = actions.copy()
        self.X_train = X_train
        self.y_train = y_train
        self.input_scaler = input_scaler
        self.X_test = X_test
        self.y_test = y_test
        
  
    # Simulation methods
    def is_active(self):
        '''
        Check if the node has some actions left.

        Returns
        -------
        status: bool
            Active/non-active status.

        '''
        status = len(self.actions) > 0
        return status
    
    def perform_action(self):
        '''
        Run the first action of the list of actions, and remove it from the list.
        The method can perform one of three possible actions:
            1) fit: train the learning model locally on the train set.
            2) share: get the best estimators from the learning model and 
                share them by writing them on the registry of the neighbors.
            3) get: read the own registry entry to get the estimators shared by
                the neighbors at previous iterations, and if any add them to 
                the learning model.
        After a 'fit' or 'get' operation is performed, the method also computes 
        some accuracy scores on the test sets, and saves the ids of the model's
        estimators. The 'share' action does not change the node status, so these
        two metrics are not computed and an empy list is returned.

        Returns
        -------
        estimators_ids : list
            The ids of the estimators currently contained in the model.
        scores : dict
            The value of several metrics, each stored in a dict entry.

        '''
        # Get the first action from the list
        action = self.actions.pop(0)
        print('Agent: %5s, Action: %5s' %(self.id, action))

        if action == 'fit':
            self.model.fit(self.X_train, self.y_train, self.input_scaler)
            
        if action == 'share':
            # Get the top estimators
            estimators = self.model.get_top_estimators(self.X_train, 
                                                       self.y_train,
                                                       self.n_share)
            # Write them to the registry
            self.registry.write(self.id, estimators)
        
        if action == 'get':
            # Read the registry
            estimators = self.registry.read(self.id)
            # Add the estimators to the model            
            self.model.add_external_estimators(estimators, 
                                               self.X_train, self.y_train)
        
        # Compute counts and scores
        estimators_ids, scores = [], []
        if action in ['fit', 'get']:
            estimators_ids = list(self.model.estimators_.keys())
            scores = get_scores(self.model, self.X_train, self.y_train, 
                                self.X_test, self.y_test)

        return estimators_ids, scores