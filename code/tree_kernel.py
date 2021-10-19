import numpy as np
import networkx as nx
import scipy.sparse
import sys
from abc import ABC, abstractmethod


#%%
class sklearn_tree():
    def __init__(self, t):
        self.n_nodes = t.node_count
        self.children_left = t.children_left
        self.children_right = t.children_right
        self.feature = np.clip(t.feature, 0, None)

        self.node_features = np.clip(t.threshold, 0, None)
        self.children = {}
        for idx in range(self.n_nodes):
            
            child = []
            label = []
            if self.children_right[idx] > 0:
                child.append(self.children_right[idx])
                label.append(self.feature[self.children_right[idx]])

            if self.children_left[idx] > 0:
                child.append(self.children_left[idx])
                label.append(self.feature[self.children_left[idx]])
                
            sort_idx = np.argsort(label)
            self.children[idx] = {'child': np.array(child)[sort_idx].tolist(), 
                                  'label': np.array(label)[sort_idx].tolist()}

    def to_nx(self):
        row_ind = np.tile(np.arange(self.n_nodes), (2, ))
        col_ind = np.r_[self.children_left, self.children_right]
        row_ind = row_ind[col_ind > 0]
        col_ind = col_ind[col_ind > 0]
        data = np.ones(col_ind.shape[0])
        
        A = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), 
                                    shape=(self.n_nodes, self.n_nodes))    
            
        G = nx.from_scipy_sparse_matrix(A)
                
        attributes = {idx: np.clip(x, 0, None) for idx, x in enumerate(self.feature)}
        nx.set_node_attributes(G, attributes, 'feature')
        
        attributes = {idx: np.clip(x, 0, None) for idx, x in enumerate(self.node_features)}
        nx.set_node_attributes(G, attributes, 'threshold')

        return G
    
    
    def plot(self, ax):
        T = self.to_nx()
        pos = hierarchy_pos(T, 1)
        
        nx.draw(T, pos=pos, 
                node_size=20, ax=ax)
        
        nx.draw_networkx_labels(T, pos={key: (pos[key][0]+0.01, pos[key][1]+0.05) for key in pos},
                labels=nx.get_node_attributes(T, 'feature'), 
                node_size=20, ax=ax)
   
    
#%%
# Abstract kernel
class Kernel(ABC):
    @abstractmethod    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def eval(self):
        pass

    def eval_prod(self, x, y, v, batch_size=100):
        N = x.shape[0]
        num_batches = int(np.ceil(N / batch_size))
        mat_vec_prod = np.zeros((N, 1)) 
        for idx in range(num_batches):
            idx_begin = idx * batch_size
            idx_end = (idx + 1) * batch_size
            A = self.eval(x[idx_begin:idx_end, :], y)
            mat_vec_prod[idx_begin:idx_end] = A @ v
        return mat_vec_prod

    @abstractmethod
    def diagonal(self, X):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass


#%%
class LinearKernelNodes(Kernel):
    def __init__(self):
        super(LinearKernelNodes, self).__init__()
        self.name = 'LinearKernel'
        
    def eval(self, x, y):
        return x[:, None] @ y[:, None].T

    def diagonal(self, x):
        return np.linalg.norm(x, axis=1) ** 2
    
    def __str__(self):
     return self.name   

    def set_params(self, par):
        pass
        

#%%
class SubtreeKernelNodes(Kernel):
    def __init__(self, reg):
        super(SubtreeKernelNodes, self).__init__()
        self.name = 'NodeSubtreeKernel'
        self.reg = reg
        
    def eval(self, children_g, children_h):
        c = -1 * np.ones((len(children_g), len(children_h)))
        for n in children_g:
            for m in children_h:
                c[n, m] = self.C(children_g, children_h, n, m, c, self.reg)
        return c

    def diagonal(self, children_g):
        c = -1 * np.ones(len(children_g))
        for n in children_g:
            c[n] = self.C(children_g, children_g, n, n, c, self.reg)
        return c
    
    def __str__(self):
     return self.name   

    def set_params(self, par):
        self.reg = par

    def C(self, children_g, children_h, n, m, c, reg):
        if c[n, m] > 0:
            return c[n, m]
        
        if children_g[n]['label'] != children_h[m]['label']: # The labels are different
            return 0
        
        if children_g[n]['label'] == [0, 0]: # The labels are equal since the nodes are terminal
            return reg
        else:
            ni = len(children_g[n]['child'])
            p = 1
            for idx in range(ni):
                n_tmp = children_g[n]['child'][idx] 
                m_tmp = children_h[m]['child'][idx]
                c[n_tmp, m_tmp] = self.C(children_g, children_h, n_tmp, m_tmp, c, self.reg)
                p *= 1 + c[n_tmp, m_tmp]
            return reg * p
   

#%%    
class TreeKernel(Kernel):
    def __init__(self, topology_kernel, feature_kernel, attribute):
        super(TreeKernel, self).__init__()
        self.topology_kernel = topology_kernel
        self.feature_kernel = feature_kernel
        self.attribute = attribute
        self.name = 'Tree kernel'
        
        
    def eval_0(self, children_g, children_h, node_features_g, node_features_h):
        A = self.topology_kernel.eval(children_g, children_h) 
        B = self.feature_kernel.eval(node_features_g, node_features_h)
        
        return np.sum(A * B)
   
    def eval(self, G, H):
        if not isinstance(G, list):
            G = [G]
        if not isinstance(H, list):
            H = [H]
            
        children_g = [g.children for g in G]
        children_h = [h.children for h in H]
        
        node_features_g = [g.node_features for g in G]
        node_features_h = [h.node_features for h in H]
                
        A = np.zeros((len(G), len(H)))
        for i in range(len(G)):
            for j in range(len(H)):
                A[i, j] = self.eval_0(children_g[i], children_h[j], 
                                      node_features_g[i], node_features_h[j])
        
        return A
            
    def diagonal(self, G):
        if not isinstance(G, list):
            G = [G]
            
        children_g = [g.children for g in G]
        node_features_g = [g.node_features for g in G]

        d = np.zeros(len(G))
        for i in range(len(G)):
            d[i] = self.eval_0(children_g[i], children_g[i], 
                                  node_features_g[i], node_features_g[i])
        
        return d
    
    def __str__(self):
        return self.name   

    def set_params(self, params):
        pass


#%%
import random
   
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)