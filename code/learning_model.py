from sklearn.ensemble import RandomForestRegressor
import numpy as np
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


#%%
class Classifier(ABC):
    def __init__(self, ID, n_new_estimators=50, 
                 max_estimators=50,
                 max_depth=20, n_rep=100, 
                 random_state=42, val_size=0.3,
                 verbose=1):
        '''
        Constructor.

        Parameters
        ----------
        ID : TYPE
            DESCRIPTION.
        n_new_estimators : TYPE, optional
            DESCRIPTION. The default is 50.
        max_estimators : TYPE, optional
            DESCRIPTION. The default is 50.
        max_depth : TYPE, optional
            DESCRIPTION. The default is 20.
        n_rep : TYPE, optional
            DESCRIPTION. The default is 100.
        random_state : TYPE, optional
            DESCRIPTION. The default is 42.
        val_size : TYPE, optional
            DESCRIPTION. The default is 0.3.
        verbose : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        '''
        super(Classifier, self).__init__()

        # initialize model
        self.estimators_ = {}
        
        self.fit_counter = 0
        
        self.verbose = verbose
        self.random_state = random_state
        self.n_new_estimators = n_new_estimators
        self.max_estimators = max_estimators
        self.max_depth = max_depth
        
        self.n_rep = n_rep
        self.val_size = val_size
        
        self.ID = ID
        self.estimator_ID_counter = 0
        
        self.ensemble_estimator = RandomForestRegressor(n_estimators=self.n_new_estimators, 
                                       verbose=self.verbose, 
                                       random_state=self.random_state,
                                       max_depth=self.max_depth)
        
        
    # Utility methods
    def next_id(self, num_ids=1):
        '''
        Generate num_ids new ids in an incremental way
  
        Parameters
        ----------
        num_ids : TYPE
            DESCRIPTION.
        
        Returns
        -------
        ids : TYPE
            DESCRIPTION.
        '''
        ids = [(self.ID, id) for id in range(self.estimator_ID_counter, self.estimator_ID_counter + num_ids)]
        self.estimator_ID_counter += num_ids
        return ids
     
        
    # Prediction methods
    def eval_estimators(self, X):
        '''
        Evaluate all the estimators in self.estimators_ on the points X. 
        If a list of ids is passed, evaluate only those estimators.
        
        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        P : TYPE
            DESCRIPTION.

        '''
        P = np.zeros((len(self.estimators_), X.shape[0]))
        for idx, id in enumerate(self.estimators_):
            P[idx] = self.estimators_[id].predict(X)
        return P    
      
    
    def predict_proba(self, X):
        '''
        Compute the predicted class of the data X, as a continuos probability 
        value in [0, 1].
        
        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        y_pred : TYPE
            DESCRIPTION.

        '''
        if self.estimators_:
            P = self.eval_estimators(X)
            y_pred = np.mean(P, axis=0)
        else:
            y_pred = np.zeros(X.shape[0])
        return y_pred
    
    
    def predict(self, X):
        '''
        Compute the predicted class of the data X, as a discrete 0/1 value.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        y_pred : TYPE
            DESCRIPTION.

        '''
        y_pred = self.predict_proba(X)
        y_pred = np.round(y_pred)
        
        return y_pred   


    # Fitting methods    
    def fit(self, X, y, input_scaler):
        '''
        Fit the estimator to the data.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        input_scaler : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.random_state += 1
        self.ensemble_estimator.set_params(random_state=self.random_state)

        self.ensemble_estimator.fit(input_scaler.transform(X), y)
        self.fit_counter += 1
        
        new_estimators = self.ensemble_estimator.estimators_
        new_ids = self.next_id(len(new_estimators))
        for idx, new_id in enumerate(new_ids):
            self.estimators_[new_id] = Pipeline([('scaler', input_scaler), 
                                                 ('tree', new_estimators[idx])]) 
        
        self.crop_estimators(X, y)        
        

    # Model update methods
    def crop_estimators(self, X, y=None):
        '''
        Remove the least important estimators, according to the sorting
        provided by selg.get_top_estimators.
        
        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        if len(self.estimators_) > self.max_estimators:
            self.estimators_ = self.get_top_estimators(X, y=y, 
                                                       top=self.max_estimators)
            

    def add_external_estimators(self, estimators, X, y=None):
        '''
        Add estimators received from other agents.

        Parameters
        ----------
        estimators : TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.
        y : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        for id in set(estimators) - set(self.estimators_):            
            self.estimators_[id] = estimators[id]    
           
        # Crop if needed
        self.crop_estimators(X, y)    

            
    @abstractmethod    
    def get_top_estimators(self, X, y=None, top=None):
        '''
        Sorts the estimators according to some rule and returns the top most
        important ones.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE, optional
            DESCRIPTION. The default is None.
        top : TYPE, optional
            DESCRIPTION. The default is None.
        Returns
        -------
        None.

        '''
        pass
    
    
#%%
class ClassifierLinReg(Classifier):
    def __init__(self, ID, n_new_estimators=50, 
                  max_estimators=50,
                  max_depth=20, n_rep=100, 
                  random_state=42, val_size=0.3,
                  verbose=1):
        '''
        Constructor.

        Parameters
        ----------
        ID : TYPE
            DESCRIPTION.
        n_new_estimators : TYPE, optional
            DESCRIPTION. The default is 50.
        max_estimators : TYPE, optional
            DESCRIPTION. The default is 50.
        max_depth : TYPE, optional
            DESCRIPTION. The default is 20.
        n_rep : TYPE, optional
            DESCRIPTION. The default is 100.
        random_state : TYPE, optional
            DESCRIPTION. The default is 42.
        val_size : TYPE, optional
            DESCRIPTION. The default is 0.3.
        verbose : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        '''
        super(ClassifierLinReg, self).__init__(ID, n_new_estimators, 
                                                        max_estimators,
                                                        max_depth, n_rep, 
                                                        random_state, val_size,
                                                        verbose)

    def get_top_estimators(self, X, y, top):
        '''
        Rank the estimator using a bootstrapped linear estimator on the training
        set.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        top : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if top > len(self.estimators_):
            return self.estimators_.copy()
        else:        
            estimators_ = {}
           
            P = self.eval_estimators(X)
            
            w_ = np.zeros(len(self.estimators_))
        
            n = X.shape[0]
            n_val = int(np.floor(n * self.val_size))
            n_train = n - n_val
            
            for _ in range(self.n_rep):
                sample_idx = np.random.permutation(n)[:n_train]
                lin_fit = LinearRegression()
                lin_fit.fit(P[:, sample_idx].transpose(), y[sample_idx])

                w_ += lin_fit.coef_
                
            w_ /= self.n_rep

            ww_ = {id: w_[idx] for idx, id in enumerate(self.estimators_)}

            weight_norm = {id: ww_[id] ** 2 for id in self.estimators_}
            sorted_ids = sorted(weight_norm, key=weight_norm.get, reverse=True)    
                
            w_sum = np.sum([ww_[id] for id in sorted_ids[:top]])
            
            if w_sum == 0:
                w_sum = 1
                
            for id in sorted_ids[:top]:
                estimators_[id] = self.estimators_[id]
                
            return estimators_.copy()
        
  