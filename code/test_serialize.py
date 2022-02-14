#%% Imports
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import write_json, read_json


#%% Generate random training data
d = 10
n = 20
X = np.random.rand(n, d)
y = np.random.randint(2, size=(n,1))


#%% Create a dummy estimator, with the same structure as in the main code
scaler = StandardScaler()
scaler.fit(X)

tree = DecisionTreeRegressor()
tree.fit(scaler.transform(X), y)

# This is a single estimator
estimator = Pipeline([('scaler', scaler), ('tree', tree)])

# This is a dictionary of estimators (just one in this case), with the same 
# structure as Simulator.Node.Model.estimators_
ID = ('Node0', 0)
estimators_ = {}
estimators_[ID] = estimator



#%% Test
filename = 'test_serialize'
write_json(estimators_[ID], ID, filename)    
estimator_reborn = read_json(filename)


# Check if the two estimators give the same prediction on X
print(estimators_[ID].predict(X))
print(estimator_reborn.predict(X))

