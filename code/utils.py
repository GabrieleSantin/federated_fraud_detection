import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os.path as osp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


#%%
def almost_equal_split(n_elements, n_agents, offset):
    # Equal splitting plus offset
    split_idx = np.linspace(0, n_elements, n_agents+1).astype('i')
    offset_idx = np.zeros(n_agents+1).astype('i')
    offset_val = int(offset * np.mean(np.diff(split_idx)))
    offset_idx[1:-1] = np.random.randint(-offset_val, offset_val, size=(n_agents-1,))
    split_idx += offset_idx
    
    return split_idx


#%%
def get_data(data_path, data_file, unused_col, label_col, random_state, test_size):
    # Load data
    data = pd.read_csv(osp.join(data_path, data_file))
    
    # Cleaning column names
    rename_pattern = {c: c.lower() for c in data.columns}
    data.rename(columns=rename_pattern, inplace=True)
    
    # Drop unused
    data.drop(columns=unused_col, inplace=True)
    
    #
    X = data.drop(columns=label_col)
    y = data[[label_col]].values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=random_state,
                                                        test_size=test_size)
        
    # Fit the scaler
    input_scaler = StandardScaler()
    input_scaler.fit(X_train) 
    
    return X_train, X_test, y_test, y_train, input_scaler


#%%
def get_scores(model, X_train, y_train, X_test, y_test, verbose=0):
    # Train sanity check
    y_train_pred = model.predict(X_train)
    train_bacc = metrics.balanced_accuracy_score(y_train, y_train_pred, adjusted=False)
    train_precision = metrics.precision_score(y_train, y_train_pred)
    train_recall = metrics.recall_score(y_train, y_train_pred)
    train_confusion = metrics.confusion_matrix(y_train, y_train_pred)

    # Test
    y_test_pred = model.predict(X_test)
    test_bacc = metrics.balanced_accuracy_score(y_test, y_test_pred, adjusted=False)
    test_precision = metrics.precision_score(y_test, y_test_pred)
    test_recall = metrics.recall_score(y_test, y_test_pred)
    test_confusion = metrics.confusion_matrix(y_test, y_test_pred)
    
    # Print the scores and some stats
    if verbose > 0:
        print('-' * 42)
        print('|                 Model: %5s           |' % model.ID)
        print('-' * 42)
        print('|       |    Prec. |   Recall |  B. Acc. |')
        print('-' * 42)
        print('| Train | %2.2e | %2.2e | %2.2e |' % (train_precision, train_recall, train_bacc)) 
        print('|  Test | %2.2e | %2.2e | %2.2e |' % (test_precision, test_recall, test_bacc)) 
        print('-' * 42)
    
    # Store and return the scores
    scores = {}
    scores['train_precision'] = train_precision
    scores['train_recall'] = train_recall
    scores['train_bacc'] = train_bacc
    scores['train_confusion'] = train_confusion
    
    scores['test_precision'] = test_precision
    scores['test_recall'] = test_recall
    scores['test_bacc'] = test_bacc
    scores['test_confusion'] = test_confusion
    
    return scores


