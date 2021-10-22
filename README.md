# Federated Fraud Detection

This code implements the federated fraud detection model.



### Usage:

* Unpack [example_dataset_5.tar.gz](https://github.com/GabrieleSantin/federated_fraud_detection/blob/main/data/example_dataset_5.tar.gz) in `data/`. This is an example dataset for a simulation with five nodes.
* Run the script [demo.py](https://github.com/GabrieleSantin/federated_fraud_detection/blob/main/code/demo.py) in `code/`.  The demo does the following:
  1. Load the configuration file [fully_connected.yaml](https://github.com/GabrieleSantin/federated_fraud_detection/blob/main/code/config/fully_connected.py) from `code/config/`. This configuration defines a fully connected network of five agents, some parameters for the learning algorithm, and a list of actions to be performed by each node. The actions are equal for each node for simplicity.
  2. Load the example dataset.
  3. Run the simulation.
  4. Compute some accuracy metrics and statistics, and visualize them.



### Structure of the code

The code is organized in four files:

* [demo.py](https://github.com/GabrieleSantin/federated_fraud_detection/blob/main/code/demo.py): The demo described above.

* [utils.py](https://github.com/GabrieleSantin/federated_fraud_detection/blob/main/code/utils.py): Some utility functions used to load the dataset and compute scores. It should be of no interest for the blockchain integration

* [learning_model.py](https://github.com/GabrieleSantin/federated_fraud_detection/blob/main/code/learning_model.py): The learning model implemented in each node. The file defines an abstract class `Classifier` and an implementation `ClassifierLinReg`, which have methods `fit`, ` add_external_estimators`, `get_top_estimators`. These (plus the constructor) are the only interface used by the external world. The implementation is likely to be updated, but this should be of no interest for the blockchain integration.

* [simulator.py](https://github.com/GabrieleSantin/federated_fraud_detection/blob/main/code/simulator.py): The simulation and communication part. The file defines the following classes:

  * `Simulator`: The object that initializes and runs the simulation. It contains the nodes and run them sequentially.

  * `Registry`: The registry used by the nodes to communicate.

  * `Node`: The object representing a single node.

    

