# Metastimuli Learn

## Atom_FFNN
The Atom_FFNN class handles the construction of the feed-forward neural network. 
Currently, many of the methods are deprecated.
Its current use is in the Learn_Master class.

Currently all of the arguments are default but if without

## Usage

* data_path: Path of data, either prediction or training data. Expects numpy arrays of atom embedded data.
* data: Loaded data from above, choose one of these options
* train_label_path: Path of training labels. Must be numpy array. If completing a prediction on a loaded Keras model, this argument is unnecessary.
* train_labels: Loaded data from above. Choose one of these options.
* test_label_path: Path of test data labels. Expects numpy arrays. All test arguments are unnecessary if using the prediction functionality.
* test_labels: Preloaded values from above.
* test_path: Test data path
* test: Test data preloaded from above.
* nullset_path: Path of nullset. Random data within the range of actual data. Do not train using this set.
* nullset: Pre-Loaded nullset from above.
* model_path: Keras neural network model path
* save_model_path: Path to save Keras neural network model
* model: Loaded Keras model.
* batch_size: int Number of examples in the forward pass of training until the backward pass is conducted.
* epochs: int Number of iterations of each training all of the data.
* regression: bool Uses the regression mode
* classification: bool probabilistic mode


Keras Tuner variables
---
Below are Keras Tuner variables. Either leave default values or learn more about them [here](https://keras-team.github.io/keras-tuner/)
 input_min, input_max, input_step, hid_min, hid_max, hid_step, min_hlayers, max_hlayers,
learn_rate_min, learn_rate_max, beta1_min, beta1_max, beta1_step, beta2_min, beta2_max, beta2_step, momentum_min, momentum_max, momentum_step, initial_acc_min, initial_acc_max, initial_acc_step, epsilon_min, epsilon_max, epsilon_step, rho_min, rho_max, rho_step, max_trials, max_executions_per, initial_points, hyper_maxepochs, hyper_factor, hyper_iters,
optimizer, kt_directory

Recurrent neural network specific arguments.
---
steps: int Number of recursive steps.


```python
FFNN = Atom_FFNN()

RNN = Atom_RNN()
```






## Atom_RNN

## Usage
