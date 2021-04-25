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


While this is mentioned in the roadmap, the deperecated variables and methods need to be deleted or modified to be more useful for the current setup.
The same goes for the Atom_RNN. 
Also, not all of the arguments should be class variables.



```python
FFNN = Atom_FFNN()
```
```python
rand_search_tuner = FFNN.random_search()
bayes_tuner = FFNN.bayesian()
hyperband_tuner = FFNN.hyperband
```


### Keras Tuner training

Keras Tuner variables may be left as default values. The default values were used for the results in [Metastimuli for Human Learning via Machine Learning and Principle Component Analysis of Personal Information Graphs.]

```python
for dim in range(output_dimensions):
 FFNN = Atom_FFNN(
  data = tr_data,
  train_labels = tr_labels,
  test_data = te_data,
  test_labels = te_labels,
  nullset = nu_labels,
  current_dim = dim,
  regression = True,
  batch_size = 100,
  )
  
  tuner_rand = FFNN.random_search()
  best_model_rand = tuner_rand.get_best_models(num_models=1)[0]
  
 FFNN = Atom_FFNN(
  data = tr_data,
  train_labels = tr_labels,
  test_data = te_data,
  test_labels = te_labels,
  nullset = nu_labels,
  current_dim = dim,
  regression = True,
  batch_size = 100,
  epochs = 500,
  model = best_model_rand
  )
  
  history = FFNN.train()
  test_results = FFNN.test()
  null_results = FFNN.test_nullset()
  
  prediction_results = FFNN.predict()
  
```



## Atom_RNN

Recurrent neural network specific arguments.
---
steps: int Number of recursive steps.


## Usage
Names of arguments, methods, and functionality is the same as the Atom_FFNN.

```python
RNN = Atom_RNN()
```

## Roadmap

The next step for both the Atom_FFNN and the Atom_RNN is to clean up the deprecated methods and arguments. 

Greater functionality needs to be built in.
Build custom models from arguments.


## Contributions

## Licenses
