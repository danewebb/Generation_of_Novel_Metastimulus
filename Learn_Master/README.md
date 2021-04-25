# Learn_Master

## Learn_Master
Learn_Master is the parent training class for the personal information management system's artificial neural network.
While there are many default arguments for the class, it is not necessary to define all of them.


* dataset_dict: Dictionary with dataset information. Dictionary made from [Label_Text_Builder](https://github.com/dialectic/Metastimuli-Project/blob/master/Tex-Processing/Label_Text_Builder.py)
* vocab: List of word embedding vocab ordered by frequency.
* ordered_encdata: List of data in the original order encoded with the indices of the vocab argument.
* ordered_labels: Labels matched with the data from ordered_encdata. Made from [Atom_Tag_Pairing](https://github.com/dialectic/Metastimuli-Project/blob/master/Atom_Embeddings/atom_tag_pairing.py)
* input_dims: List of input dimensions. Must line up with dimensions of the word embedding models of we_models.
* we_models: Keras word embedding model paths.
* output_dims: List of projection output dimensions. Lines up with labels.
* keyword_weight_factor: Integer value that is used to weight keywords. A value of one results in no weighting applied.
* optimizer: List of gradient optimizers 'sgd', 'adamax', 'adagrad', 'adam', 'adadelta', 'rmsprop'
* atom_methods: list of atom methods 'bowsum', 'bowavg', 'ndelta'
* nn_architectures: list of NN architectures 'ff', 'rnn'
* hyperoptimizers: list of hyperparameter optimizer methods 'random', 'bayes', 'hyper'


## Usage

Instantiating the Learn_Master class includes a lot of arguments. 
The objective variables include from the union of the input_dims and we_models to the hyperoptimizers arguments.
All of the default argument have to do with the genetic algorithm, checkpoint saving, or recurrent neural network specific variables.

For the default arguments, special attention should be paid to the fitness_epochs and maxiter arguments. 
The fitness_epochs determines the length of time the Keras Tuner search algorithms spend searching 
```python
LM = Learn_Master(
dataset_dict, vocab, ordered_encdata, ordered_labels, input_dims, we_models, output_dims, keyword_weight_factor, optimizers, atom_methods, nn_architectures, hyperoptimizers, fitness_epochs=100, maxiter=100, savepath_model=model_paths, rnn_steps=3, kt_master_dir=kt_master_dir, checkpoint_filepath=checkpoint_filepath, raw_paras=raw_paras, fitness_savepath=fit_savepath
)
```

Finding an optimal hyper-parameters and meta-parameters begins with the gentic algorithm in the genetic_alg() method.
* npop: Number of meta-parameter sets for each generation. Must be even.
* numparents: Must be half of npop.
* mutation_rate: Range <pre>[0, 1]</pre> determines gene mutation rate. Higher mutation rate -> random gene selection while lower mutation rate can lead to early optimum choice.
* stagnate: Bool, if stagnate==True the mutation rate will decrease by <img src="https://bit.ly/32MvCQE" align="center" border="0" alt="M = M_{0} ( \frac{T - t}{T} )" width="179" height="65" />


```python
npop = 20
numparents = npop/2
mutation_rate = 0.5
ga_run_name = 'ga_run_00'
fit_path = 'fitness.pkl'
gasaver = 'gasaver.pkl'
saver_every = 5
LM.genetic_alg(numparents, npop, gasaver, save_every, mutation=mutation_rate, stagnate=True, trainfor=25, n_nulls=0)
```


## Contributions

## Licenses
