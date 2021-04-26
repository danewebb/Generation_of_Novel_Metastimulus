# Sciart_Word_Embedding
Creates the Keras word embedding for the Tensorflow dataset [Scientific_Papers](https://www.tensorflow.org/datasets/catalog/scientific_papers).
This class has only been tested with Scientific_Papers but may work with other datasets.

## Usage
* data_chunks: encoded data in numpy arrays
* model_path: Path for a Keras model. To continue training an already defined model.
* labels: Do not define unless the entire NN is desired rather than just the word embeddings.
* vocab_path: path to fully processed vocab
* batch_size: How many examples per forward pass.
* set_epochs: How many epochs to train the word embedding
* embedding_dim: The number of embedding dimensions. This is the input dimension for the [Learn_Master](https://github.com/dialectic/Metastimuli-Project/blob/master/Learn_Master/Learn_Master.py) meta-parameters.
* save_model_path: Where to save the Keras word embedding model.

Instantiate the class with the data, vocab, model_save_path, and the embedding dimension.
```python
SWE = Sciart_Word_Embedding(data_chunk='data_01.pkl', vocab_path='clean_vocab.pkl', model_save_path=model_dim10, embedding_dim=10, set_epochs = 1000)
```

The method main_train() trains the Keras word embedding with the data given.
```python
SWE.main_train()
```

```python
SWE.retrieve_word_embeddings('model_path', 'vector_save_path', 'meta_save_path')
```
# Word_Embedding_PIMS
Creates a Keras model for a word embedding.
ANN neural network is quite simple but all that is needed to create a word embedding is an Embedding layer.
Neither labels nor layer information needs to be defined to create a valid word embedding layer.

## Usage
* encoded_data: The PIMS corpus encoded data. The data is encoded with the vocab indices.
* ranked_vocab: vocab that is used to create the encodings of encoded_data.
* batch_size: How many examples per forward pass.
* set_epochs: How many epochs to train the word embedding
* embedding_dim: The number of embedding dimensions. This is the input dimension for the [Learn_Master](https://github.com/dialectic/Metastimuli-Project/blob/master/Learn_Master/Learn_Master.py) meta-parameters.
* save_location: Path to save the Keras word embedding
* model_path: To continue training an existing model.

Instantiate the class
```python
WE = Word_Embedding_PIMS(data, vocab, 20, save_location='model_path', model_path = 'model_path')
```

Calling the method train_embedding() with the arguments defined in the instantiation trains and save a word embedding.
```python
WE.train_embedding()
```


# Weighting_Keywords
This class finds keywords in atoms and assign weights to their word embeddings.

## Usage

* vocab: Vocabulary of the data containing the keywords.
* weight: The numerical weight factor to apply to the keyword word embeddings
* keywords: list of keywords not hardcoded into the class.

Instantiate the class with the vocab and the weight factor.
```python
WK = Weighting_Keyword(vocab, weight)
```

First, a keyword_search() method must be called to find the keywords in the vocabulary and assign the encoded values to them.
Then, apply_weights() method finds the keywords in the atoms and applies the weighting factor.
```python
WK.keyword_search()
weighted_atoms = WK.apply_weights(atoms)
```

# Contributions

# Roadmap

# Licenses


