#Atom_Embeddings
<!--- 
Take out the PVDM scripts
-->

## Atom_Embedder

File atom_embedding.py contains the class Atom_Embedder that conducts the various atom embeddings from word embeddings.

Available atom embedding methods:
* Bag of words (average)
* Bag of words (sum)
* Nabla
* PVDM


## Usage
Instantiate the class with word embedding weights from a Keras word embedding model and vocab in a list ordered from most frequent to least.

```python
import keras
from atom_embedding import

we_model = keras.models.load_model('keras_model_path')
weights = we_model.layers[0].get_weights()[0]
vocab = load('vocab_path')
AE = Atom_Embedder(weights, vocab)
```




## Atom_Tag_Pairing

## Usage


## Contributing

## Roadmap

## Licenses

