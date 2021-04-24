# Atom_Embeddings

<!--- 
Take out the PVDM scripts
-->

## Atom_Embedder

File atom_embedding.py contains the class Atom_Embedder that conducts the various atom embeddings from word embeddings.

Available atom embedding methods:
* Bag-of-words (average)
* Bag-of-words (sum)
* Nabla
* PVDM


## Usage
Instantiate the class with word embedding weights from a Keras word embedding model and vocab in a list ordered from most frequent to least.

```python
import keras
from atom_embedding import Atom_Embedder

we_model = keras.models.load_model('keras_model_path')
weights = we_model.layers[0].get_weights()[0]
vocab = load('vocab_path')
AE = Atom_Embedder(weights, vocab)
embedded_atoms = []
```


Both of the bag-of-words versions, sum_atoms and avg_atoms, are used identically. Each method expects a list of integers that represent the frequency encoded words. The variable encoded_data is a list of lists of the frequency encoded words that make up each atom.
```python
# Bag of words (sum)
for atom in encoded_data:
  embedded_atoms.append(AE.sum_atoms(atom))
```

Currently, sum_atoms(), avg_atoms(), and nabla() methods allow for a keyword weighting. 
Keyword weighting is handled by the Weighting_Keyword class in [Word_Embeddings](https://github.com/dialectic/Metastimuli-Project/blob/master/Word_Embeddings/Weighting_Keywords.py).
Weighting keywords for PVDM embeddings may be in a future update.
```python
from Weighting_Keywords import Weighting_Keyword

keyword_weight_factor = 10
WK = Weight_Keyword(vocab, keyword_weight_factor)
WK.keywords_in_vocab()

for atom in encoded_data:
  atom_weights = WK.keyword_search(atom)
  embedded_atoms.append(AE.sum_atoms(atom), weights=atom_weights)
```

The nabla() method has an additional required argument *ndel*. 
The ndel argument determines how many *differences* are taken for the sum of the difference.
```python
# Nabla
### TODO change method name from sum_of_ndelta to nabla
ndel = 5
for atom in encoded_data:
  embedded_atoms.append(AE.nabla(atom, ndel, weights=atom_weights))
```

The PVDM() method must be first trained using the cleaned but unencoded text. 
The package Gensim handles the word embeddings and atom embeddings internally.
The second argument is a holdover from a previous version and is deprecated. 
For now, insert a *1*.
```ptyhon
# PVDM
raw_atoms = load('raw_atoms_path')
AE.pvdm_train(raw_atoms)
for ratom in raw_atoms:
  embedded_atoms.append(AE.pvdm(ratom, 1))

```
## Atom_Tag_Pairing
Atom_Tag_Pairing has numerous methods that deal with labels.

## Usage

```python
data_dict: Expects dictionary format like the dictionary returned from 
<pre>
<a href="https://github.com/dialectic/Metastimuli-Project/blob/master/Tex-Processing/Label_Text_Builder.py">Label_Text_Builder</a>
</pre>

```

## Contributing

## Roadmap

## Licenses

