# Process_Sciart

The Process_Sciart class is built primarily to clean and process the Tensorflow's dataset [Scientific_Papers](https://www.tensorflow.org/datasets/catalog/scientific_papers).
The class may be usable for other Tensorflow datasets but no others have been tested.

Features include tokenization, removal of symbols and numbers, build an ordered vocabulary, lemmatization, and more.

* data_list: List of text data to be cleaned.
* out_datafile_name: save location of clean data.
* vocabfile: list of ranked vocab w/ most common in index 1 and least common in the final index. Index 0 holds a placehodler 0.

## Usage

Instantiation of the class.
```python
PSA = Process_Sciart()
```

The method encode_data_list() handles the full processing of the dataset into paragraph atoms for use with the word embedding/atom embedding methods.
```python
PSA.encode_data_list()
```
## Roadmap

## Contributions

## Licenses
