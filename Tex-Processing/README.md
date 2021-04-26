# Label_Text_Builder
Creates a dictionary from Latex code organizing it into book, chapter, section, subsection/subsubsection, and paragraph.
Also add the tags associated with each atom into the dictionary.

* books: Expects Latex files with the command **\tag{}** at the end of each atom. The braces contain tags associated with each atom. 
* vocab_save_path: Where to save the vocabulary pickle file.
* data_save_path: Where to save the data dictionary pickle file.
## Usage

Instantiate the class.
```python
LTB = Label_Text_Builder(books, vocab_save_path='vocab.pkl', data_save_path='data_dict.pkl')
```

Simply run the main() method. If the argument the_count is set to True, the integer number of words in total.
```python
LTB.main(the_count=True)
```


# Tex_Processing
Class that removes symbols, numbers from Latex code.
Tokenizes, lemmatizes, ranks vocab, and encodes paragraphs.

## Usage

Instantiate the class with the data_dir and vocab_dir.
These are paths to the data dictionary and the vocab pickle files created from the [Label_Text_Builder](https://github.com/dialectic/Metastimuli-Project/blob/master/Tex-Processing/Label_Text_Builder.py)

```python
PCS = Tex_Processing(data_dir='data_dict.pkl', vocab_dir='vocab.pkl')
```


If a vocabulary has not been built in Label_Text_Builder or elsewhere it can be built in Tex_Processing.
The rank_vocab() method sorts the vocabulary by frequency in the text provided in the data_dir.
The cutoff argument eliminates words with a frequency of that value or lower. 
Must be a natural number.
Method encode_paras replaces the words in data_dict with the vocab indices.
The zero index is reserved for 'unk' words.
```python
vocab_dict = PCS.build_vocab()
ranked_vocab = PCS.rank_vocab(vocab_dict, cutoff)
enc_atoms = PCS.encode_paras
```
## Contributions

## Licenses


