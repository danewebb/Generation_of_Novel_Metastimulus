# Shuffler
The Shuffler class randomizes the data keeping the labels with their atoms. 
Also splits data into training and testing datasets.
There is also a normalization method that reads through all of the data and labels and normalizes values between range <pre>[0, 1]</pre>
 
* x: data
* y: labels
* dims: deprecated
* trsplit: <pre>[0, 1]</pre>  percentage of all data that is part of the training set.
* tesplit: <pre>[0, 1]</pre>  percentage of all data that is part of the testing set.
## Usage
```python
SH = Shuffler()
```

```python
train, test = SH.split()
```
## Roadmap
Remove deprecated arguments, dims


## Contributions

## Licenses
