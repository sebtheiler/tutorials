# Basics of Using Pre-trained GloVe Vectors in Python
#### Downloading, loading, and using pre-trained GloVe vectors

Available here: https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db

Download GloVe vectors from [here](https://nlp.stanford.edu/projects/glove/) or get the exact ones used in the article from [here](http://nlp.stanford.edu/data/glove.6B.zip) (822MB Download)

## Requirements
* Tested in Python 3.7.7 (pretty much any version of Python 3 should work)
* scipy==1.3.0
* matplotlib==3.2.0

The code will likely still work with other versions of these packages, but that hasn't been tested.

### Conda Instructions
```
conda create -n using-pretrained-glove-vectors -y python=3.7.7
conda activate using-pretrained-glove-vectors
conda install -y scipy==1.3.0
conda install -y -c conda-forge matplotlib==3.2.0
```
