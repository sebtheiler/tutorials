# Multiprocessing for Data Scientists in Python
#### Why pay for a powerful CPU if you can’t use all of it?

Available here: https://medium.com/analytics-vidhya/multiprocessing-for-data-scientists-in-python-427b2ff93af1

If you ever encounter the error: `FileExistsError: [Errno 17] File exists: 'data'` run:
```
import SharedArray
SharedArray.delete('data')
```

Download the example files for `multiprocessing-load_images.py` from [here](https://www.kaggle.com/c/understanding_cloud_organization/data).

## Requirements
* Tested in Python 3.7.7 (pretty much any version of Python 3 should work)
* sharedarray==3.2.1
* opencv==3.4.2

The code will likely still work with other versions of these packages, but that hasn't been tested.

### Conda Instructions
```
conda create -n multiprocessing -y python=3.7.7
conda activate multiprocessing
conda install -y -c conda-forge sharedarray==3.2.1
conda install -y opencv==3.4.2
```
