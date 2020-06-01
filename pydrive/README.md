# How To Connect Google Drive to Python using PyDrive
#### The Ultimate PyDrive Tutorial

Available here: https://medium.com/analytics-vidhya/how-to-connect-google-drive-to-python-using-pydrive-9681b2a14f20

## Requirements
* Tested in Python 3.7.7 (pretty much any version of Python 3 should work)
* PyDrive==1.3.1
* tensorflow==2.1.0

TensorFlow is only required for Keras callback.  The code will likely still work with other versions of these packages, but that hasn't been tested.  If you have a capable GPU, you can substitute `tensorflow` with `tensorflow-gpu`.

### Conda Instructions
```
conda create -n pydrive -y python=3.7.7
conda activate pydrive
conda install -y -c conda-forge pydrive
conda install -y tensorflow==2.1.0
```

Only install TensorFlow if you're going to be using the Keras callback.
