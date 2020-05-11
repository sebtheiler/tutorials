# Implementing a GAN in Keras
#### "the most interesting idea in the last 10 years in ML"

Available here: https://medium.com/analytics-vidhya/implementing-a-gan-in-keras-d6c36bc6ab5f

## Requirements
* Tested in Python 3.7.7 (pretty much any version of Python 3 should work)
* tensorflow==1.15.0
* keras==2.2.4
* matplotlib==3.2.0
* pillow==7.1.2

The code will likely still work with other versions of these packages, but that hasn't been tested.  If you have a capable GPU, you can substitute `tensorflow` with `tensorflow-gpu`.

### Conda Instructions
``` 
conda create -n gans -y python=3.7.7
conda activate gans
conda install -y tensorflow==1.15.0 keras==2.2.4
conda install -y -c conda-forge matplotlib==3.2.0 pillow==7.1.2
```
