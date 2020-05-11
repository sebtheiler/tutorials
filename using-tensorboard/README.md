# Basics of Using TensorBoard in TensorFlow 1 & 2
#### Stop using Matplotlib to plot your losses --- visualize graphs & models, filters, losses…

Available here: https://medium.com/analytics-vidhya/basics-of-using-tensorboard-in-tensorflow-1-2-b715b068ac5a

## Requirements
For `tensorboard-examples_tf1.ipynb`:
* Tested in Python 3.7.7 (pretty much any version of Python 3 should work)
* tensorflow==1.15.0
* keras==2.2.4
* keras_contrib==2.0.8

For `tensorboard-examples_tf2.ipynb`:
* Tested in Python 3.7.7 (pretty much any version of Python 3 should work)
* tensorflow==2.1.0

The code will likely still work with other versions of these packages, but that hasn't been tested.  If you have a capable GPU, you can substitute `tensorflow` with `tensorflow-gpu`.

### Conda Instructions
#### TensorFlow 1.x
```
conda create -n using-tensorboard-tf1 -y python=3.7.7
conda activate using-tensorboard-tf1
conda install -y tensorflow==1.15.0
conda install keras==2.2.4
```

`keras_contrib` is required for the InstanceNormalization layer. Unfortunately, it isn't available directly through Anaconda, and must be installed with pip. If you wish to install it in your Anaconda environment you can use: `/your/path/to/anaconda/envs/using-tensorboard-tf1/bin/pip install git+https://www.github.com/keras-team/keras-contrib.git`. Given my installation of Anaconda, I use, `~/anaconda3/envs/using-tensorboard-tf1/bin/pip install git+https://www.github.com/keras-team/keras-contrib.git`.

#### TensorFlow 2.x
```
conda create -n using-tensorboard-tf2 -y python=3.7.7
conda activate using-tensorboard-tf2
conda install -y tensorflow==2.1.0
```
