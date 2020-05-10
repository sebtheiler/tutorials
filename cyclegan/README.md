# Transforming the World Into Paintings with CycleGAN
#### In Keras and Tensorflow 2.0

Available here: https://medium.com/analytics-vidhya/transforming-the-world-into-paintings-with-cyclegan-6748c0b85632

`ResNet-generator.py` is the untested implementation of the modified ResNet generator used in the paper.  In the article we use a modified U-Net generator, as seen in `cyclegan.*`

Thank you to [Yury Petrov](https://medium.com/@petrovy1) for spotting some mistakes in the ResNet generator!

## Requirements
* Tested in Python 3.7.7 (pretty much any version of Python 3 should work)
* tensorflow==2.1.0
* matplotlib==3.2.0
* tensorflow_addons==0.9.1
* tensorflow_datasets==1.2.0

The code will likely still work with other versions of these packages, but that hasn't been tested.  If you have a capable GPU, you can substitute `tensorflow` with `tensorflow-gpu`; this is highly recommended, training a CycleGAN takes a long time even with a GPU.

### Conda Instructions
```
conda create -n cyclegan -y python=3.7.7
conda activate cyclegan
conda install -y tensorflow==2.1.0
conda install -y -c anaconda tensorflow-datasets==1.2.0
conda install -y -c conda-forge matplotlib==3.2.0
```

`tensorflow_addons` is required for the InstanceNormalization layer.  Unfortunately, it isn't available directly through Anaconda, and must be installed with pip.  If you wish to install it in your Anaconda environment you can use: `/your/path/to/anaconda/envs/cyclegan/bin/pip install tensorflow-addons==0.9.1`.  Given my installation of Anaconda, I use, `~/anaconda3/envs/cyclegan/bin/pip install tensorflow-addons==0.9.1`.
