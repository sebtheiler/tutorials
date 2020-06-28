# Building a Powerful DQN in TensorFlow 2.0 (explanation & tutorial)
#### And scoring 350+ by implementing extensions such as double dueling DQN and prioritized experience replay

Available here: https://medium.com/analytics-vidhya/building-a-powerful-dqn-in-tensorflow-2-0-explanation-tutorial-d48ea8f3177a

# What Does a DQN Think?
#### A brief visualization of how a DQN chooses to play Breakout

Available here: https://medium.com/analytics-vidhya/what-does-a-dqn-think-4f9c9517f7ed

Explanation of files:
* `train_dqn.ipynb`: The Jupyter Notebook for training the DQN
* `train_dqn.py`: The regular Python file version of the Jupyter Notebook. This is also where we import classes from.
* `config.py`: The parameters and config for the model and environnment
* `evaluation.py`: Load a saved agent and watch it play
* `visualize.py`: Watch the DQN's "thoughts" as it plays (code for second article)
* `visualize-replaybuffer.py`: Turn the DQN's experiences into a scatterplot (code for second article)

---
 
I would like to give a shout-out to both [this](https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb) notebook by Fabio M. Graetz, and [this](https://www.youtube.com/watch?v=5fHngyN8Qhw) video by Machine Learning with Phil for inspiring this project and getting me started with the code. Both sources have been useful countless times, and I feel safe to say this article would not exist without them.

There is currently a slight bug where the target network and main network are updated at a rate controlled by the same variable.  To see the fixed version, please head to the branch `dqn-fix`.  I'm currently away and cannot test the fix to completion (although it appears to work as expected), but as soon as it is fully tested `dqn-fix` will be merged.

## Requirements (important)
* Tested in Python 3.7.7 (pretty much any version of Python 3 should work)
* tensorflow-gpu==2.1.0
* opencv==3.4.2
* gym==0.17.1
* matplotlib

**TensorFlow 2.0.0 has a memory leak.  Make sure to use TensorFlow 2.1.0.**  The code will likely still work with other versions of these packages, but that hasn't been tested. If you don't have a capable GPU, you can substitute `tensorflow-gpu` with `tensorflow`; this is not recommended as training a DQN takes a long time even with a GPU.

### Conda Instructions
```
conda create -n dqn -y python=3.7.7
conda activate dqn
conda install -y tensorflow-gpu==2.1.0 opencv==3.4.2
conda install -y -c conda-forge matplotlib
```

Installing OpenAI Gym with pip is far more convenient even when using Anaconda.  To install Gym with pip inside of your Anaconda environment use: `/your/path/to/anaconda/envs/dqn/bin/pip install gym[atari]==0.17.1`.  Given my installation of Anaconda, I use `~/anaconda3/envs/dqn/bin/pip install gym[atari]==0.17.1`.
