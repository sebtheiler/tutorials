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
