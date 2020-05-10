# Building a Twitter Sentiment-Analysis App Using Streamlit
#### "The fastest way to build custom ML tools"

Available here: https://medium.com/analytics-vidhya/building-a-twitter-sentiment-analysis-app-using-streamlit-d16e9f5591f8

* `main.py`: Streamlit app.  Use `streamlit run main.py` to run the app.
* `sentiment_analysis.ipynb`: Jupyter notebook version of the file to train the model.
* `sentiment_analysis.py`: Python file version of the file to train the model.

Download the Sentiment140 dataset from its Kaggle page [here](https://www.kaggle.com/kazanova/sentiment140#training.1600000.processed.noemoticon.csv).

## Requirements
* Tested in Python 3.7.7 (pretty much any version of Python 3 should work)
* flair==0.4.5
* streamlit==0.59.0

The code will likely still work with other versions of these packages, but that hasn't been tested.

### Conda Instructions
Unfortunately, both Flair and Streamlit are difficult to install with just Conda.  If you still want to use a Conda venv, you can do something like:
`~/anaconda3/envs/twitter-sentiment/bin/pip install flair==0.4.5 streamlit==0.59.0`
And just edit the installation path to your Conda.
