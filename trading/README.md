# The Alpaca API Explained for People Who Want to Get Started Trading
#### You don't have to be an economist to get power out of Alpaca

Available here: https://medium.com/analytics-vidhya/the-alpaca-api-explained-for-people-who-want-to-get-started-trading-7e57f0af7a

Important: marketwatch.com, the backend for how we previously were getting live stock prices now blocks webscrapers.  This may be possible to bypass, but currently both `simple_algo.py` and `simple_algo.ipynb` are rendered useless because of this.

## Requirements
* Tested in Python 3.7.7 (pretty much any version of Python 3 should work)
* beautifulsoup4==4.9.0
* alpaca-trade-api==0.46

### Conda Instructions
```
conda create -n trading -y python=3.7.7
conda activate trading
conda install -y -c anaconda beautifulsoup4==4.9.0
```

As of writing, installing the Alpaca Trade API with Anaconda doesn't appear to be working.  You can install it into your Conda environment with pip using: `/your/path/to/anaconda/envs/trading/bin/pip install alpaca-trade-api==0.46`.  For example, I would personally run: `~/anaconda3/envs/trading/bin/pip install alpaca-trade-api==0.46`
