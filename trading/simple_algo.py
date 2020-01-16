# Hyperparameters
stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB', 'BABA', 'INTC', 'NVDA', 'CRM', 'PYPL', 'TSLA', 'AMD', 'ATVI', 'EA', 'MTCH', 'TTD', 'ZG', 'YELP', 'TIVO']
num_stocks_to_invest_in = 10
DAYS = 100
sell_all_at_market_close = True

# Initialize API
import alpaca_trade_api as tradeapi

API_KEY = 'your-key-id'
SECRET_KEY = 'your-secret-key'
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(key_id=API_KEY, secret_key=SECRET_KEY, base_url=BASE_URL, api_version='v2')

account = api.get_account()
budget = float(account.buying_power)
print(f'Your budget is: {budget}')

# Get live price of share
import bs4 as bs
import requests

message = """At the time of writing https://marketwatch.com/robots.txt does not disallow web scraping https://www.marketwatch.com/investing/stock/*. With this said, I cannot guarantee this for the future, so please check their robots.txt before continuing. I am not responsible for any actions you may take.  After verifying that marketwatch.com still allows scraping this part of their site, please manually set I_UNDERSTAND (line 25) to True"""
I_UNDERSTAND = False

if not I_UNDERSTAND:
    raise UserWarning(message)

def get_live_price(ticker):
    while True:
        try:
            # Send request to marketwatch.com for the given ticker
            resp = requests.get(f"https://www.marketwatch.com/investing/stock/{ticker.replace('-', '.')}")
            soup = bs.BeautifulSoup(resp.text, features='lxml')

            # Find HTML element on the page
            value = soup.find('bg-quote', {'class': 'value'})

            # Read its value
            return float(value.text.replace(',', ''))
        except Exception:
            continue

# Get barset data of stock(s) into pd.DataFrame format
import pandas as pd

def get_pandas_barset(symbols, timeframe, limit, start=None, end=None, after=None, until=None):
    barset = api.get_barset(symbols, timeframe, limit, start, end, after, until)
    dataframes = {}
    
    for symbol in barset.keys():
        bars = barset[symbol]

        data = {'close': [bar.c for bar in bars],
                'high': [bar.h for bar in bars],
                'low': [bar.l for bar in bars],
                'open': [bar.o for bar in bars],
                'time': [bar.t for bar in bars],
                'volume': [bar.v for bar in bars]}
        
        dataframes[symbol] = pd.DataFrame(data)
    
    return dataframes

# Sell everything owned
print('Selling all current positions...')
api.cancel_all_orders()
positions = api.list_positions()

for position in positions:
    api.submit_order(
        symbol=position.symbol,
        qty=position.qty,
        side='sell',
        type='market',
        time_in_force='gtc'
    )

# Calculate percent increases for each stock in the past d days
print('Calculating increases...')
stock_data = get_pandas_barset(stocks, 'day', DAYS)

percent_increases = [] # could also use ordered dict for this

for symbol in stocks:
    percent_increases.append((symbol, stock_data[symbol].iloc[-1].close/stock_data[symbol].iloc[0].close - 1))

percent_increases = sorted(percent_increases, key=lambda x: x[1], reverse=True)

# Divvy up budget to each stock
print('Calculating how many stocks to buy...')
total_increase_sum = 0

for symbol, increase in percent_increases[:num_stocks_to_invest_in]:
    total_increase_sum += increase

shares_to_buy = {}

for symbol, increase in percent_increases[:num_stocks_to_invest_in]:
    shares_to_buy[symbol] = budget*(increase/(total_increase_sum*get_live_price(symbol)))

print(shares_to_buy)

# Send requests to Alpaca API to buy shares
print('Buying shares...')
import math

for symbol in shares_to_buy.keys():
    if shares_to_buy[symbol] >= 1:
        api.submit_order(
            symbol=symbol,
            qty=math.floor(shares_to_buy[symbol]),
            side='buy',
            type='market',
            time_in_force='gtc'
        )

# Sell everything at the end of the day
import time, datetime

if sell_all_at_market_close:
    clock = api.get_clock()
    seconds = (clock.next_close - clock.timestamp - datetime.timedelta(minutes=10)).total_seconds()
    print('Waiting', seconds, 'seconds until market is nearly closed.')
    time.sleep(seconds)

    api.cancel_all_orders()
    positions = api.list_positions()

    for position in positions:
        api.submit_order(
            symbol=position.symbol,
            qty=position.qty,
            side='sell',
            type='market',
            time_in_force='gtc'
        )