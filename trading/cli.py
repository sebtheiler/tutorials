import alpaca_trade_api as tradeapi

# Config
API_KEY = 'your-key-id'
SECRET_KEY = 'your-secret-key'
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize API
api = tradeapi.REST(key_id=API_KEY, secret_key=SECRET_KEY, base_url=BASE_URL, api_version='v2')

# Check if market is closed
if not api.get_clock().is_open:
    print('Market is closed.  Try again later.')
    exit()

# CLI
print('==================================')
print('Interactive paper-stock trading')
print('==================================')
while True:
    print('---')
    try:
        # Display account information
        account = api.get_account()
        print('Your cash: ', account.cash)
        print('Your buying power: ', account.buying_power)

        # Get buy/sell input
        buy_or_sell = input('[B]uy or [s]ell? ').lower()

        if buy_or_sell == 'b':
            side = 'buy'
        elif buy_or_sell == 's':
            side = 'sell'
        else:
            print('Unrecognized action.  Try again.')
            continue

        # Get other information
        ticker = input('Ticker of the stock you would like to buy/sell: ')

        quantity = int(input('Quantity of the stock you would like to buy/sell: '))
        
        # Submit order
        try:
            api.submit_order(
                symbol=ticker,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='gtc'
            )
        except tradeapi.rest.APIError as e: # If the API doesn't recognize the stock ticker...
            print('Error:', e)
            continue
            
        print('Order submitted')
        
    except KeyboardInterrupt: # When ctrl+c is pressed...
        print('\nProgram interrupted by user.  Exiting...')
        exit()