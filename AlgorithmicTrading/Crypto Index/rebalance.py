#import cbpro
import json
import ast
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter().pprint

info_file = open('info.txt')
info = ast.literal_eval(info_file.read())
info

###################
demo = False
###################

if demo == False:   
    auth_client = cbpro.AuthenticatedClient(info['api_key'], info['api_secret'],
                                            info['passphrase'])#, demo_url)
else:
    auth_client = cbpro.AuthenticatedClient(info['demo_api_key'], info['demo_api_secret'], 
                                            info['demo_passphrase'], info['demo_url'])

# ###################

def get_coin_market_cap():
    ## Crypto Assets (basket)
    coin_market_cap_key = 'd3b73307-1958-4d59-93aa-799373ebee59'

    ## Poll coinmarket cap for capitalisations 
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
        'start' : '1',
        'limit' : '50',
        'sort_dir' : 'desc'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': coin_market_cap_key,
    }
    session = Session()
    session.headers.update(headers)
    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
        #pp(data)
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)

    ## Select only the ticker, name, market cap and circulating supply
    market_cap = {curr['symbol']:{'full_name':curr['slug'],'market_cap':curr['quote']['USD']['market_cap'],'circulating':curr['circulating_supply']} for curr in data['data']}
    return market_cap

# # Test ##
# market_cap = get_coin_market_cap()


def get_coinbase_coins():
    ## Get Coinbase products
    coinbase_products = auth_client.get_products()
    coinbase_base_currencies = {coin['base_currency'] for coin in coinbase_products}
    coinbase_quote_currencies = {coin['quote_currency'] for coin in coinbase_products}
    coinbase_coins = {*coinbase_base_currencies, *coinbase_quote_currencies}
    coinbase_coins = [coin for coin in coinbase_coins if coin not in ['USD','GBP','EUR']]
    #print(coinbase_coins, len(coinbase_coins))
    return coinbase_coins

# # Test ##
# coinbase_coins = get_coinbase_coins()


def combine_cmc_coinbase():
    market_cap = get_coin_market_cap()
    coinbase_coins = get_coinbase_coins()
    ## Combine coinmarket cap with available coinbase coins
    coinbase_coins_market_cap = {}
    for cmc_coin in list(market_cap.keys()):
        if cmc_coin in coinbase_coins:
            market_cap[cmc_coin]['coinbase_code'] = cmc_coin
            coinbase_coins_market_cap[cmc_coin] = market_cap[cmc_coin]
        else:
            print('cannot find coinbase coin',cmc_coin)
    return coinbase_coins_market_cap

# # Test ##
# coinbase_coins_market_cap = combine_cmc_coinbase()


def top_market_cap_coins(number_coins):
    coinbase_coins_market_cap = combine_cmc_coinbase()        
    ## take only the top x coins
    top_coins = list(coinbase_coins_market_cap.keys())[:number_coins]
    coinbase_coins_market_cap = {coin:values for coin,values in coinbase_coins_market_cap.items() if coin in top_coins}
    return coinbase_coins_market_cap

# # Test ## 
# number_coins = 4
# coinbase_coins_market_cap = top_market_cap_coins(number_coins)


def init_portfolio_weighting(min_weight, number_coins):
    coinbase_coins_market_cap = top_market_cap_coins(number_coins)
    ## Portfolio weighting
    ### Set a minuimum weight for any currency
    #min_weight = 0.25
    low_weight_coins = []
    added_weight = 0
    total_cap = sum([coin['market_cap'] for coin in list(coinbase_coins_market_cap.values())])
    for coin,values in coinbase_coins_market_cap.items():
        weighting = values['market_cap'] / total_cap
        if weighting < min_weight:
            low_weight_coins.append(coin)
            added_weight += (min_weight - weighting)
            weighting = min_weight
        coinbase_coins_market_cap[coin]['weighting'] = weighting
    return coinbase_coins_market_cap, low_weight_coins, added_weight

# # Test ##
# number_coins = 4
# min_weight = 0.25
# coinbase_coins_market_cap, low_weight_coins, added_weight = init_portfolio_weighting(min_weight, number_coins)


def ascending_weighting():
    ## make dictionary that is sorted reverse by weighting
    reverse_coins = list(coinbase_coins_market_cap.keys())[::-1]
    reverse_coinbase_coins_market_cap = {coin:coinbase_coins_market_cap[coin] for coin in reverse_coins}
    return reverse_coinbase_coins_market_cap

# # Test ##
# number_coins = 4
# min_weight = 0.25
# coinbase_coins_market_cap, low_weight_coins, added_weight = init_portfolio_weighting(min_weight, number_coins)
# reverse_coinbase_coins_market_cap = ascending_weighting()


def set_portfolio_weighting():
    ## reduce highest weight coins to compensate for minuimum
    print(low_weight_coins, added_weight)
    sum_weight_high_coins = sum([values['weighting'] for coin,values in coinbase_coins_market_cap.items() if coin not in low_weight_coins])
    print('sum_weight_high_coins',sum_weight_high_coins)
    for coin,values in reverse_coinbase_coins_market_cap.items():
        if coin not in low_weight_coins:
            weighting = coinbase_coins_market_cap[coin]['weighting']
            print(coin,'original weighting',weighting)
            decrease = (weighting / sum_weight_high_coins) * added_weight
            print(coin,'proposed descrease',decrease)
            if (weighting - decrease) < min_weight:
                diff = min_weight - (weighting - decrease)
                print(coin,'below minuimum by',diff)
                decrease = decrease - diff
                sum_weight_high_coins -= (weighting - decrease)
                print(coin,'amended decrease',decrease)
            coinbase_coins_market_cap[coin]['weighting'] -= decrease
    total_weight = sum([coin['weighting'] for coin in list(coinbase_coins_market_cap.values())])
    print(total_weight,'should equal 1')
    return coinbase_coins_market_cap

# # Test ##
# number_coins = 4
# min_weight = 0.25
# coinbase_coins_market_cap, low_weight_coins, added_weight = init_portfolio_weighting(min_weight, number_coins)
# reverse_coinbase_coins_market_cap = ascending_weighting()
# coinbase_coins_market_cap = set_portfolio_weighting()


def plot_weighting():
    ## weighting
    coins = list(coinbase_coins_market_cap.keys())
    caps = [cap['weighting']*100 for cap in coinbase_coins_market_cap.values()]
    plt.figure(num=1,figsize=(20,6))
    plt.bar(range(len(coins)),caps,tick_label=coins)
    plt.show()

# # Test ##
# number_coins = 4
# min_weight = 0.25
# coinbase_coins_market_cap, low_weight_coins, added_weight = init_portfolio_weighting(min_weight, number_coins)
# reverse_coinbase_coins_market_cap = ascending_weighting()
# coinbase_coins_market_cap = set_portfolio_weighting()
# plot_weighting()


def deposit_account(curr, percentage_under_algo_management):
    account = [acc for acc in auth_client.get_accounts() if acc['currency'] == curr][0]
    initial_investment = float(account['available']) #1000
    investment_capital = initial_investment * percentage_under_algo_management
    print('investment_capital',investment_capital)
    return investment_capital

# # Test ##
# deposit_currency = 'GBP'
# percentage_under_algo_management = 0.95
# investment_capital = deposit_account(deposit_currency, percentage_under_algo_management)


## list of transaction needed to distribute capital into portfolio
transactions = {
    'GBP': {
        'BTC':[('buy','BTC-GBP')],
        'ETH':[('buy','ETH-GBP')],
        'XRP':[('buy','BTC-GBP'),('buy','XRP-BTC')],
        'BCH':[('buy','BCH-GBP')],
        'LTC':[('buy','LTC-GBP')],
        'XLM':[('buy','BTC-GBP'),('buy','XLM-BTC')],
        'ETC':[('buy','ETC-GBP')],
        'ZEC':[('buy','BTC-GBP'),('sell','BTC-USDC'),('buy','ZEC-USDC')],
        'BAT':[('buy','BTC-GBP'),('sell','BTC-USDC'),('buy','BAT-USDC')],
       'USDC':[('buy','BTC-GBP'),('sell','BTC-USDC')],
        'EOS':[('buy','BTC-GBP'),('buy','EOS-BTC')],
        'MKR':[('buy','BTC-GBP'),('buy','MKR-BTC')],
    }
}


def get_coinbase_currencies():
    coinbase_currencies = auth_client.get_currencies()
    coinbase_currencies = {pair['id']:pair for pair in coinbase_currencies}
    return coinbase_currencies

# # Test ##
# coinbase_currencies = get_coinbase_currencies()


def get_coinbase_products():
    coinbase_products = auth_client.get_products()
    coinbase_products = {pair['id']:pair for pair in coinbase_products}
    return coinbase_products

# # Test ##
# coinbase_products = get_coinbase_products()


def find_rounding (str_decimal):
    exponent = int(1/float(str_decimal))
    exponent = str(exponent).split('1')
    if len(exponent) == 1:
        if '9' not in exponent[0]:
            print(exponent, 'looks wierd')
            rounding = 0
        else:
            rounding = len(exponent[0])
    else:
        rounding = len(exponent[1])
    return rounding

# # Test ##       
# print(find_rounding('0.0001'))


def calculate_transacations_queue():
    fees = transaction_fee + slippage
    transaction_queue = []
    skip_coins = []
    total_num_coins = len(coinbase_coins_market_cap)

    for coin, values in coinbase_coins_market_cap.items():
        base_value_amount = investment_capital * values['weighting'] * (1 - fees)
        amount_to_transact = base_value_amount
        ## traverse across list of exchange pairs needed to acquire final coin
        print('-----',coin)
        for side, pair in transactions[deposit_currency][coin]:
            num_transactions = len(transactions[deposit_currency][coin])
            index = transactions[deposit_currency][coin].index((side,pair))+1
            seq = -(index / num_transactions)
            print('transaction',index,'of',num_transactions)
            ticker = auth_client.get_product_ticker(product_id=pair)
            print(ticker)
            print('>>',coin, amount_to_transact, pair)#, ticker)
            base_pair = pair.split('-')
            try:
                if side == 'buy':
                    size = amount_to_transact/ float(ticker['ask'])
                elif side == 'sell':
                    size = amount_to_transact/ float(ticker['bid'])  
            except:
                print('ticker not available, so skipping', ticker)
                skip_coins.append(coin)
            
            min_size = float(coinbase_products[pair]['base_min_size'])
            if size < min_size:
                print(size,'size is too small for',pair,'min size is',min_size)
                skip_coins.append(coin)
            
            rounding = find_rounding(coinbase_currencies[base_pair[0]]['min_size'])
            #print('rounding',rounding)
            if rounding > 0:
                size = round(size, rounding)
            else:
                size = int(size)
                
            amount_to_transact = size if side == 'buy' else size* float(ticker['bid'])
            print(total_num_coins-seq, side, pair, size)
            transaction_queue.append({"seq":total_num_coins-seq,"coin":coin,"base_value_amount":base_value_amount,"side":side, "pair":pair, "size":size})
        total_num_coins-=1            
    return transaction_queue, skip_coins

# # Test ##
# number_coins = 4
# min_weight = 0.25
# coinbase_coins_market_cap, low_weight_coins, added_weight = init_portfolio_weighting(min_weight, number_coins)
# reverse_coinbase_coins_market_cap = ascending_weighting()
# coinbase_coins_market_cap = set_portfolio_weighting()
# plot_weighting()

# deposit_currency = 'GBP'
# percentage_under_algo_management = 0.95
# investment_capital = deposit_account(deposit_currency, percentage_under_algo_management)

# coinbase_currencies = get_coinbase_currencies()
# coinbase_products = get_coinbase_products()

# transaction_fee = 0.003
# slippage = 0.003
# transaction_queue, skip_coins = calculate_transacations_queue()
# final_list = sorted(transaction_queue, key=lambda k:k['seq'])


def send_coinbase_orders():
    ## send orders to Coinbase
    sum_base_value = sum([trade['base_value_amount'] for trade in final_list if trade['seq'] % 1 == 0])
    print('expected raw cost',sum_base_value)
    print('available capital',investment_capital)
    print(sum_base_value/investment_capital)

    #var
    price_threshold = 0.05 #% into book to place order

    num_trades = len(final_list)
    for trade in final_list:
        if trade['coin'] in skip_coins:
            print(trade['coin'],'not included',trade)
            continue
        order_func = auth_client.buy if trade['side'] == 'buy' else auth_client.sell
        
        ## get current bid/ask to send limit order as coinbase sevearly limits market order account size
        ticker = auth_client.get_product_ticker(product_id=trade['pair'])
        if side == 'buy':
            order_price = float(ticker['ask']) * (1 + price_threshold)
        elif side == 'sell':
            order_price = float(ticker['bid']) / (1 - price_threshold)
        
        print({**trade,'order_price':order_price})
        
        ### FILTER TEST ##
        #if trade['pair'] == 'XRP-BTC':
        #    order = order_func(price= price, size= 40,order_type= 'limit',product_id= trade['pair'])
        #    print(order)
            
        ### send orders ##    
        #order = order_func(price = price, size= trade['size'],order_type= 'market',product_id= trade['pair'])
        #    print(order)

# # Test ##
# number_coins = 4
# min_weight = 0.25
# coinbase_coins_market_cap, low_weight_coins, added_weight = init_portfolio_weighting(min_weight, number_coins)
# reverse_coinbase_coins_market_cap = ascending_weighting()
# coinbase_coins_market_cap = set_portfolio_weighting()
# plot_weighting()

# deposit_currency = 'GBP'
# percentage_under_algo_management = 0.95
# investment_capital = deposit_account(deposit_currency, percentage_under_algo_management)

# coinbase_currencies = get_coinbase_currencies()
# coinbase_products = get_coinbase_products()

# transaction_fee = 0.003
# slippage = 0.003
# transaction_queue, skip_coins = calculate_transacations_queue()
# final_list = sorted(transaction_queue, key=lambda k:k['seq'])   
# send_coinbase_orders() 


# # test trades

# auth_client.buy(price = 0.00005445, size = 40, order_type = 'limit',product_id = 'XRP-BTC')
# auth_client.buy(price = 0.000056, size = 40, order_type = 'limit',product_id = 'XRP-BTC')


#def portfolio_value_in_base(base_curr):
balanced_accounts = [account for account in auth_client.get_accounts() if float(account['balance']) > 0]
print(balanced_accounts)
