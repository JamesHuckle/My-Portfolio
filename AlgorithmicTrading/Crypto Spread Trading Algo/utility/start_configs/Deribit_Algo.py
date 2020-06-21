# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
#%run "Deribit_Algo_testing.ipynb" ##comment out when running from terminal with notebook merge

# %%
date = datetime.datetime.utcnow()
fut_sym, back_sym = contract_sym(date,roll_buffer=0)

account_name = "spread"
algo = deri_algo(account_name,testing=False)
algo.var.account_name = account_name

## Important vars ##
algo.var.m1_sym = "BTC-PERPETUAL"
algo.var.m2_sym = fut_sym
algo.var.tick = 0.25
algo.var.pay_up_ticks = 200
algo.var.log = "log_Deribit_Algo_"+algo.var.account_name+"_"+str(date.hour)+".txt"
##algo.var.account_max_lots = 1000 ## manual term only

algo.var.m1_quote = {"buy":True, "sell":True}
algo.var.m2_quote = {"buy":True, "sell":True}
algo.var.mov_avg_speed_filter = 4  #speed of moving average*length/size of band (filter)
algo.var.abs_funding_filter = 0.1/100 # 2 dollars a day @ 1BTC (nearly the same cost a legging into a spread)
algo.var.abs_funding_any = 0.51/100 # funding at which point you take trades without any care for band price
#algo.var.funding_8h_mov_avg = 1
algo.var.funding_mov_avg = 60

algo.var.mov_avg = 150
algo.var.margin = 0.3/100 
algo.var.size = 500
algo.var.max_inv = 10

####
algo.var.fixed_target = True
algo.var.fixed_target_size = 1
####
#algo.var.boll = 2
algo.var.taker_fee = np.mean([0.05,0.075])/100
algo.var.maker_fee = np.mean([-0.025,-0.02])/100
algo.var.fee = round(algo.var.taker_fee + algo.var.maker_fee,8)
algo.var.slippage = 0.5
####
algo.var.max_buy_price = 25
algo.var.min_sell_price = -25
algo.var.abs_band_any_with_max_funding = np.inf ## band price which you will buy or sell regardless on abs_funding_filter 
################################################## but still within the bounds of abs_funding_any
algo.var.max_m1_bid_ask_spread = 50
algo.var.max_m2_bid_ask_spread = 100

algo.data = data_variables(algo.var)

algo.main_algo()

