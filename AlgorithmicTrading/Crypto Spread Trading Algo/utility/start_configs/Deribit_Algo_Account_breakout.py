# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
#%run "Deribit_Algo_testing.ipynb" ##comment out when running from terminal with notebook merge

# %%
date = datetime.datetime.utcnow()
fut_sym, back_sym = contract_sym(date,roll_buffer=0)

account_name = "breakout" #"main" "breakout"
algo = deri_algo(account_name,testing=True)
algo.var.account_name = account_name

## Important vars ##
algo.var.m1_sym = "BTC-PERPETUAL"
algo.var.tick = 0.25
algo.var.pay_up_ticks = 200
algo.var.log = "log_Deribit_Algo_"+algo.var.account_name+"_"+str(date.hour)+".txt"

algo.var.m1_quote = {"buy":True, "sell":True}
algo.var.size = 1

####
algo.data = data_variables(algo.var)

algo.channel_breakout_strat(timeframe=15,lookback=80,target_size_percent=16,avg_band_size=3.7) 

