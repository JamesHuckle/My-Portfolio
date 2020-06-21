# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\..\JAMESH~1\AppData\Local\Temp'))
	print(os.getcwd())
except:
	pass

# %%
#%run "Deribit_Algo_testing_CLASS.ipynb" ##comment out when running from terminal with notebook merge


# %%
date = datetime.datetime.utcnow()
fut_sym, back_sym = contract_sym(date)

account_name = "fixed_levels"
algo = deri_algo(account_name,testing=False)
algo.var.account_name = account_name

## Important vars ##
algo.var.m1_sym = "BTC-PERPETUAL"
algo.var.m2_sym = "BTC-29MAR19"
algo.var.tick = 0.25
algo.var.pay_up_ticks = 200
algo.var.log = "log_Deribit_Algo_"+algo.var.account_name+"_"+str(date.hour)+".txt"
algo.var.account_max_lots = 1000000 ## manual term only

algo.var.manual = True
algo.var.settings_file = True

algo.data = data_variables(algo.var)

algo.main_algo()  

