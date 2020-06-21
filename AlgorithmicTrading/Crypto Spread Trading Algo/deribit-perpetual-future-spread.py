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
import pandas as pd
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 100
import numpy as np
import re
import datetime
import time
import matplotlib.pyplot as plt
import asyncio
import pickle
import math
from copy import deepcopy
import threading
from collections import OrderedDict

#import rapidjson as json
import json
import hmac, hashlib, requests, base64
from requests.auth import AuthBase
from multiprocessing import Pool
import traceback
import sys
import csv
import os ,os.path
import tailer as tl
import io
from IPython.display import clear_output
from win10toast import ToastNotifier
toaster = ToastNotifier()

from dateutil.relativedelta import relativedelta, FR
from lomond import WebSocket
from lomond.persist import persist
from deribit_api import RestClient
from dateutil.relativedelta import relativedelta, FR

# %% [markdown]
# ## Logging wrapper

# %%
### Function to print: Either to StdOut of Logfile
### StdOut is significantly slower 0.001 vs 0.00001
class print_to_log():
    def __init__ (self,log_file_name,stdout=False):
        self.log_file_name = log_file_name
        self.log_file_append = open(log_file_name,"a+")
        self.stdout = stdout
                                
    def p(self,*args,stdout=False):
        if self.stdout ==True:
            print(datetime.datetime.now(),*args)
            print()
        try:
            print(datetime.datetime.now(),*args,file=self.log_file_append)
            self.log_file_append.flush()
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
        
    def t(self,*args):
        try:
            traceback.print_exc(file=sys.stdout)
            traceback.print_exc(file=self.log_file_append)
            self.log_file_append.flush()
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
                
    def clear(self):
        log_clear = open(self.log_file_name,"w+")
        log_clear.flush()
                
    def set_logging(self):
        return self.p
    
    def set_traceback_logging(self):
        return self.t

# %% [markdown]
# ## Class based algorithm

# %%
#########################################################
######## ALGO #########
class deri_algo():        
    def __init__(self,account_name,testing=False):
        self.testing = testing
        self.var = fixed_variables()
        self.data = data_variables(var)
        self.stat = status_variables()
        self.arb_algo_path = os.getcwd().replace("\\python_scripts","")
        now = datetime.datetime.utcnow().date()# - datetime.timedelta(days=1)
        ### logging statement changes
        self.old_arb_status = () 
        self.old_funds_status = ()
        self.old_dont_quote_status = ()
        #### functions ####
        self.logging = print_to_log(log_file_name=self.var.log,stdout=self.testing).set_logging()
        self.clear_log = print_to_log(log_file_name=self.var.log).clear
        #### account login #####    
        if account_name == "main":
            url = "https://www.deribit.com"
            key = "#####"
            secret = "#####"
            
        elif account_name == "trading_hucks":
            url = "https://www.deribit.com"
            key = "#####"
            secret = "#####"
            
        elif account_name == "fixed_levels":
            url = "https://www.deribit.com"
            key = "#####"
            secret = "#####"
            
        elif account_name == "demo":
            url = "https://test.deribit.com"
            key = "#####"
            secret = "#####" 
        else:
            raise Exception("account name doens't match list of known accounts")
            
        self.client = RestClient(key, secret, url)
        self.account_name = account_name
    ################################################
    
    def generate_signature(self, key, secret, action, data):
        tstamp = int(time.time()* 1000)
        signature_data = {
            '_': tstamp,
            '_ackey': key,
            '_acsec': secret,
            '_action': action
        }
        signature_data.update(data)
        sorted_signature_data = OrderedDict(sorted(signature_data.items(), key=lambda t: t[0]))

        def converter(data):
            key = data[0]
            value = data[1]
            if isinstance(value, list):
                return '='.join([str(key), ''.join(value)])
            else:
                return '='.join([str(key), str(value)])

        items = map(converter, sorted_signature_data.items())
        signature_string = '&'.join(items)
        sha256 = hashlib.sha256()
        sha256.update(signature_string.encode("utf-8"))
        sig = key + "." + str(tstamp) + "." 
        sig += base64.b64encode(sha256.digest()).decode("utf-8")
        return sig


    defself.contract_sym(self, date, roll_buffer=0): ## roll_buffer is how many hours you want to switch to the seconds contract before it really expires
        expiry = [datetime.datetime(date.year,3,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1)),
                  datetime.datetime(date.year,6,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1)),
                  datetime.datetime(date.year,9,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1)),
                  datetime.datetime(date.year,12,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1)),
                  datetime.datetime(date.year+1,3,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1)),
                  datetime.datetime(date.year+1,6,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1))]    

        contracts = [[expiry[0]-datetime.timedelta(hours=roll_buffer),"BTC-"+expiry[0].date().strftime("%d%b").upper()],
                     [expiry[1]-datetime.timedelta(hours=roll_buffer),"BTC-"+expiry[1].date().strftime("%d%b").upper()],
                     [expiry[2]-datetime.timedelta(hours=roll_buffer),"BTC-"+expiry[2].date().strftime("%d%b").upper()],
                     [expiry[3]-datetime.timedelta(hours=roll_buffer),"BTC-"+expiry[3].date().strftime("%d%b").upper()],
                     [expiry[4]-datetime.timedelta(hours=roll_buffer),"BTC-"+expiry[4].date().strftime("%d%b").upper()],
                     [expiry[5]-datetime.timedelta(hours=roll_buffer),"BTC-"+expiry[5].date().strftime("%d%b").upper()]]

        for x in range(len(contracts)):
            if date < contracts[x][0]:
                front_sym = contracts[x][1] + str(contracts[x][0].year)[2:4]
                second_sym = contracts[x+1][1] + str(contracts[x+1][0].year)[2:4]
                break
            else:
                continue

        return front_sym,second_sym
    

    def ws(self,account_name):  
        client, key, secret, url = select_account(account_name)
        
        global time_now
        time_now = datetime.datetime.now()
        
        global all_orders
        all_orders = {}
        global account_and_positions
        account_and_positions = {}
        
        nicknames = {}
        file_names = {}
        headers = {}
        opened_files = {}
        old_ob = {}
        deri_ob = {}
        high = {}
        low = {}
        
        global send_ob
        send_ob = {}
        global trades 
        trades = {}
        global min_data
        min_data = {}            
        
        save_min = False
        ws_fund = False
        heartbeat1h = False
        old_date = datetime.datetime(2008,1,1)
        ### Start websocket ###
        while True:
            try:
                time_now = datetime.datetime.utcnow()
                fut_sym, back_sym = self.contract_sym(time_now)    
                instruments = ["BTC-PERPETUAL",fut_sym,back_sym]
                channels = ["quote","trade"]  
                                
                for inst in instruments:
                    ### create data stores ###
                    #deri_ob[inst] = [] dont create as it creates itself
                    trades[inst] = []
                    old_ob[inst] = {} 
                    high[inst] = -np.inf
                    low[inst] = np.inf
                spread_high = -np.inf
                spread_low = np.inf
                
                ws_delay = False
                if account_name == "demo":
                    websocket = WebSocket('wss://test.deribit.com/ws/api/v1/')
                else:
                    websocket = WebSocket('wss://www.deribit.com/ws/api/v1')
                    
                now_time = datetime.datetime.utcnow()
                for event in persist(websocket):
                    
                    time_now = datetime.datetime.utcnow()            
                    #################################################################################################### 
                    
                    if event.name == "ready":
                        ############ need to code up way to auto subscribe when instruments change ##############
                        #fut_sym, back_sym =self.contract_sym(time_now)
                        ######### connect to ws ############
                        ws_args = {"instrument":instruments,"event":["order_book","trade","portfolio","user_order"]}
                        signature = self.generate_signature(key,secret, "/api/v1/private/subscribe",ws_args)
                        print("connecting",datetime.datetime.now(),signature)
                        websocket.send_json(id = 5533,action="/api/v1/private/subscribe",arguments= ws_args,sig = signature)    
                        ######### funding data ##############
                        ws_args = {"instrument":"BTC-PERPETUAL"}
                        signature = self.generate_signature(key,secret, "/api/v1/public/getsummary",ws_args)  
                        ws_funding = websocket.send_json(id = 62,action="/api/v1/public/getsummary",arguments= ws_args,sig = signature)  
                        ########### account ##################                
                        ### rest/ws poll for orders ####
                        ws_args = {"action":"/api/v1/private/getopenorders"}
                        signature = self.generate_signature(key,secret, ws_args["action"],ws_args)  
                        ws_funding = websocket.send_json(id = 1,action=ws_args["action"],arguments= ws_args,sig = signature)        
                        
                        ws_args = {"action":"/api/v1/private/orderhistory"}
                        signature = self.generate_signature(key,secret, ws_args["action"],ws_args)  
                        ws_funding = websocket.send_json(id = 2,action=ws_args["action"],arguments= ws_args,sig = signature) 
                        #################################
                        
                    elif event.name == "text":
                        now_time = datetime.datetime.utcnow()
                        result = event.json
                        #print("actual new websocket object coming through")
        ####    ###########################################################
                        ######### funding if requested ####
                        if "id" in result:
                            if result["id"] == 62:
                                current_funding = result["result"]["currentFunding"]
                                funding_8h = result["result"]["funding8h"]
                                
                            if "result" in result:
                                if len(result["result"])>0:
                                    orders = result["result"]
                                    if result["id"] == 1:
                                        for o in orders[::-1]:
                                            if "modified" not in o:
                                                o["modified"] = o["lastUpdate"]
                                            all_orders[o["orderId"]] = o
                                    elif result["id"] == 2:
                                        for o in orders[::-1]:
                                            all_orders[o["orderId"]] = o
                                            
                                    ## write ##
                                    all_orders
                        ######################################
                        if "notifications" in result:
                            if "message" in result["notifications"][0]:
                                if result["notifications"][0]["message"] == "user_orders_event":
                                    orders = result["notifications"][0]["result"]
                                    print_fills = []
                                    filled_order = ""
                                    for o in orders:
                                        if "modified" not in o:
                                            o["modified"] = o["lastUpdate"]
                                        if o["state"] == "filled":
                                            print_fills.append(["order has been filled",datetime.datetime.now(),o])  
                                            filled_order = o["orderId"]
                                        all_orders[o["orderId"]] = o
                                    ##### rearrage so you keep all open orders and up to 50 closed orders history ##
                                    open_orders = {o["orderId"]:o for o in all_orders.values() if o["state"] == "open"}
                                    recent_x_orders_list = sorted(all_orders.values(),key=lambda k:k["modified"],reverse=True)
                                    recent_x_ids = [o["orderId"] for o in recent_x_orders_list]
                                    recent_ids = set(list(open_orders.keys())+recent_x_ids)
                                    recent_orders = {ids:all_orders[ids] for ids in all_orders.keys() if ids in recent_ids}
                                    all_orders = recent_orders.copy()
                                    ## write ##
                                    all_orders
                                    
                                    if len(print_fills)>0:
                                        print(print_fills)
                                    if filled_order!= "":
                                        if filled_order not in recent_ids:
                                            print("filled order wasnt in the dammed list! Some error in logic")
                                            raise Exception("stop websocket because filled order didnt make it to final list")
                                    
                                if result["notifications"][0]["message"] == "portfolio_event":
                                    account_and_positions = result["notifications"][0]["result"]
                                    ### if websocket gives you a blank positions packge (which should never happen because it includes the index and mark price for each market with always exists), then ignore it.
                                    ## POSITIONS CAN BE VERY UNRELIABLE BECAUSE OF THESE EMPTY PACKETS WHICH ARE PLAIN WRONG BECAUSE USUALLY YOU DO HAVE A POSITION
                                    ## write ##
                                    account_and_positions
                                    
                                if result["notifications"][0]["message"] == "order_book_event":
                                    inst = result["notifications"][0]["result"]["instrument"]
                                    chan = "quote"
                                    
                                    deri_ob[inst] = []                            
                                    for lev in range(len(result["notifications"][0]["result"]["bids"])):
                                        ob = {}
                                        ob["instrument"] = inst
                                        ob["level"] = lev+1
                                        ob["bid_price"] = result["notifications"][0]["result"]["bids"][lev]["price"]
                                        ob["bid_size"] = result["notifications"][0]["result"]["bids"][lev]["quantity"]
                                        ob["ask_price"] = result["notifications"][0]["result"]["asks"][lev]["price"]
                                        ob["ask_size"] = result["notifications"][0]["result"]["asks"][lev]["quantity"]
                                        deri_ob[inst].append(ob)
                                        if lev == 0:
                                            if ob["ask_price"] > high[inst]:
                                                high[inst] = ob["ask_price"]
                                            if ob["bid_price"] < low[inst]:
                                                low[inst] = ob["bid_price"]
                                    
                                    #for m1,m2 in [("BTC-PERPETUAL",fut_sym),("BTC-PERPETUAL",back_sym),(fut_sym,back_sym)]
                                    #perp = deri_ob.get("BTC-PERPETUAL",[{"bid_price":np.nan}])[0]["bid_price"]
                                    #fut = deri_ob.get(fut_sym,[{"bid_price":np.nan}])[0]["bid_price"]
                                    #spread = perp-fut
                                    #if spread > spread_high:
                                    #    spread_high = spread
                                    #if spread < spread_low:
                                    #    spread_low = spread
                                                
                                    if inst in instruments:
                                        if deri_ob[inst] != old_ob[inst]:
                                            old_ob[inst] = deri_ob[inst].copy()
                                            send_ob[inst] = deri_ob[inst].copy()
                                            for level in send_ob[inst]:
                                                level["timestamp"] = str(pd.to_datetime(result["notifications"][0]["result"]["tstamp"],unit="ms"))
                                                level["my_utc_time"] = str(now_time)
                                                if pd.to_datetime(level["my_utc_time"]) > pd.to_datetime(level["timestamp"])+datetime.timedelta(seconds=60):
                                                    print("websocket is delayed, restarting","deribit time",pd.to_datetime(level["timestamp"]),"my_time",level["my_utc_time"])
                                                    ws_delay = True
                                                    break
                                            # breaks from websocket loop to restart
                                            if ws_delay == True:
                                                break
                                                
                                            ## write to something
                                            send_ob[inst]
                                        
                    
                                    if now_time.second >= 30:
                                        save_min = True
                                    if now_time.second < 30:
                                        ws_fund = True
                                    
                                    if now_time.second > 45 and ws_fund == True:
                                        ### quickly send a ws request and loop to collect it, should be very quick and you are back here##
                                        if ws_fund == True:
                                            ws_args = {"instrument":"BTC-PERPETUAL"}
                                            signature = generate_signature(key,secret, "/api/v1/public/getsummary",ws_args)  
                                            ws_funding = websocket.send_json(id = 62,action="/api/v1/public/getsummary",arguments= ws_args,sig = signature) 
                                            ws_fund = False
            
                                    if now_time.second < 30 and save_min == True:
                                        min_data = []
                                        for inst in instruments:
                                            send_ob_min = deri_ob[inst][0].copy()
                                            send_ob_min["high"] = high[inst]
                                            send_ob_min["low"] = low[inst]
                                            high[inst] = -np.inf
                                            low[inst] = np.inf
                                            send_ob_min["timestamp"] = str(pd.to_datetime(result["notifications"][0]["result"]["tstamp"],unit="ms"))
                                            min_data.append(send_ob_min)
                                         
                                        ## add in 8h funding ##
                                        dummy = min_data[0]
                                        dummy = {key:np.nan for key,value in dummy.items()}
                                        dummy["instrument"] = "8h_funding"  
                                        dummy["bid_price"] = funding_8h
                                        dummy["ask_price"] = current_funding
                                        dummy["timestamp"] = str(pd.to_datetime(result["notifications"][0]["result"]["tstamp"],unit="ms"))
                                        min_data.append(dummy)                          
                                        
                                        ## write ##
                                        min_data
                                        save_min = False                           
                                        
                                    if now_time.minute == 0 and now_time.second == 0:
                                        heartbeat1h = True
                                    elif now_time.minute == 0 and now_time.second > 0 and heartbeat1h == True:
                                        print("ws full alive",now_time)
                                        heartbeat1h = False
                                            
                                if result["notifications"][0]["message"] == "trade_event":
                                    inst = result["notifications"][0]["result"][0]["instrument"]                  
                                    for trade in result["notifications"][0]["result"]:
                                        trades_dict = {}
                                        trades_dict["instrument"] = trade["instrument"]
                                        trades_dict["qty"] = trade["quantity"]
                                        trades_dict["timestamp"] = str(pd.to_datetime(trade["timeStamp"],unit="ms"))
                                        trades_dict["my_utc_time"] = str(now_time)
                                        trades_dict["price"] = trade["price"]
                                        trades_dict["direction"] = trade["direction"] 
                                        
                                        trades[inst].append(trades_dict)
                                        trades[inst] = trades[inst][-50:]
                                    
                                    ## write ##
                                    trades[inst]
                                                            
        ####    ###########################################################               
            
                    elif event.name == "pong":
                        continue
                    elif event.name == "poll": # happens every 5 seconds
                        new_fut, new_back =self.contract_sym(now_time)
                        if (fut_sym,back_sym) != (new_fut,new_back):
                            ## break websocket so it can change contracts
                            print("breaking websocket to change contracts")
                            break
                        if now_time+datetime.timedelta(seconds=5) < datetime.datetime.utcnow():
                            print("websocket has been stale for 5 seconds","ws_time",now_time,"now",datetime.datetime.utcnow())
                            break
            except Exception as e:
                print("websocket had an error",str(e))
                traceback.print_exc(file=sys.stdout)
                
                
    def start_websocket_thread(self):
        self.t = threading.Thread(target=self.ws, args=(self.account_name,))
        self.t.start()
        while True:
            if self.var.m1_sym in send_ob and self.var.m2_sym in send_ob:
                break
            else:
                print("waiting to fully populate ws")
                time.sleep(1)
        
        
    def print_ws_data(self):
        print()
        print("all_orders")
        print(all_orders)
        print()
        print("account_and_positions")
        print(account_and_positions)
        print()
        print("send_ob")
        print(send_ob)
        print()
        print("trades")
        print(trades)
        print()
        print("min_data")
        print(min_data)
         
            
    def reset_data_stat(self):
        self.data = data_variables(self.var)
        self.stat = status_variables()            

        
    def set_logging_file(self):
        self.logging = print_to_log(log_file_name=self.var.log,stdout=self.testing).set_logging()
        self.clear_log = print_to_log(log_file_name=self.var.log).clear         
        
        
    def set_opened_files(self):
        now = datetime.datetime.utcnow().date()
        perp_file = self.arb_algo_path+"\data\deri_data\\"+str(now)+"_btc_perpetual_quote.csv"
        fut_file = self.arb_algo_path+"\data\deri_data\\"+str(now)+"_btc_front_quarter_quote.csv"
        back_file = self.arb_algo_path+"\data\deri_data\\"+str(now)+"_btc_back_quarter_quote.csv"
        exist = [os.path.isfile(perp_file),os.path.isfile(fut_file),os.path.isfile(back_file)]
        if all(exist):
            self.perp_open_file = open(self.arb_algo_path+"\data\deri_data\\"+str(now)+"_btc_perpetual_quote.csv",'r',newline='')
            self.fut_open_file = open(self.arb_algo_path+"\data\deri_data\\"+str(now)+"_btc_front_quarter_quote.csv",'r',newline='')
            self.back_open_file = open(self.arb_algo_path+"\data\deri_data\\"+str(now)+"_btc_back_quarter_quote.csv",'r',newline='')
        else:
            self.logging("ws data files do not exist (yet?)")
        
   
    def get_settings(self):
        old_reload_size = [self.var.sell_reload,self.var.buy_reload,self.var.size]
        submit = False
        ### read data from settings file
        file = self.arb_algo_path+"\python_scripts\settings_Deribit_"+self.account_name+".txt"
        with open(file,"r") as f:
            data = f.read().splitlines() 
            if self.testing==True:
                print(data)
        settings = {}
        for setting in data:
            try:
                k,v = setting.split("=")
                v = v.strip(" ")
                settings[k] = v
            except:
                None
        
        ### see if submit_reload has changed, or its the first pass and you want to initialise reloads (inv_count)
        sell_reload = settings["sell_reload"]
        sell_reload = round(int(sell_reload)) if sell_reload != '' else 0
        buy_reload = settings["buy_reload"]
        buy_reload = round(int(buy_reload)) if buy_reload != '' else 0
        submit_reload = settings["submit_reload"]
        ### auto submit ###
        if self.var.settings_file == False and self.var.submit_reload == '':
            self.var.submit_reload = "s"
            submit = True
        ### setting file submit ###
        elif submit_reload == "s" and self.var.submit_reload == '':
            submit = True
        if submit == True:
            self.logging("submit reload is true, get_settings()","settings file",self.var.settings_file)
            self.var.sell_reload = sell_reload
            self.var.buy_reload = buy_reload
        self.var.submit_reload = submit_reload

        
        ### pull in buy and sell prices, and work out what inst and sides you need to be quoting
        sell_price = settings["sell_price"]
        buy_price = settings["buy_price"]
        quote_m1 = settings["quote_m1"] in ["True","true","t"]
        quote_m2 = settings["quote_m2"] in ["True","true","t"]
        
        self.var.quote_m1 = {"buy":quote_m1, "sell":quote_m1}
        self.var.quote_m2 = {"buy":quote_m2, "sell":quote_m2}
        
        if sell_price == '' or self.var.sell_reload == 0:
            self.var.quote_m1["sell"] = False
            self.var.quote_m2["buy"] = False
        else:
            self.var.sell_price = round(float(sell_price),2)
            
        if buy_price == '' or self.var.buy_reload == 0:
            self.var.quote_m1["buy"] = False
            self.var.quote_m2["sell"] = False
        else:
            self.var.buy_price = round(float(buy_price),2) 
        
        ## set algo size and max_inv
        size = int(settings["size"])
        if size != self.var.size:
            for m in ["m1","m2"]:
                inst = self.data.d[m]
                for side in ["buy","sell"]:
                    size_change = size - self.var.size
                    inst["working_orders_info"][side]["quote_qty_left"] += size_change
                    inst["working_orders_info"][side]["original_size"] += size_change
                    self.stat.order_action["amend_quote"].append((m,side)) 
                    self.logging("adjusting manual size","old size",self.var.size,"new size",size,"get_settings()")
                    self.amend_quotes()
            self.var.size = size
        self.var.max_inv = self.var.account_max_lots / self.var.size
        self.var.change_contracts_max_inv = self.var.max_inv
        
        ### end
        self.var.settings = settings
        ### if there are a change in the reloads then send it to the log file
        if old_reload_size != [self.var.sell_reload,self.var.buy_reload,self.var.size]:
            self.logging("buy_reload",self.var.buy_reload,"sell_reload",self.var.sell_reload,"size",self.var.size)

    def read_dict(self,file_name):
        error = 0
        while True:
            try:
                with (open(file_name,"r")) as file:
                    result = json.load(file)
                break
            except Exception as e:
                error +=1
                self.logging("read() error:"+file_name)
                if error > 20:
                    raise Exception("read() error whilst trying to read file:"+file_name)
                time.sleep(0.0005)
                continue  
        return result
                  
    def unpickle(self,file_name):
        error = 0
        while True:
            try:
                result = pickle.load(open(file_name,"rb"))
                break
            except (EOFError,TypeError):
                error +=1
                self.logging("unpickle() eo error:"+file_name)
                if error > 10:
                    raise Exception("unpickle() error whilst trying to unpickle file:"+file_name)
                time.sleep(0.005)
                continue
            except pickle.UnpicklingError:
                error +=1
                self.logging("unpickle() pickle error:"+file_name)
                if error > 10:
                    raise Exception("unpickle() error whilst trying to unpickle file:"+file_name)
                time.sleep(0.005)
                continue   
            
        return result
            
    def deri_order(self,info):
        complete = False
        service_error = False
        order = []
        error = 0
        
        if info["side"] == "buy":
            func = self.client.buy
        elif info["side"] == "sell":
            func = self.client.sell
        
        if info["order"] == "limit":
            post = "true"
        elif info["order"] == "market":
            post = "false"
    
        while True:
            try:
                self.logging("deri_order() instrument:",info["inst"],"side",info["side"],"qty:",info["qty"],"price:",info["price"],"post_only:",post)
                order = func(instrument=info["inst"], quantity=info["qty"], price=info["price"], postOnly=post)
                self.stat.poll +=1
                complete = True
                break
            except Exception as e:
                if ("Max retries exceeded" in str(e)) or ("OSError" in str(e)) or ("Wrong response code: 405" in str(e)) or ("A connection attempt failed" in str(e)):
                    error +=1
                    self.logging("deri_order() problem with rest polling, retry",str(e))
                    if error >10:
                        raise Exception("Service Unavailable?(deri_order) for 4 seconds")
                    continue
                elif str(e) == "Failed: invalid_quantity":
                    raise Exception("qty is invalid (deri_order)",str(e))
                elif str(e) == "Failed: other_reject overlapping_order_found":
                    self.logging("deri_order(),attempted to cross on ourselves, cancel resting order and retry")
                    cancel_inst = "m1" if info["market"] == "m2" else "m2"
                    cancel_side = info["side"]
                    ids = self.data.d[cancel_inst]["working_orders"][cancel_side]["orderId"]
                    order, hedge = self.deri_cancel(ids)
                    if hedge == True:
                        raise Exception("deri_order(),order trying to be cancelled has already been filled!",cancel_inst,cancel_side,ids)
                    else:
                        self.logging("seems to be a successful order cancellation",order,"trying to send order again...")
                    continue                 
                else:
                    self.logging("deri_order() unknown error, stop algo:",str(e))
                    raise e
                    
        if complete == True:
            order = order["order"]
            
        return order, service_error
    
    
    def deri_cancel(self,ids):
        order = []
        error = 0
        error1 = 0
        hedge = False
        while True:
            try:
                order = self.client.cancel(ids)
                self.stat.poll +=1
            except Exception as e:
                if ("Max retries exceeded" in str(e)) or ("OSError" in str(e)) or ("Wrong response code: 405" in str(e)) or ("A connection attempt failed" in str(e)):
                    error +=1
                    self.logging("deri_cancel() problem with rest polling, retry",str(e))
                    if error >10:
                        raise Exception("Service Unavailable?(deri_cancel) for 4 seconds")
                    continue
                elif str(e) == "Failed: order_not_found":
                    error1 +=1
                    self.logging("order doesn't exist,",e)
                    if error1 >5:
                        raise Exception("OrderId does not exist? (deri_cancel)",ids,str(e))   
                    continue
                else:
                    self.logging("unknown error (deri_cancel), stop algo",ids,str(e))
                    raise e
            
            self.logging("cancelling deri order",ids,order)
            order = order["order"]
            if len(order) >0:
                if order["state"] == "cancelled": 
                    break
                else:
                    if order["state"] == "filled":
                        self.logging("couldnt cancel order because it has already been filled! (deri_cancel)",ids)
                        hedge = True
                        break
                    elif order["state"] == "cancelled":
                        self.logging("couldnt cancel order because it has already been cancelled! (deri_cancel)",ids) 
                        break
                    else:
                        self.logging("cancel failed:",ids,order)
                        raise Exception("Order could not be cancelled!?,maybe wrong ID was passed? (deri_cancel)",ids)
                    
        return order, hedge

    ## not very useful as it doesn't return and order list
    def deri_cancel_all(self):
        cancel_error = 0
        while True:
            try:
                self.client.cancelall()  
                self.stat.poll +=1
                break
            except Exception as e:
                cancel_error +=1
                if cancel_error > 30:
                    time.sleep(5)
                    self.logging("error tying to cancel orders",e)
                elif cancel_error > 1440: # 2 hours
                    raise Exception("Couldnt cancel orders for 2 hours")                   
                self.logging("error tying to cancel orders",e)
                time.sleep(0.1)
                        
    def deri_amend_order(self,amend):
        order = []
        service_error = False
        order_error = False
        error = 0
        while True:
            try:
                self.logging("attempting deri_amend_order() ","id:",amend["order_id"],"amend price:",amend["price"],"amend qty:",amend["qty"])        
                order = self.client.edit(orderId=amend["order_id"], price=amend["price"],quantity=amend["qty"])
                self.stat.poll +=1
                break
            except Exception as e:
                if ("Max retries exceeded" in str(e)) or ("OSError" in str(e)) or ("Wrong response code: 405" in str(e)) or ("A connection attempt failed" in str(e)):
                    error +=1
                    self.logging("deri_amend_order() problem with rest polling, retry",str(e))
                    if error >100:
                        raise Exception("Service Unavailable?(deri_amend_order()) for 4 seconds")
                    continue
                elif str(e) == "Failed: invalid_quantity":
                    raise Exception("qty is invalid (deri_amend_order),",amend["order_id"],str(e))
                                    
                elif str(e) in ["Failed: other_error already_closed","Failed: not_open_order","Failed: order_not_found"]:
                    self.logging("deri_amend_order() order may have been filled, or cancelled, check fills",amend["order_id"],str(e))
                    order_error = True
                    break
                else:
                    self.logging("deri_amend_order unknown error",str(e))
                    raise Exception("unknown error (deri_amend_order), stop algo",amend["order_id"])
     
        if order_error == True or service_error == True:
            self.logging(" order or service error, deri_amend_order()","order:",order, "service_error:",service_error,"order_error:",order_error)
            return order, service_error, order_error
        else:
            self.logging("success deri_amend_order()","order:",order["order"], "service_error:",service_error,"order_error:",order_error)
            return order["order"], service_error, order_error                        
                        

    def deri_positions(self):
        pos = []
        error = 0
        while True:
            try:
                pos = self.client.positions()
                break
            except Exception as e:
                if ("Max retries exceeded" in str(e)) or ("OSError" in str(e)) or ("Wrong response code: 405" in str(e)) or ("A connection attempt failed" in str(e)):
                    error +=1
                    self.logging("deri_positions() problem with rest polling, retry",str(e))
                    if error >10:
                        raise Exception("Service Unavailable?(deri_positions) for 4 seconds")
                    continue
                else:
                    raise e
        return pos
    
    def deri_account_and_positions(self):
        ## account_and_positions
        #{'portfolio': [{'currency': 'BTC', 'equity': 102.641007803, 'maintenanceMargin': 0.0, 'initialMargin': 0.0, 'availableFunds': 102.641007803, 'unrealizedPl': 0.0, 'realizedPl': 2.7684e-05, 'totalPl': 0.0}],
        # 'positions': [{'instrument': 'BTC-PERPETUAL', 'kind': 'future', 'size': 0, 'amount': 0.0, 'averagePrice': 0.0, 'direction': 'zero', 'sizeBtc': 0.0, 'floatingPl': 0.0, 'realizedPl': 2.7684e-05,'estLiqPrice': 0.0,
        #                'markPrice': 3971.04, 'indexPrice': 3971.87, 'maintenanceMargin': 0.0, 'initialMargin': 0.0, 'settlementPrice': 3962.32, 'delta': 0.0, 'openOrderMargin': 0.0, 'profitLoss': 0.0},
        #               {'instrument': 'BTC-28DEC18', 'kind': 'future', 'size': 0, 'amount': 0.0, 'averagePrice': 0.0, 'direction': 'zero', 'sizeBtc': 0.0, 'floatingPl': 0.0, 'realizedPl': 0.0, 'estLiqPrice': 0.0,
        #                'markPrice': 3920.68, 'indexPrice': 3971.87, 'maintenanceMargin': 0.0, 'initialMargin': 0.0, 'settlementPrice': 3909.35, 'delta': 0.0, 'openOrderMargin': 0.0, 'profitLoss': 0.0}]}
        ## from json ##
        #account_and_positions = self.read_dict(self.arb_algo_path+"\python_scripts\deribit_account_and_positions_"+self.account_name+".json")
        ## from ws ##
        account_and_positions
        
        account = account_and_positions["portfolio"][0]
        raw_positions = account_and_positions["positions"]
        positions = {}
        for pos in raw_positions:
            inst = pos["instrument"]
            positions[inst] = pos
         
        ## dont rely on the positions data, its usually wrong fromt the websocket (can send a [] balnk message even when we have positions
        #if len(positions) == 0:
        #    print("deri_account_and_positions()","no positions from websocket (seems wrong)","raw_positions",raw_positions)
        #if len(account) == 0:
        #    print("deri_account_and_positions()","account",account)
        
        self.data.positions = positions   
        self.data.account = account    
        return account, positions
    
    
    ### active orders really are only active, once one is filled is dissapears immediatly and therefore does not update fill quantity, open status...etc from here, the same goes for if an order is cancelled,
    # you will not know from polling this.
    # Even polling with the individual trade ID does not show any cancelled orders
    def deri_active_orders(self,order_id="",inst="",details=False):
        con_error = 0
        orders = []
        cols = ["orderId","instrument","direction","quantity","price","type","state","filledQuantity","avgPrice","modified"]
        while True:
            try:
                self.stat.poll +=1
                if inst == "" and order_id == "":
                    orders = self.client.getopenorders()
                    for o in orders:
                        o["modified"] = pd.to_datetime(o["lastUpdate"]*10**6)
                        if details == False:
                            o = {key:value for key,value in o.items() if key in [cols]}
                    break
                elif inst != "":
                    orders = self.client.getopenorders(instrument=inst)
                    for o in orders:
                        o["modified"] = pd.to_datetime(o["lastUpdate"]*10**6)
                        if details == False:
                            o = {key:value for key,value in o.items() if key in [cols]}
                    break
                elif order_id != "":
                    orders = self.client.getopenorders(orderId=order_id)
                    break    
                    
            except ConnectionError as e:
                self.logging("deri_active_orders(), connection error",str(e))
                if con_error == 5:
                    raise e
                con_error +=1
                
            except Exception as e:
                if ("Max retries exceeded" in str(e)) or ("OSError" in str(e)) or ("Wrong response code: 405" in str(e)) or ("A connection attempt failed" in str(e)):
                    error +=1
                    self.logging("deri_active orders() problem with rest polling, retry",str(e))
                    if error >10:
                        raise Exception("Service Unavailable?(deri_active orders()) for 4 seconds")
                    continue
                else:
                    raise e
        return orders
  

    ## best for polling to check if orders have been filled, partial, or cancelled. A filled order will get sent here with a "state" of "filled". A partially filled order with still have a state ##
    # of "open" and a "qunatity" of 200 and "filledQuantity" of 100 for example. A cancelled order will show up cancelled, even if it has been paritally filled before ###
    # Also you can tell if an order has not been submitted via the API, which could come in handy at some point. Only issue is that you can only poll one order at a time. ###
    # No other function shows you cancelled orders however, you could only deduce that if you order isnt open, and hasnt been filled, that it has been cancelled, maybe that would mean less polls in the long run? ###        
    def deri_order_status(self,order_id):
        order = []
        error = 0
        error1 = 0
        cancelled = False
        while True:
            try:
                order = self.client.orderstate(order_id)
                self.stat.poll +=1
                ##order["created"] = pd.to_datetime(order["created"],unit="ms")
                ##order["modified"] = pd.to_datetime(order["created"],unit="ms")
                break
    
            except Exception as e:
                if ("Max retries exceeded" in str(e)) or ("OSError" in str(e)) or ("Wrong response code: 405" in str(e)) or ("A connection attempt failed" in str(e)):
                    error +=1
                    self.logging("deri_order_status() problem with rest polling, retry",str(e))
                    if error >10:
                        raise Exception("Service Unavailable?(deri_order_status) for 4 seconds")
                    continue
                elif str(e) == "Failed: order_not_found":
                    self.logging("deri_order_status() order cannot be found so cannot have be filled either, submit new order!",str(e))
                    cancelled = True
                else:
                    self.logging("unidentified error from deri_order_status()",str(e),"relooping...")
                    error1 +=1
                    #time.sleep(0.1)
                    if error1 > 10:
                        raise Exception("unidentified error(deri_order_status) for 4 seconds"+str(e))
                    continue
                    
        return order, cancelled        
 

    def deri_hist_orders(self,num,order_id="",details=False,testing=False):
        con_error = 0
        error = 0
        orders = []
        cols = ["orderId","instrument","direction","quantity","price","type","state","filledQuantity","avgPrice","modified"]        
        while True:
            try:
                orders = self.client.orderhistory(num)
                self.stat.poll +=1
                for o in orders:
                    o["created"] = pd.to_datetime(o["created"],unit="ms")
                    o["modified"] = o["created"]
                    if details == False:
                        o = {key:value for key,value in o.items() if key in [cols]}
                        
                if order_id != "":
                    orders = [o for o in orders if o["orderId"] == order_id]
                       
                break
               
            except ConnectionError as e:
                self.logging("deri_hist_orders() connection error",str(e))
                if con_error == 5:
                    raise e
                con_error +=1
                                
            except Exception as e:
                if self.testing == True:
                    print(e)
                if ("Max retries exceeded" in str(e)) or ("OSError" in str(e)) or ("Wrong response code: 405" in str(e)) or ("A connection attempt failed" in str(e)):
                    error +=1
                    self.logging("deri_hist_orders() problem with rest polling, retry",str(e))
                    if error >10:
                        raise Exception("Service Unavailable?(deri_hist_orders) for 4 seconds")
                    continue
                else:
                    raise e
        return orders

    
    def deri_all_orders(self,num,conn="ws"):
        if conn == "rest":
            loop = asyncio.get_event_loop()
            
            futures = [loop.run_in_executor(None,self.deri_active_orders),                       loop.run_in_executor(None,self.deri_hist_orders,num)]
            
            result = loop.run_until_complete(asyncio.gather(*futures))   
            orders = [*result[0],*result[1]]
            
            indexed_orders = {}
            for o in orders:
                indexed_orders[o["orderId"]] = o
            open_orders = {ids:order for ids,order in all_orders.items() if order["state"] == "open"}   
            self.data.open_orders = open_orders                   
            return indexed_orders 
        
        elif conn == "ws":
            ## from json ####
            #all_orders = self.read_dict(self.arb_algo_path+"\python_scripts\deribit_all_orders_"+self.account_name+".json")
            ### convert the keys to integers (they are order ids), because .json turns all keys into strings (grrrr)
            #all_orders = {int(key):value for key,value in all_orders.items()}
            
            ### from ws ###
            all_orders
            
            ### need to augment with any information we already know about open orders, as rest response will be quicker than websocket
            ### and you don't want the websocket overwriting any information that you know is correct from working orders
            for m in ["m1","m2"]:
                for side in ["buy","sell"]:
                    working_order = self.data.d[m]["working_orders"][side]
                    if len(working_order)>0:
                        ids = working_order["orderId"]
                        if "lastUpdate" in working_order:
                            working_order["modified"] = working_order["lastUpdate"]
                        if ids not in all_orders:
                            all_orders[ids] = working_order
                        elif "modified" in working_order:
                            if working_order["modified"] > all_orders[ids]["modified"]:
                                all_orders[ids] = working_order
            
            open_orders = {ids:order for ids,order in all_orders.items() if order["state"] == "open"}
            self.data.open_orders = open_orders
            return all_orders      
        
        
    def deri_ob(self,inst,conn,df=True):
        if conn == "rest":
            def rest_ob(inst,df):
                error = 0
                while True:
                    try:
                        price_data = self.client.getorderbook(inst)
                        self.stat.poll +=1
                        ob = []
                        for x in range(min(len(price_data["bids"]),len(price_data["asks"]))):
                            level = {}
                            level["level"] = x
                            level["instrument"] = price_data["instrument"]
                            level["timestamp"] = pd.to_datetime(price_data["tstamp"],unit="ms")
                            level["bid_price"] = price_data["bids"][x]["price"]
                            level["bid_size"] = price_data["bids"][x]["quantity"]
                            level["ask_price"] = price_data["asks"][x]["price"]
                            level["ask_size"] = price_data["asks"][x]["quantity"]
                            level["last"] = price_data["last"]
                            level["24h_low"] = price_data["low"]
                            level["24h_high"] = price_data["high"]
                            ob.append(level)
                            
                        if df == True:
                            return pd.DataFrame(ob)
                        else:
                            return ob
                    
                    except Exception as e:
                        self.logging("data error deri_ob()",e,"relooping")
                        error +=1
                        if error == 5:
                            raise Exception ("data failed 5 times")
            if inst != "all":
                data = rest_ob(inst,df)
                return data
            else:
                ##fut_sym, back_sym =self.contract_sym(date,roll_buffer=3) 
                loop = asyncio.get_event_loop()
                futures = [loop.run_in_executor(None,rest_ob,self.perp_open_file,df),
                           loop.run_in_executor(None,rest_ob,self.fut_open_file,df),
                           loop.run_in_executor(None,rest_ob,self.back_open_file,df)]  
                
                result = loop.run_until_complete(asyncio.gather(*futures)) 
                data = {"BTC-PERPETUAL":result[0],fut_sym:result[1],back_sym:result[2]}
                return data
                
                        
        elif conn == "ws":
            #################################################
            ## from json ##
            #### asycn function, look below it for more detail!
            #def ws_ob(opened_file,columns=['instrument', 'level', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'timestamp', 'my_utc_time']):
            #    lastLines = tl.tail(opened_file,20)
            #    ws_file_data = pd.read_csv(io.StringIO('\n'.join(lastLines)), header=None)
            #    ws_file_data.columns = columns
            #    return ws_file_data

            #loop = asyncio.get_event_loop()
            #futures = [loop.run_in_executor(None,ws_ob,self.perp_open_file),
            #           loop.run_in_executor(None,ws_ob,self.fut_open_file),
            #           loop.run_in_executor(None,ws_ob,self.back_open_file)]  
            #
            #result = loop.run_until_complete(asyncio.gather(*futures))             
            #
            #data = {"BTC-PERPETUAL":result[0],fut_sym:result[1],back_sym:result[2]}
            
            ## from ws ##
            data = send_ob.copy()
            if inst != "all":
                data = data[inst]
            return data
          
            
    def deri_ob_merge(self,ob_df_list):
        suffix = ["x","y","z","a","b"]
        ### add matching suffix to each column header
        for idx in range(len(ob_df_list)):
            cols = list(ob_df_list[idx].columns)
            for x in range(len(cols)):
                if cols[x] != "level":
                    cols[x] = cols[x]+"_"+suffix[idx]
            ob_df_list[idx].columns = cols
        
        ### merge the dataframes
        combined = ob_df_list[0]
        for ob in ob_df_list[1:]:
            combined = combined.merge(ob,on="level",how="outer")        
        return combined
    
    
    def deri_weighted_price(self,ob,size):
        prices = {}
        sizes = {}
        prices["bid"] = [x["bid_price"] for x in ob]
        sizes["bid"] = [x["bid_size"] for x in ob]
        prices["ask"] = [x["ask_price"] for x in ob]
        sizes["ask"] = [x["ask_size"] for x in ob]
        weighted_avg = {"bid":0,"ask":10000000}
        for side in ["bid","ask"]:
            if sizes[side][0] >= size:
                weighted_avg[side] = prices[side][0]
            elif sum(sizes[side]) >= size:
                found_size = 0
                for level in range(len(prices[side])):
                    found_size += sizes[side][level]
                    if found_size >= size:
                        depth = level
                        break
                ### find average of depth-1 , then add amaount required from last level
                avg_price = 0
                for x in range(depth):
                    avg_price += sizes[side][x] * prices[side][x]
                tot_sizes = sum(sizes[side][:depth])
                size_on_last_level = size-tot_sizes
                avg_price += size_on_last_level*prices[side][depth]
                ### calc avg ###
                weighted_avg[side] = round(avg_price/size*2,0)/2  
            else:
                self.logging("20 levels do not have enough size to trade a "+side)
                print("20 levels do not have enough size to trade a "+side)
        return weighted_avg
    
    
    def set_minute_data(self):
        orders={}
        loop = asyncio.get_event_loop()
        current_week = datetime.datetime.utcnow().isocalendar()[1]
        if current_week == 1:
            prev_week = 52
        else:
            prev_week = current_week - 1
        futures = [loop.run_in_executor(None,pd.read_csv,self.arb_algo_path+"\data\deri_1min\week_"+str(prev_week)+"_1min"+".csv"),
                   loop.run_in_executor(None,pd.read_csv,self.arb_algo_path+"\data\deri_1min\week_"+str(current_week)+"_1min"+".csv")] 
        result = loop.run_until_complete(asyncio.gather(*futures))   
        a = result[0]
        b = result[1]
        minute_data = pd.concat([a,b])
        self.data.minute_data = minute_data
                
                
    def get_minute_data(self): 
        ## from file ##
        #columns=['instrument', 'level', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'timestamp', 'my_utc_time', 'high', 'low']
        #file = self.arb_algo_path+"\data\deri_data\week_"+str(datetime.datetime.utcnow().isocalendar()[1])+"_1min"+".csv"
        #lastLines = tl.tail(open(file,"r",newline=''),4)
        #ws_file_data = pd.read_csv(io.StringIO('\n'.join(lastLines)), header=None)
        #ws_file_data.columns = columns
        
        ## from ws ##
        ws_file_data = min_data.copy()
        ## only keep the last week 5040 mins to stop the file getting too big
        self.data.minute_data = self.data.minute_data.append(ws_file_data)[-5040*4:]     
                
    def resample_min_data(self,timeframe,market):
        data = pd.DataFrame(self.data.minute_data)
        data = data[data["instrument"]==market]
        data = data.resample(timeframe).ohlc()
        new_data = pd.DataFrame()
        new_data["timestamp"] = data.index
        for side in ["bid","ask"]:
            for point in ["open","high","low","close"]:
                major = side+"_"+point
                new_data[major] = data[major][point].values  
        data = new_data
        data = data.dropna(axis="rows")
        data = data.reset_index(drop=True)
        return data       
                 
    def assign_prices(self,conn="ws"):
        data = self.deri_ob(inst="all",conn=conn)
        m1 = self.data.d["m1"]
        m2 = self.data.d["m2"]
        m1["ob"] = data[self.var.m1_sym]
        m1["ob"] = data[self.var.m2_sym]
        
        ## only app.ies when getting data form json, otherwise comes as a list of dicts
        ## much quicker to assign things in a list of dics, then in DF
        m1_ob = data[self.var.m1_sym]#.to_dict("records")
        m2_ob = data[self.var.m2_sym]#.to_dict("records")
        
        m1_weighted = self.deri_weighted_price(m1_ob,self.var.size)
        m2_weighted = self.deri_weighted_price(m2_ob,self.var.size)
        
        m1["bid_price"] = m1_ob[0]["bid_price"]
        m1["ask_price"] = m1_ob[0]["ask_price"]
        m1["bid_size"] = m1_ob[0]["bid_size"]
        m1["ask_size"] = m1_ob[0]["ask_size"]
        m1["spread"] = m1_ob[0]["ask_price"] - m1_ob[0]["bid_price"]
        m1["bid_price_weighted"] = m1_weighted["bid"]
        m1["ask_price_weighted"] = m1_weighted["ask"] 
        
        m2["bid_price"] = m2_ob[0]["bid_price"]
        m2["ask_price"]= m2_ob[0]["ask_price"]
        m2["bid_size"] = m2_ob[0]["bid_size"]
        m2["ask_size"]= m2_ob[0]["ask_size"]
        m2["spread"] = m2_ob[0]["ask_price"] - m2_ob[0]["bid_price"]
        m2["bid_price_weighted"] = m2_weighted["bid"]
        m2["ask_price_weighted"]= m2_weighted["ask"]
        
        self.data.spread_price_weighted["sell"] = m1_weighted["ask"] - m2_weighted["ask"]
        self.data.spread_price_weighted["buy"] = m1_weighted["bid"] - m2_weighted["bid"]
        
        if self.var.manual == True:
            self.data.band["buy"] = algo.var.buy_price
            self.data.band["sell"] = algo.var.sell_price
            
        self.arb_price(self.data.d["m1"],self.data.d["m2"],self.data.band)
        return None
        
############################ DATA FUNCTIONS ######################################################
    def arb_price(self,m1,m2,band):       
        error=0 
        sell_spread_target_price_m1 = np.nan
        sell_spread_target_price_m2 = np.nan
        buy_spread_target_price_m1 = np.nan
        buy_spread_target_price_m2 = np.nan
        self.stat.quote_type["buy"] = "band"
        self.stat.quote_type["sell"] = "band"
                   
        # Calculate working order price 
        # Using bids on both markets, or offers on both markets
        working_band = {"m1":{"buy": m2["bid_price"] + band["buy"],
                              "sell": m2["ask_price"] + band["sell"]},
                        "m2":{"buy": m1["bid_price"] - band["sell"],
                              "sell": m1["ask_price"] - band["buy"]}}
        
        final_working = deepcopy(working_band)
                                    
        #### Calculate one tick (currently 0.01 on GDAX BTC-EUR/USD) from the current bid or ask, so can beat the best bid or ask
        ###better_ask = round(max((current_gdax_bid + 0.01),(current_gdax_ask - 0.01)),2)
        ###better_bid = round(min((current_gdax_ask - 0.01),(current_gdax_bid + 0.01)),2)
        
        if self.var.fixed_target == True:
            ######## Put a tighter profit target limit order in if a trade needs one, it will supercede any existing band limit order ###############
            ##stat = {"inv_qtys":[-50,-50,-50,-50,-50],"inv_spreads":[15,20,17,22,11],"inv_band_sizes":[12,10,12,11,11]}
            if self.stat.inv_count > 0:
                best_buy_entry = min(self.stat.inv_spreads)
                entry_idx = self.stat.inv_spreads.index(best_buy_entry)
                band_size = self.stat.inv_band_sizes[entry_idx]
                buy_spread_target_price_m1 = (m2["ask_price"] + best_buy_entry) + (band_size * var.fixed_target_size)
                buy_spread_target_price_m1 = round(buy_spread_target_price_m1 * 2) / 2            
                buy_spread_target_price_m2 = (m1["bid_price"] - best_buy_entry) - (band_size * var.fixed_target_size)
                buy_spread_target_price_m2 = round(buy_spread_target_price_m2 * 2) / 2
                ### is the new sell target price lower/closer than min working ask, if so, set it as min working ask
                if buy_spread_target_price_m1 < working_band["m1"]["sell"]:
                    final_working["m1"]["sell"] = buy_spread_target_price_m1
                    self.stat.quote_type["sell"] = "fixed_target"
                if buy_spread_target_price_m2 > working_band["m2"]["buy"]:
                    final_working["m2"]["buy"] = buy_spread_target_price_m2
                    self.stat.quote_type["sell"] = "fixed_target"
                    
            elif self.stat.inv_count < 0:
                best_sell_entry = max(self.stat.inv_spreads)
                entry_idx = self.stat.inv_spreads.index(best_sell_entry)
                band_size = self.stat.inv_band_sizes[entry_idx]
                sell_spread_target_price_m1 = (m2["bid_price"] + best_sell_entry) - (band_size * var.fixed_target_size)
                sell_spread_target_price_m1 = round(sell_spread_target_price_m1 * 2) / 2
                sell_spread_target_price_m2 = (m1["ask_price"] - best_sell_entry) + (band_size * var.fixed_target_size)
                sell_spread_target_price_m2 = round(sell_spread_target_price_m2 * 2) / 2
                ### is the new buy target price higher/closer than min working bid, if so, set it as min working bid
                if sell_spread_target_price_m1 > working_band["m1"]["buy"]:
                    final_working["m1"]["buy"] = sell_spread_target_price_m1
                    self.stat.quote_type["buy"] = "fixed_target"
                if sell_spread_target_price_m2 < working_band["m2"]["sell"]:
                    final_working["m2"]["sell"] = sell_spread_target_price_m2
                    self.stat.quote_type["buy"] = "fixed_target"
                    
        #####################################################################################################################################
        m1 = self.data.d["m1"]
        m2 = self.data.d["m2"]
        
        # if the current better price is wider than the min arb, then set the working order to that, else keep it as min distance            
        m1["working_levels"]["sell"] = max(m1["ask_price"],final_working["m1"]["sell"])        
        m2["working_levels"]["sell"] = max(m2["ask_price"],final_working["m2"]["sell"])    
        m1["working_levels"]["buy"]  = min(m1["bid_price"],final_working["m1"]["buy"])
        m2["working_levels"]["buy"]  = min(m2["bid_price"],final_working["m2"]["buy"])
        
        arb_status = ("\n"+" arb_price(): "+" m1_bid: "+str(m1["bid_price"])+" m1_ask: "+str(m1["ask_price"])+" m2_bid: "+str(m2["bid_price"])+" m2_ask: "+str(m2["ask_price"])+"\n"+
                           " arb_price(): "+" spread_price_weighted: "+str(self.data.spread_price_weighted)+"\n"+
                           " *arb_price(): "+" m1 ask_price: "+str(m1["ask_price"])+' final_working["m1"]["sell"]: '+str(final_working["m1"]["sell"])+' working_band["m1"]["sell"]: '+str(working_band["m1"]["sell"])+
                                             " buy_spread_target_price_m1: "+str(buy_spread_target_price_m1)+"\n"+
                           " *arb_price(): "+" m2_bid_price: "+str(m2["bid_price"])+' final_working["m2"]["buy"]: '+str(final_working["m2"]["buy"])+' working_band["m2"]["buy"]: '+str(working_band["m2"]["buy"])+
                                             " buy_spread_target_price_m2: "+str(buy_spread_target_price_m2)+"\n"+
                           " *arb_price(): "+" m1_bid_price: "+str(m1["bid_price"])+' final_working["m1"]["buy"]: '+str(final_working["m1"]["buy"])+' working_band["m1"]["buy"]: '+str(working_band["m1"]["buy"])+
                                             " sell_spread_target_price_m1: "+str(sell_spread_target_price_m1)+"\n"+
                           " *arb_price(): "+" m2_ask_price: "+str(m2["ask_price"])+' final_working["m2"]["sell"]: '+str(final_working["m2"]["sell"])+' working_band["m2"]["sell"]: '+str(working_band["m2"]["sell"])+
                                             " sell_spread_target_price_m2: "+str(sell_spread_target_price_m2)+"\n"+
                           " m1_working_levels: "+str(m1["working_levels"])+"\n"+
                           " m2_working_levels: "+str(m2["working_levels"])+"\n"+
                           " target_type: "+str(self.stat.quote_type)+"\n"+
                           " inv_qtys: "+str(self.stat.inv_qtys)+"\n"+
                           " inv_spreads: "+str(self.stat.inv_spreads)+"\n"+
                           " band_sizes: "+str(self.stat.inv_band_sizes))
        if arb_status != self.old_arb_status:
            self.old_arb_status = arb_status
            self.logging(arb_status)
        
        return None

    
    def price_data(self):
    ############ Price data ####################################
        error = 0
        while True:
            try:
                ## get data 
                band = self.data.band
                ### work out bands from stored 24h data, or set manual
                if self.var.manual == False:
                    minute_data = self.data.minute_data.copy()
                    if pd.to_datetime(minute_data["timestamp"][-1:].values[0]) < datetime.datetime.utcnow()-datetime.timedelta(minutes=5):
                        raise Exception("one minute stored data is not up to date")
                    if len(minute_data) < self.var.mov_avg:
                        None
                        #self.logging("one minute stored data is not sufficent in bars")
                     
                    funding_rates = minute_data[minute_data["instrument"]=="8h_funding"][-self.var.funding_mov_avg:]["ask_price"] # bid_price = 8h avg, ask_price = current(~1min)
                    self.data.funding_rate = funding_rates.iloc[-1]
                    self.data.funding_rate_avg = round(funding_rates.mean(),4)
                        
                    m1_min_raw = minute_data[minute_data["instrument"] == self.var.m1_sym]
                    m2_min_raw = minute_data[minute_data["instrument"] == self.var.m2_sym]
                    m1_mids_raw = m1_min_raw[["bid_price","ask_price"]].mean(axis=1).reset_index(drop=True)
                    m2_mids_raw = m2_min_raw[["bid_price","ask_price"]].mean(axis=1).reset_index(drop=True)
                    m1_mids = m1_mids_raw[-self.var.mov_avg:]
                    m1_mov_avg = m1_mids.mean()
                    m2_mids = m2_mids_raw[-self.var.mov_avg:]
                    m2_mov_avg = m2_mids.mean()
                    com_mid_avg = np.mean([m1_mov_avg, m2_mov_avg])
                    com_spread_mov_avg = m1_mov_avg - m2_mov_avg    
                    band_size = com_mid_avg*(self.var.margin + self.var.fee)
                    
                    band["buy"] = round((com_spread_mov_avg - band_size) * 2,0)/2
                    band["sell"] = round((com_spread_mov_avg + band_size) * 2,0)/2
                    
                    com_spreads = m1_mids - m2_mids
                    com_std_dev = com_spreads.std()
                    self.data.boll["buy"] = round((com_spread_mov_avg - (com_std_dev * self.var.boll)) * 2,0)/2
                    self.data.boll["sell"] = round((com_spread_mov_avg + (com_std_dev * self.var.boll)) * 2,0)/2
                    
                    ### filter based on mov_avg_speed middle
                    ## calculate mov_avg from one period back to see the differenc between now
                    m1_mids_before = m1_mids_raw[-(self.var.mov_avg+1):-1]
                    m1_mov_avg_before = m1_mids_before.mean()
                    m2_mids_before = m2_mids_raw[-(self.var.mov_avg+1):-1]
                    m2_mov_avg_before = m2_mids_before.mean()
                    com_spread_mov_avg_before = m1_mov_avg_before - m2_mov_avg_before
                    self.data.mov_avg_speed = ((com_spread_mov_avg - com_spread_mov_avg_before)*self.var.mov_avg)/(band_size*2)
                
                    
                    ### filter for price trending (1min price beyond mid avg rolling)
                    #window_len = 30
                    #m1_min_data = m1_min_raw[-window_len:]
                    #m2_min_data = m2_min_raw[-window_len:]
                    #perp_sell = [x-y for x,y in zip(list(m1_min_data["bid_price"].values),list(m2_min_data["ask_price"].values))]
                    #perp_buy = [x-y for x,y in zip(list(m1_min_data["ask_price"].values),list(m2_min_data["bid_price"].values))]
                    #mids_window_len = window_len + self.var.mov_avg
                    #m1_mids_trend = m1_mids_raw[-mids_window_len:].values
                    #m2_mids_trend = m2_mids_raw[-mids_window_len:].values
                    #mids = [x-y for x,y in zip(list(m1_mids_trend),list(m2_mids_trend))]
                    #percentages = []
                    #for i in range(len(perp_sell)):
                    #    side = False
                    #    rolling_mid = np.mean(mids[i:self.var.mov_avg+i])
                    #    if perp_sell[i] > rolling_mid:
                    #        percentages.append(1)
                    #        side = True
                    #    if perp_buy[i] < rolling_mid:
                    #        percentages.append(-1)
                    #        side = True      
                    #    if side == False:
                    #        percentages.append(0)
                    ##print(percentages)
                    #self.data.beyond_mid_avg_rolling = np.mean(percentages)
                    
                    
                
                elif self.var.manual == True:
                    band["buy"] = algo.var.buy_price
                    band["sell"] = algo.var.sell_price
                
                self.logging("price_data()","buy_band",band["buy"],"sell_band",band["sell"],"funding rate:",self.data.funding_rate_avg,"manual trading",self.var.manual) 
                return None
            
            except Exception as e:
                if self.testing == True:
                    raise e
                self.logging("data error price_data()",str(e),"relooping")
                error +=1
                if error == 20:
                    raise e
                continue

                
    def check_for_fills(self,known_info,order_id="",all_orders="",msg=""):
        order = []
        hedge = False
        cancelled_no_message = False
        cancelled = False
        hedge_qty = 0
        qty_left = known_info["quote_qty_left"]

        if all_orders != "" and msg == "":
            try:
                order = all_orders[order_id]
            except KeyError:
                cancelled_no_message = True
                status = "cancelled"   
        elif msg != "" and all_orders == "":
            order = msg
        else:
            raise Exception("wrong parameters")
        
        if cancelled_no_message == False:
            side = order["direction"]
            status = order["state"]
            qty_left = int(order["quantity"] - order["filledQuantity"])
                    
            if qty_left < known_info["quote_qty_left"]:
                hedge = True
                hedge_qty = int(known_info["quote_qty_left"] - qty_left)
                self.logging("(check_for_fills)","hedge",hedge,". A",side,"quote order has been (order_status) ",status," by (hedge_qty):",hedge_qty, "Order in question:",order)    
                        
        if status == "cancelled":
            cancelled = True
            self.logging("check_for_fills()","order_id",order_id,"has been cancelled!?!?",order)            

        return order, hedge, hedge_qty, qty_left, cancelled    

    
    def list_known_working_orders(self,add_blank_msg=False):
        known_working_orders = []
        for m in ["m1","m2"]:
            inst = self.data.d[m]
            for side in ["buy","sell"]:
                if inst["working_orders"][side] != {}:
                    if add_blank_msg == True:
                        known_working_orders.append((m,side,"")) ##inst,side,message
                    else:
                        known_working_orders.append((m,side))
        return known_working_orders  

    
    def remove_order_action_dups(self,info,only_execute_known_orders=False):
        ##### remove dupes from order actions #########            
        new_info = []
        for order in info:
            if order not in new_info:
                if only_execute_known_orders == True:
                    q_inst = order[0]
                    q_side = order[1]
                    known_orders = self.list_known_working_orders()
                    if (q_inst,q_side) in known_orders:
                        new_info.append(order)
                else:
                    new_info.append(order)                
        return new_info        

    
    # master_check_for_fills = [("m1","sell",msg=""),("m2","sell",msg="")]
    def master_check_for_fills(self,info="",all_orders=False):
    ###### any fills from order data? ######    
        if all_orders == True:
            info = self.list_known_working_orders(add_blank_msg=True)
        else: 
            ##### remove dupes from order actions #########
            self.logging("master_check_for_fills()","before remove_dups info",info)
            info = self.remove_order_action_dups(info,only_execute_known_orders=True)  
            self.logging("master_check_for_fills()","after remove_dups info",info)
        ###############################################
        
        ## if you find a tuple that contains no message, then you will need to do a poll for its current status (deri_all_orders)
        need_poll = True if len([order for order in info if order[2] == ""]) > 0 else False
        all_orders = self.deri_all_orders(25) if need_poll == True else ""
                    
        for m, side, message in info: 
            inst = self.data.d[m]
            known_info = inst["working_orders_info"][side]
            order_id = inst["working_orders"][side]["orderId"] ## get current working order id
            ## reassign it with new data after checking for fills ##
            inst["working_orders"][side], hedge, known_info["hedge_qty"], known_info["quote_qty_left"], cancelled = self.check_for_fills(known_info,order_id,all_orders,message)
            if hedge == True:
                self.logging('stat.order_action = "send_hedge" (master_check_for_fills)',m,side,known_info["hedge_qty"])
                self.setup_hedge_orders(m,side)
            if cancelled == True:
                inst["working_orders"][side] = {} 
                inst["working_orders_info"][side]["state"] = "cancelled"
                inst["working_orders_info"][side]["original_size"] = np.nan
                if self.stat.quote[m][side] == True:
                    self.logging('stat.order_action = "new_quote" (master_check_for_fills) because order was cancelled',m,side)
                    self.stat.order_action["new_quote"].append((m,side))
       
    
    # self.stat.order_action["new_quote"] = ()
    # self.stat.order_action["new_quote"].append(("m1","buy"))
    def new_quotes(self):
    ######## new quote orders ################################  
    #### there are no open positions in either markets, and no working orders ### 
    
        ##### remove dupes from order actions #########
        self.stat.order_action["new_quote"] = self.remove_order_action_dups(self.stat.order_action["new_quote"])  
        if len(self.stat.order_action["new_quote"]) == 0: 
            self.logging("there are no new orders to send new_quotes()","\n")
            return None
        ###############################################
    
        while True:
            service_error = False
            
            self.logging('order_action["new_quote"]',self.stat.order_action["new_quote"],"m1 working levels",self.data.d["m1"]["working_levels"],"m2 working levels",self.data.d["m2"]["working_levels"],"new_quotes()")
            
            ############# Order Info #####################################################################################            
            quote_orders = {}
            for m, side in self.stat.order_action["new_quote"]:
                inst = self.data.d[m]
                indexer = inst["sym"]+"_"+side
                quote_orders[indexer] = {"inst":inst["sym"],
                                         "side":side,
                                         "order":"limit",
                                         "qty":inst["working_orders_info"][side]["quote_qty_left"],
                                         "price":inst["working_levels"][side],
                                         "market":m}
            
            self.logging(quote_orders.keys(),"attempting new_quotes()")
            
            loop = asyncio.get_event_loop()
            futures = []
            for quote_order in quote_orders.values():
                futures.append(loop.run_in_executor(None,self.deri_order,quote_order))
            
            ############# Send Orders #################################################################        
            result = loop.run_until_complete(asyncio.gather(*futures))  
            self.logging("new_quotes() async result:",result)
            ######### Errors and Assignments ###################################
            incomplete = []
            for info in self.stat.order_action["new_quote"]:
                m = info[0]
                inst = self.data.d[m]
                side = info[1]
                idx = self.stat.order_action["new_quote"].index(info)
                ### get order info ###
                new_order, service_error = result[idx]    
                if service_error == True:  
                    self.logging(info,"new_quote service error")
                    incomplete.append(info)
                    #time.sleep(0.1)
     
                else:
                    self.logging("quote_order",list(quote_orders.values())[idx],"outputted new_order",new_order)                                                   
                    inst["working_orders"][side] = new_order
                    inst["working_orders_info"][side]["original_size"] = new_order["quantity"]

                    hedge_inst = "m2" if m=="m1" else "m1"
                    bid_ask ="bid_price_weighted" if side=="buy" else "ask_price_weighted"
                    inst["working_orders_info"][side]["targeted_fill"] = self.data.d[hedge_inst][bid_ask]
            
            if incomplete == []:
                self.stat.order_action["new_quote"]= []
                #####    
                break
                #####     
            else:
                self.stat.order_action["new_quote"] = incomplete 
                ### Get new price data if looping, unpack ################
                self.assign_prices(conn="ws")    
                self.logging("new_quotes(), looping again as some orders were incomplete:",incomplete)
                

    # self.stat.order_action["cancel_quote"] = ()
    # self.stat.order_action["cancel_quote"].append(("m1","buy"))   
    def cancel_quotes(self,all_known_orders=False,check_fills=True):  
    ######## cancel quote orders ################################  
        if all_known_orders == True:
            self.stat.order_action["cancel_quote"] = self.list_known_working_orders()     

        ############# Order Info #####################################################################################      
        order_ids = []
        for m, side in self.stat.order_action["cancel_quote"]:
            inst = self.data.d[m]
            order_ids.append(inst["working_orders"][side]["orderId"])
            self.logging(inst["sym"],side,"attempting cancel_quotes()")

        self.logging("orderIds to cancel cancel_quotes()",order_ids)

        loop = asyncio.get_event_loop()
        futures = []
        for order_id in order_ids:
            futures.append(loop.run_in_executor(None,self.deri_cancel,order_id))
            
        ############# Send Orders #################################################################        
        result = loop.run_until_complete(asyncio.gather(*futures))  
        self.logging("cancel_quotes() async result:",result)
        ######### Errors and Assignments ###################################
        check_fill_messages = []
        for idx in range(len(result)):
            cancelled_order = result[idx][0]
            hedge = result[idx][1]
            m, side = self.stat.order_action["cancel_quote"][idx]
            inst = self.data.d[m]
            if hedge == True:
                self.logging("cancel_quotes() has found an order that needs to be hedged",inst["sym"],side,cancelled_order)
            check_fill_messages.append((m,side,cancelled_order))
            self.logging("cancel_quotes()","market:",m,"\n","side:",side,"\n","cancelled_order:",cancelled_order)
        
        self.stat.order_action["cancel_quote"] = [] 
        
        if check_fills == True:
            self.logging("cancel_quotes() messages to send to master_check_for_fills",check_fill_messages)
            self.master_check_for_fills(check_fill_messages)   

            
    # self.stat.order_action["amend_quote"] = ()
    # self.stat.order_action["amend_quote"].append(("m1","buy"))
    def amend_quotes(self):
    ################## AMEND ###########################################        
    ################ amend quote orders ############################## 
    
        ##### remove dupes from order actions #########
        self.stat.order_action["amend_quote"] = self.remove_order_action_dups(self.stat.order_action["amend_quote"],only_execute_known_orders=True) 
        if len(self.stat.order_action["amend_quote"]) == 0: 
            self.logging("there are no orders to amend amend_quotes()")        
            return None
        ###############################################
        
        while True:              
            ############# Order Info ##################################################################################### 
            self.logging("amend_quote()",'self.stat.quote_type',self.stat.quote_type) 
            amend_orders = {}
            for m,side in self.stat.order_action["amend_quote"]:
                inst = self.data.d[m]
                indexer = inst["sym"]+"_"+side
                amend_orders[indexer] = {"order_id":inst["working_orders"][side]["orderId"],
                                         "price":inst["working_levels"][side],
                                         "qty":inst["working_orders_info"][side]["original_size"]}
                                        
            self.logging(amend_orders.keys(),"attempting amend_quotes()")
            
            loop = asyncio.get_event_loop()
            futures = []
            for amend_order in amend_orders.values():
                futures.append(loop.run_in_executor(None,self.deri_amend_order,amend_order))            
             
            ############# Send Orders #################################################################        
            result = loop.run_until_complete(asyncio.gather(*futures)) 
            self.logging("amend_quotes() async result:",result)
            ######### Errors and Assignments ###################################
            incomplete = []
            check_fill_messages = []
            for info in self.stat.order_action["amend_quote"]:
                error = 0
                ws_error = False
                found_order = False
                m = info[0]
                inst = self.data.d[m]
                side = info[1] 
                idx = self.stat.order_action["amend_quote"].index(info)
                ### get order info ###
                amend_order, service_error, order_error = result[idx]    
                if service_error == True:
                    self.logging("service error during amend orders, probably should look to check if filled at some point, maybe after 5 passes?") 
                    incomplete.append(info)
     
                if order_error == True:
                    self.logging("order may have been filled or cancelled? amend_quote()",self.stat.order_action,inst["working_orders"][side]) 
                    ### need to poll websocket until order state changes from open (because it clearly isnt open anymore?
                    while True:
                        all_orders = self.deri_all_orders(25)
                        ids = inst["working_orders"][side]['orderId']
                        try:
                            amend_order = all_orders[ids]
                            self.logging("amend_quotes()","polling ws order data for state change",amend_order)
                        except Exception as e:
                            print("amend_quotes()","polling ws data failed, getting data manually",str(e))
                            ws_error = True

                        if amend_order["state"] == "open" or ws_error == True:
                            error +=1
                            if error > 2 or ws_error ==True:
                                ### manually poll if you cannot find order from ws after 10 passes
                                amend_order, cancelled = self.deri_order_status(ids)
                                self.logging("amend_quotes() had to do MANUAL REST  call for amend order because websocket didnt have it",amend_order)
                                found_order = True
                                break
                            time.sleep(0.005)
                            continue
                        else:
                            self.logging("amend_quotes()","found change!")
                            found_order = True
                            break                   
                else:
                    found_order = True
                
                if found_order == True:
                    hedge_inst = "m2" if m=="m1" else "m1"
                    bid_ask ="bid_price_weighted" if side=="buy" else "ask_price_weighted"
                    inst["working_orders_info"][side]["targeted_fill"] = self.data.d[hedge_inst][bid_ask]
                    check_fill_messages.append((m,side,amend_order))
                    
            self.master_check_for_fills(check_fill_messages) ## quick update without polling
                                    
            if incomplete == []:
                self.stat.order_action["amend_quote"]= []
                #####    
                break
                #####     
            else:
                self.stat.order_action["amend_quote"] = incomplete
                ### Get new price data if looping, unpack ################
                self.assign_prices(conn="ws")  
                self.logging("amend_quotes(), looping again as some amends were incomplete:",incomplete)
  

    # self.stat.order_action["send_hedge"] = ()
    # self.stat.order_action["send_hedge"].append(("m1","buy"))
    def setup_hedge_orders(self,q_market,q_side):
        best_entry_idx = np.nan
        ## set hedge instrument
        
        quote_price = self.data.d[q_market]["working_orders"][q_side]["price"]
        
        if q_market == "m1":
            h_market = "m2"
        elif q_market == "m2":
            h_market = "m1"
            
        if q_side == "sell":
            h_side = "buy"
            hedge_limit = round(self.data.d[h_market]["ask_price"] + self.var.pay_up_ticks,0)
        elif q_side == "buy":
            h_side = "sell"
            hedge_limit = round(self.data.d[h_market]["bid_price"] - self.var.pay_up_ticks,0)
        
        if (q_market == "m1" and q_side == "sell") or (q_market == "m2" and q_side == "buy"): 
            full_inv_qty = -self.var.size
            ## set whether its opening a new position, or closing an old one
            if self.stat.inv_count <= 0:
                position_action = "open"
            else:
                position_action = "close"
                ## close the best position in terms of entry 
                best_entry_idx = self.stat.inv_spreads.index(min(self.stat.inv_spreads))    
                
        elif (q_market == "m1" and q_side == "buy") or (q_market == "m2" and q_side == "sell"): 
            full_inv_qty = self.var.size             
            ## set whether its opening a new position, or closing an old one
            if self.stat.inv_count >= 0:
                position_action = "open"
            else:
                position_action = "close"
                ## close the best position in terms of entry 
                best_entry_idx = self.stat.inv_spreads.index(max(self.stat.inv_spreads))              
        
        self.stat.order_action["send_hedge"].append([q_market, q_side, h_market, h_side, hedge_limit, quote_price, position_action, best_entry_idx, full_inv_qty])  


    # self.stat.order_action["send_hedge"] = ()
    # self.stat.order_action["send_hedge"].append(("m1","buy"))
    def send_hedges(self):
    ############## HEDGE #######################################                
    ########### hedge market orders ##############################
    
        ##### remove dupes from order actions #########
        self.stat.order_action["send_hedge"] = self.remove_order_action_dups(self.stat.order_action["send_hedge"],only_execute_known_orders=True)
        if len(self.stat.order_action["send_hedge"]) == 0: 
            self.logging("there are no hedge orders to send send_hedges()")               
            return None
        ###############################################
        
        while True:                     
            hedge_orders = {}
            for q_market, q_side, h_market, h_side, hedge_limit, *_ in self.stat.order_action["send_hedge"]:
                q_inst = self.data.d[q_market]
                h_inst = self.data.d[h_market]
                indexer = q_inst["sym"]+"_"+q_side
                hedge_orders[indexer] = {"inst":h_inst["sym"],
                                         "side":h_side,
                                         "order":"market",
                                         "qty":q_inst["working_orders_info"][q_side]["hedge_qty"],
                                         "price":hedge_limit,
                                         "market":h_market}
                
                self.logging(hedge_orders[indexer],"attempting send_hedges()")
            
            loop = asyncio.get_event_loop()
            futures = []
            for hedge_order in hedge_orders.values():
                futures.append(loop.run_in_executor(None,self.deri_order,hedge_order)) 
            
            ############# Send Orders #################################################################        
            result = loop.run_until_complete(asyncio.gather(*futures))   
            self.logging("send_hedges() result:",result)
            ######### Errors and Assignments ###################################
            incomplete = []
            for info in self.stat.order_action["send_hedge"]:
                idx = self.stat.order_action["send_hedge"].index(info)
                ### get order info ###
                new_order, service_error = result[idx]    
                if service_error == True:  
                    self.logging(self.data.d[info[0]]["sym"],info[1],"send_hedges() service error")
                    incomplete.append(info)
                    #time.sleep(0.1)
     
                else:
                    q_market, q_side, h_market, h_side, hedge_limit, quote_price, position_action, best_entry_idx, full_inv_qty = self.stat.order_action["send_hedge"][idx]    
                    q_inst = self.data.d[q_market]
                    h_inst = self.data.d[h_market]
                    
                    hedge_price = new_order["avgPrice"]
                    if q_market == "m1":
                        spread_price = quote_price - hedge_price
                    elif q_market == "m2":
                        spread_price = hedge_price - quote_price
                        
                    self.logging("send_hedges()",list(hedge_orders.values())[idx],"hedge_price",hedge_price,"spread_price",spread_price,"outputted new_order",new_order)                
                    ## if we have fully filled an order
                    if q_inst["working_orders_info"][q_side]["quote_qty_left"] == 0:
                        ## self.stat.inv_qtys = [-50,-50,-50,-50,-50]  ## self.stat.inv_spreads = [15,20,17,22,11]  ## self.stat.inv_band_sizes = [12,10,12,11,11]
                        ## self.data.d["m1"]["partial"] = {"buy":{"spreads":[], "band_sizes":[], "qtys":[]},"sell":{"spreads":[], "band_sizes":[], "qtys":[]}}
                        if position_action == "close":
                            if self.stat.quote_type[q_side] == "fixed_target":
                                order_idx = best_entry_idx
                            elif self.stat.quote_type[q_side] == "band":
                                order_idx = 0
                            del self.stat.inv_spreads[order_idx]
                            del self.stat.inv_qtys[order_idx]
                            del self.stat.inv_band_sizes[order_idx]                        
                        elif position_action == "open":
                            if len(q_inst["partial"][q_side]["spreads"]) >0:
                                spread_weighted = sum([x*y for x,y in zip(q_inst["partial"][q_side]["spreads"],q_inst["partial"][q_side]["qtys"])])
                                spread_avg = spread_weighted/sum(q_inst["partial"][q_side]["qtys"])
                                self.stat.inv_spreads.append(spread_avg)
                                band_weighted = sum([x*y for x,y in zip(q_inst["partial"][q_side]["band_sizes"],q_inst["partial"][q_side]["qtys"])])
                                band_avg = band_weighted/sum(q_inst["partial"][q_side]["qtys"])
                                self.stat.inv_band_sizes.append(band_avg)
                                q_inst["partial"][q_side] = {"spreads":[],"band_sizes":[],"qtys":[]}
                            else:
                                self.stat.inv_spreads.append(spread_price)
                                comms = (q_inst["working_orders"][q_side]["price"]*self.var.maker_fee) + (hedge_price*self.var.taker_fee)
                                self.stat.inv_band_sizes.append((self.data.band["sell"]-self.data.band["buy"]) + self.var.slippage + comms) ## only one side of slippage as the target is added onto an exsiting entry
                            ### negative inv_qty if we went short on m1, positive if we went long on m1 ##
                            self.stat.inv_qtys.append(full_inv_qty)
                            
                        #### complete order! ###
                        if self.var.manual == True:
                            if full_inv_qty >0:
                                self.var.buy_reload -= 1
                            elif full_inv_qty <0:
                                self.var.sell_reload -=1
                        targeted_price = q_inst["working_orders_info"][q_side]["targeted_fill"]
                        self.stat.inv_count = sum(self.stat.inv_qtys)/self.var.size             
                        self.logging("order_action:send_hedges()","hedge instrument:",h_market,"hedge side:",h_side,"another trade completed!","open/close position:",position_action,
                                     "full_inv_qty:",full_inv_qty,"best_entry_index:",best_entry_idx,"spread_price:",spread_price,"targeted_fill:",targeted_price,"inv_count:",self.stat.inv_count)
                        
                        targeted_spread = quote_price-targeted_price if q_market == "m1" else targeted_price-quote_price
                        inv_side = "buy" if full_inv_qty>0 else "sell"

                        slip_side = -full_inv_qty/abs(full_inv_qty)
                        now_time_excel = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S:%f")
                        new_dict = {"quote_id":q_inst["working_orders"][q_side]["orderId"],"hedge_id":new_order["orderId"],"quote instrument":q_market,"quote side":q_side,"real_side":inv_side,
                                    "open/close position":position_action,"full_inv_qty":full_inv_qty,"best_entry_index":best_entry_idx,"quote_price":quote_price,"hedge_targeted_price":targeted_price,
                                    "hedge_price":hedge_price,"targeted_spread":targeted_spread,"final_spread_price":spread_price,"slippage":(targeted_spread-spread_price)*slip_side,
                                    "inv_count":self.stat.inv_count,"my_time":now_time_excel,"inv_count":self.stat.inv_count}
                        self.logging("trade overview:",new_dict)
                        #write(new_dict,"slow","deribit_live_trades.csv")   
                        q_inst["working_orders"][q_side] = {}
                        q_inst["working_orders_info"][q_side] = {"hedge_qty":0, "quote_qty_left":self.var.size, "original_size":np.nan, "state":"", "targeted_fill":0}
                        
                    else:
                        ## self.data.d["m1"]["partial"] = {"buy":{"spreads":[], "band_sizes":[], "qtys":[]},"sell":{"spreads":[], "band_sizes":[], "qtys":[]}}
                        self.logging('q_inst["partial"]',q_inst["partial"],'q_inst["working_orders"]',q_inst["working_orders"])
                        q_inst["partial"][q_side]["spreads"].append(spread_price)
                        q_inst["partial"][q_side]["qtys"].append(q_inst["working_orders"][q_side]["quantity"])
                        comms = (q_inst["working_orders"][q_side]["price"]*self.var.maker_fee) + (hedge_price*self.var.taker_fee)
                        q_inst["partial"][q_side]["band_sizes"].append((self.data.band["sell"]-self.data.band["buy"]) + self.var.slippage + comms) ## only one side of slippage as the target is added onto an exsiting entry
    
            
            if incomplete == []:
                self.stat.order_action["send_hedge"]= []
                self.logging("send_hedges() ending...",
                             'self.stat.order_action',self.stat.order_action,
                             'self.stat.inv_count:',self.stat.inv_count,
                             'self.stat.inv_spreads:',self.stat.inv_spreads,
                             'self.stat.inv_qtys:',self.stat.inv_qtys,
                             'self.stat.inv_band_sizes:',self.stat.inv_band_sizes,
                             "open orders m1:",self.data.d["m1"]["working_orders"],
                             "open orders m2:",self.data.d["m2"]["working_orders"])
                #####    
                break
                #####     
            else:
                self.stat.order_action["send_hedges"] = incomplete 
                ### Get new price data if looping ################
                self.assign_prices(conn="ws")     
                self.logging("send_hedges(), looping again as some hedges were incomplete:",incomplete)
 

    def dont_quote(self,best_bid_offered):
        #### variable #################
        # do we have to be best bid/offered to quote?
        best_bid_offered = False
        ################################
        ## shortcuts ##
        quote = self.stat.quote
        order_action = self.stat.order_action
        data = self.data.d
        no_quote = []
        
        ## helper function ##
        def order_alive(m,side):
            inst = data[m]
            if quote[m][side] == True and inst["working_orders"][side] != {}:
                return True
            else:
                return False

        ## reset quotes
        quote["m1"] = self.var.quote_m1.copy()
        quote["m2"] = self.var.quote_m2.copy()
        ## cancel all orders
        cancel_all_known = False
        
        #if (ws_check["last ws latency s"] >= ws_check["avg ws latency s"]+4 or ws_check["avg ws latency s"] > 4):
        #    if quote==True:
        #        self.logging("Dont quote because websocket is slow") 
        #        self.logging("inv_count:",self.stat.inv_count,"order_action:",order_action,"dont_quote()",datetime.datetime.now())
        #        ## cancel any working orders and reset order action
        #        high_level_cancel_all()
        #        quote = False
        
        #### if quotes change (normally due to manual intervention in the settings) ####
        for m in ["m1","m2"]:
            inst = data[m]
            for side in ["buy","sell"]:
                if quote[m][side] == False and inst["working_orders"][side] != {}:
                    self.logging("Dont_quote() cancel order because stat.quote is false yet you have a live position",m,side) 
                    order_action["cancel_quote"].append((m,side))          
        
        #### inventory count and contract change ####
        if self.var.change_contracts == True:
            no_quote.append(["attempting to change contracts so maxuimum inv has been set to:",self.var.change_contracts_max_inv,self.var.change_contracts])
        if self.stat.inv_count >= self.var.max_inv or self.stat.inv_count >= self.var.change_contracts_max_inv:
            no_quote.append(["Dont_quote, maxuimum long positions hit","inv_count",self.stat.inv_count,"max_inv:",self.var.max_inv,
                             "self.var.change_contracts_max_inv:",self.var.change_contracts_max_inv,self.var.change_contracts])
            if any([order_alive("m1","buy"),order_alive("m2","sell")]):
                ### always as a result of sending a sucessful hedge, therefore one side may be still present and needs to be cancelled 
                self.logging("cancelling working orders","Dont_quote, maxuimum long positions hit","inv_count",self.stat.inv_count)
                ## cancel any working orders 
                order_action["cancel_quote"].append(("m1","buy")) 
                order_action["cancel_quote"].append(("m2","sell"))  
            quote["m1"]["buy"] = False
            quote["m2"]["sell"] = False           
        elif self.stat.inv_count <= -self.var.max_inv or self.stat.inv_count <= -self.var.change_contracts_max_inv:
            no_quote.append(["Dont_quote, maxuimum short positions hit","inv_count",self.stat.inv_count,"max_inv:",-self.var.max_inv,
                             "self.var.change_contracts_max_inv:",-self.var.change_contracts_max_inv,self.var.change_contracts])
            if any([order_alive("m1","sell"),order_alive("m2","buy")]):
                ### always as a result of sending a sucessful hedge, therefore one side may be still present and needs to be cancelled
                self.logging("cancelling working orders","Dont_quote, maxuimum short positions hit","inv_count",self.stat.inv_count)
                ## cancel any working orders 
                order_action["cancel_quote"].append(("m1","sell")) 
                order_action["cancel_quote"].append(("m2","buy"))  
            quote["m1"]["sell"] = False
            quote["m2"]["buy"] = False
            
        #### dont quote if you are going to cross on yourself for sure ####
        times_qty = 10
        cancel_cross_quotes = []
        for m in ["m1","m2"]:
            market = data[m]
            for side, bid_ask in [["sell","ask_price"],["buy","bid_price"]]:
                if order_alive(m,side):
                    if (market["working_orders"][side]["price"] == market[bid_ask]): #and market["working_orders_info"][side]["quote_qty_left"] > market[bid_ask_size]/times_qty):
                        ### always as a result of sending a sucessful hedge, therefore one side may be still present and needs to be cancelled 
                        ## cancel any working orders 
                        del_inst = "m2" if m=="m1" else "m1"              
                        ### if there is already an m1 order that need to be cancelled on the same side, you can leave the m2 on the same side in, dont cancel it
                        if m == "m2" and ("m1",side) in cancel_cross_quotes:
                            None
                        else:
                            self.logging("cancelling working orders","Dont_quote, as m2 could cross m1","quote_side:",side,"market:",m,"price:",market[bid_ask])
                            cancel_cross_quotes.append((del_inst,side))
        for inst, side in cancel_cross_quotes:
            order_action["cancel_quote"].append((inst,side)) 
            quote[inst][side] = False
       
        #### Manual Trading ###
        if self.var.manual == True:
            #### reload count ####
            if self.var.buy_reload == 0:
                no_quote.append(["Dont_quote, completed all buy reloads","buy_reload",self.var.buy_reload])
                if any([order_alive("m1","buy"),order_alive("m2","sell")]):
                    ### always as a result of sending a sucessful hedge, therefore one side may be still present and needs to be cancelled 
                    self.logging("cancelling working orders","Dont_quote, completed all buy reloads","buy_reload",self.var.buy_reload)
                    ## cancel any working orders 
                    order_action["cancel_quote"].append(("m1","buy")) 
                    order_action["cancel_quote"].append(("m2","sell"))  
                quote["m1"]["buy"] = False
                quote["m2"]["sell"] = False           
            elif self.var.sell_reload ==0:
                no_quote.append(["Dont_quote, completed all sell reloads","sell_reload",self.var.sell_reload]) 
                if any([order_alive("m1","sell"),order_alive("m2","buy")]):
                    ### always as a result of sending a sucessful hedge, therefore one side may be still present and needs to be cancelled
                    self.logging("cancelling working orders","Dont_quote, completed all sell reloads","sell_reload",self.var.sell_reload)
                    ## cancel any working orders 
                    order_action["cancel_quote"].append(("m1","sell")) 
                    order_action["cancel_quote"].append(("m2","buy"))  
                quote["m1"]["sell"] = False
                quote["m2"]["buy"] = False
        
        #### Algo Trading ###
        elif self.var.manual == False:               
            #### bid ask spread size ####
            if (data["m1"]["spread"] > self.var.max_m1_bid_ask_spread or data["m2"]["spread"] > self.var.max_m2_bid_ask_spread):
                no_quote.append(["Dont_quote because wide spread","m1 spread",data["m1"]["spread"],"m2 spread",data["m2"]["spread"],"funding_filter:",self.var.abs_funding_filter,"dont_quote()"])
                if any([order_alive(m,side) for m in ["m1","m2"] for side in ["buy","sell"]]): 
                    ## cancel any working orders
                    self.logging("cancelling working orders","Dont_quote because wide spread","m1 spread",data["m1"]["spread"],"m2 spread",data["m2"]["spread"],"funding_filter:",self.var.abs_funding_filter,"dont_quote()")
                    cancel_all_known = True
                quote["m1"]["buy"] = False
                quote["m1"]["sell"] = False
                quote["m2"]["buy"] = False
                quote["m2"]["sell"] = False
            
            #### current spread price level ####
            if (max(self.data.band["sell"],self.data.spread_price_weighted["sell"]) < self.var.min_sell_price and self.stat.inv_count <= 0 and
                self.data.funding_rate_avg < self.var.abs_funding_any):
                no_quote.append(["Dont_quote, sell bands are out of range (sell too low)",self.data.band,"funding_rate_avg",self.data.funding_rate_avg,"funding_filter:",self.var.abs_funding_filter,"dont_quote()"])
                if any([order_alive("m1","sell"),order_alive("m2","buy")]):
                    ## cancel any working orders 
                    self.logging("cancelling working orders","Dont_quote, sell bands are out of range (sell too low)",self.data.band,"funding_rate_avg",self.data.funding_rate_avg,"funding_filter:",self.var.abs_funding_filter,"dont_quote()")
                    order_action["cancel_quote"].append(("m1","sell")) 
                    order_action["cancel_quote"].append(("m2","buy"))
                quote["m1"]["sell"] = False
                quote["m2"]["buy"] = False               
            elif (min(self.data.band["buy"],self.data.spread_price_weighted["buy"])> self.var.max_buy_price and self.stat.inv_count >= 0 and
                  self.data.funding_rate_avg > -self.var.abs_funding_any):
                no_quote.append(["Dont_quote, buy bands are out of range (buy too high)",self.data.band,"funding_rate_avg",self.data.funding_rate_avg,"funding_filter:",self.var.abs_funding_filter,"dont_quote()"])
                if any([order_alive("m1","buy"),order_alive("m2","sell")]):
                    ## cancel any working orders 
                    self.logging("cancelling working orders")
                    order_action["cancel_quote"].append(("m1","buy")) 
                    order_action["cancel_quote"].append(("m2","sell"))
                quote["m1"]["buy"] = False
                quote["m2"]["sell"] = False        
            
            ### funding rate #####
            if self.stat.inv_count <= 0:  
                funding_rate = min(self.data.funding_rate/5,self.data.funding_rate_avg) ## use both current(/1.5) and average funding rate as a filter
                if ((funding_rate < -self.var.abs_funding_filter and self.data.band["sell"] < self.var.abs_band_any_with_max_funding) or  ## funding rate is not great AND sell band is not in perfect position
                    (funding_rate <= -self.var.abs_funding_any)): ## funding rate is is literally horrible 
                    no_quote.append(["Dont_quote(), funding is negative (costs to sell as perp is below index)","funding avg:",self.data.funding_rate_avg,"curr funding:",self.data.funding_rate,
                                     "filter:",-self.var.abs_funding_filter])
                    if any([order_alive("m1","sell"),order_alive("m2","buy")]):
                        ## cancel any working orders 
                        self.logging("cancelling working orders","Dont_quote(), funding is negative (costs to sell as perp is below index)","funding:",self.data.funding_rate_avg,"filter:",-self.var.abs_funding_filter)
                        order_action["cancel_quote"].append(("m1","sell")) 
                        order_action["cancel_quote"].append(("m2","buy"))
                    quote["m1"]["sell"] = False  
                    quote["m2"]["buy"] = False  
            elif self.stat.inv_count >= 0:
                funding_rate = max(self.data.funding_rate/5,self.data.funding_rate_avg) ## use both current(/1.5) and average funding rate as a filter
                if ((funding_rate > self.var.abs_funding_filter and self.data.band["buy"] > - self.var.abs_band_any_with_max_funding) or ## funding rate is not great AND buy band is not in perfect position
                    (funding_rate >= self.var.abs_funding_any)):  ## funding rate is is literally horrible                
                    no_quote.append(["Dont_quote(), funding is positive (costs to buy as perp is above index)","funding",self.data.funding_rate_avg,"curr funding:",self.data.funding_rate,
                                     "filter:",-self.var.abs_funding_filter])
                    if any([order_alive("m1","buy"),order_alive("m2","sell")]):
                        ## cancel any working orders 
                        self.logging("cancelling working orders","Dont_quote(), funding is positive (costs to buy as perp is above index)","funding",self.data.funding_rate_avg,"filter:",-self.var.abs_funding_filter)
                        order_action["cancel_quote"].append(("m1","buy")) 
                        order_action["cancel_quote"].append(("m2","sell"))
                    quote["m1"]["buy"] = False  
                    quote["m2"]["sell"] = False  
                    
            ### trending filter #####
            if self.stat.inv_count <= 0:  
                if self.data.mov_avg_speed > self.var.mov_avg_speed_filter:
                    no_quote.append(["Dont_quote(), it is trending higher (dangerous to sell)","mov_avg_speed",self.data.mov_avg_speed,"mov_avg_speed_filter",self.var.mov_avg_speed_filter])
                    if any([order_alive("m1","sell"),order_alive("m2","buy")]):
                        ## cancel any working orders 
                        self.logging("cancelling working orders","Dont_quote(), it is trending higher (dangerous to sell)","mov_avg_speed",self.data.mov_avg_speed,"mov_avg_speed_filter",self.var.mov_avg_speed_filter)
                        order_action["cancel_quote"].append(("m1","sell")) 
                        order_action["cancel_quote"].append(("m2","buy"))
                    quote["m1"]["sell"] = False  
                    quote["m2"]["buy"] = False  
            elif self.stat.inv_count >= 0:
                if self.data.mov_avg_speed < -self.var.mov_avg_speed_filter:               
                    no_quote.append(["Dont_quote(), it is trending lower (dangerous to buy)","mov_avg_speed",self.data.mov_avg_speed,"mov_avg_speed_filter",-self.var.mov_avg_speed_filter])
                    if any([order_alive("m1","buy"),order_alive("m2","sell")]):
                        ## cancel any working orders 
                        self.logging("cancelling working orders","Dont_quote(), it is trending lower (dangerous to buy)","mov_avg_speed",self.data.mov_avg_speed,"mov_avg_speed_filter",-self.var.mov_avg_speed_filter)
                        order_action["cancel_quote"].append(("m1","buy")) 
                        order_action["cancel_quote"].append(("m2","sell"))
                    quote["m1"]["buy"] = False  
                    quote["m2"]["sell"] = False 
                    
            #### best bid or offered ####
            if best_bid_offered == True:
                for inst_info in [("m1",quote["m1"]),("m2",quote["m2"])]:
                    m = inst_info[0]
                    inst = data[m]
                    quote = inst_info[1]
                    for side_info in [("buy","bid_price"),("sell","ask_price")]:
                        side = side_info[0]
                        best_price = side_info[1]
                        if inst["working_levels"][side] != inst[best_price]:
                            if order_alive(m,side):
                                self.logging("cancelling working orders","Dont_quote, not best offered",inst,side)
                                order_action["cancel_quote"].append((m,side))
                            quote[m][side] = False 
        
            #### do you have enough margin to open new positions?
            funds_available = round(self.data.account["availableFunds"],8)
            order_size_btc = round(self.var.size*10/data["m1"]["bid_price"],8)
            single_order_btc_initial_margin = round(order_size_btc/(100/(1+(order_size_btc/100)*0.5)),8)
            # remember you need to be able to hedge the other side too, so do margin for RT
            RT_order_comms = round(order_size_btc*(0.055/100),8)
            RT_order_btc_margin = round(((single_order_btc_initial_margin*2)+RT_order_comms)*1.5,8) ##1.3 is the weight that sort of close to real world order margin if far away from price
            num_RT_with_funds_available = int(funds_available/RT_order_btc_margin)
            num_buy_quotes = sum([quote["m1"]["buy"],quote["m2"]["sell"]])
            num_sell_quotes = sum([quote["m1"]["sell"],quote["m2"]["buy"]])
            num_buy_working = sum([len(data["m1"]["working_orders"]["buy"]),len(data["m2"]["working_orders"]["sell"])])
            num_sell_working = sum([len(data["m1"]["working_orders"]["sell"]),len(data["m2"]["working_orders"]["buy"])])
            # margin only increases if we are increasing a position
            funds_status = (" funds_available: "+str(funds_available)+" RT_order_comms: "+str(RT_order_comms)+" RT_order_btc_margin: "+str(RT_order_btc_margin)+" num_RT_with_funds_available: "+str(num_RT_with_funds_available)+
                            " num_buy_quotes: "+str(num_buy_quotes)+" num_sell_quotes: "+str(num_sell_quotes)+" inv_count: "+str(self.stat.inv_count))
            if funds_status != self.old_funds_status:
                self.old_funds_status = funds_status
                self.logging(funds_status)     
            if self.stat.inv_count >= 0:
                if num_RT_with_funds_available+num_buy_working < num_buy_quotes:
                    self.logging("cancelling working orders","not enough funds available to quote all buy orders dont_quote(")
                    del_buy_quotes = num_buy_quotes - num_RT_with_funds_available+num_buy_working
                    if del_buy_quotes == 1:
                        if quote["m1"]["buy"] == True:
                            quote["m1"]["buy"] = False
                            order_action["cancel_quote"].append(("m1","buy")) 
                        elif quote["m2"]["sell"] == True:
                            quote["m2"]["sell"] = False
                            order_action["cancel_quote"].append(("m2","sell")) 
                    elif del_buy_quotes == 2:
                        quote["m1"]["buy"] = False
                        quote["m2"]["sell"] = False
                        order_action["cancel_quote"].append(("m1","buy")) 
                        order_action["cancel_quote"].append(("m2","sell"))
            if self.stat.inv_count <= 0:
                if num_RT_with_funds_available+num_sell_working < num_sell_quotes:
                    self.logging("cancelling working orders","not enough funds available to quote all sell orders dont_quote(")
                    del_sell_quotes = num_sell_quotes - num_RT_with_funds_available+num_sell_working
                    if del_sell_quotes == 1:
                        if quote["m1"]["sell"] == True:
                            quote["m1"]["sell"] = False
                            order_action["cancel_quote"].append(("m1","sell")) 
                        elif quote["m2"]["buy"] == True:
                            quote["m2"]["buy"] = False
                            order_action["cancel_quote"].append(("m2","buy")) 
                    elif del_sell_quotes == 2:
                        quote["m1"]["sell"] = False
                        quote["m2"]["buy"] = False
                        order_action["cancel_quote"].append(("m1","sell")) 
                        order_action["cancel_quote"].append(("m2","buy"))
            
            #### total order size is it bigger than 4!
            total_order_size = sum([abs(o["quantity"]) for o in self.data.open_orders.values()])
            number_of_orders = len([abs(o["quantity"]) for o in self.data.open_orders.values()])
            ## because all quotes could potentially send a hedge order at the same time, that is a total of 8 orders that are theoretically possible
            if total_order_size > (8 * self.var.size):
                raise Exception ("dont_quote() more than twice max size is being quoted, stop and restart!","total order size",total_order_size,"var.size",var.size,"open orders",self.data.open_orders)
            if number_of_orders > 8:
                raise Exception ("dont_quote() more than 8 orders are open, stop and restart!","number of orders",number_of_orders,self.data.open_orders)
        
        ##################################################################################################################
        ### send to log file, any cancel statments that have changed since the last time ##
        if no_quote != self.old_dont_quote_status:
            self.old_dont_quote_status = no_quote
            for x in no_quote:
                self.logging(x)
            self.logging("stat.quote:",quote)
        ############## dont worry about order action dups as they are looked after in each functions #####################   
        ### remove dupes and only execute known orders ###
        #self.logging("dont_quote()","cancel_quotes before remove dups:",order_action["cancel_quote"])
        order_action["cancel_quote"] = self.remove_order_action_dups(order_action["cancel_quote"],only_execute_known_orders=True) 
        self.logging("dont_quote()","cancel_quotes after remove dups and known:",order_action["cancel_quote"]) if len(order_action["cancel_quote"])>0 else None
        #self.logging("dont_quote() ending stat.quote:",quote)
        
        if len(order_action["cancel_quote"]) > 0:
            self.cancel_quotes(cancel_all_known,check_fills=True) 
        
  
    def re_quote_and_working_levels(self):
        ###### if orders have made it past dont quote filter, then send in new orders if they are quotable again ###################    
        for m in ["m1","m2"]:
            inst = self.data.d[m]
            quote = self.stat.quote[m]
            for side in ["buy","sell"]:
                if quote[side] == True:
                    ### if there is no working order at the moment, but there should be ###
                    if inst["working_orders"][side] == {}:
                        self.stat.order_action["new_quote"].append((m,side))
                        self.logging("excited new quote",inst["sym"],side,"re_quote_and_working_levels()") 
                        
                    ### if there is a working order but the price has changed ###    
                    elif inst["working_levels"][side] != inst["old_working_levels"][side]:
                        self.stat.order_action["amend_quote"].append((m,side)) 
                        inst["old_working_levels"][side] = inst["working_levels"][side]
                        self.logging("excited amend quote",inst["sym"],side,"re_quote_and_working_levels()")                      
            ####logging("seems that something is tradable","order_action",self.stat.order_action,"inv_count",self.stat.inv_count)

           
    def auto_positions(self,first_pass=False):
        #[{'instrument': 'BTC-PERPETUAL', 'kind': 'future', 'size': 1, 'amount': 10.0, 'averagePrice': 3948.000101069, 'direction': 'buy', 'sizeBtc': 0.002531242, 'floatingPl': 1.686e-06,
        #  'realizedPl': -7.15e-05, 'estLiqPrice': -0.1, 'markPrice': 3950.63, 'indexPrice': 3953.74, 'maintenanceMargin': 1.4555e-05, 'initialMargin': 3.1008e-05, 'settlementPrice': 3968.95,
        #  'delta': 0.002531242, 'openOrderMargin': 0.0, 'profitLoss': 1.686e-06},{"instrument":"BTC-DEC28",...}]
        abs_inv_count = np.nan
        side = np.nan
        avg_spread_entry = np.nan
        hedged = False
        error = 0
        h_error = 0
        partial = 0
        
        for reloop in range(2): ##should only take the seconds loop to get hedged properly
            while True:
                try:
                    pos = self.client.positions()
                    break
                except Exception as e:
                    self.logging("auto_positions() error getting positions",str(e))
                    if error >10:
                        raise Exception ("error when polling positions")
                    error +=1
                    time.sleep(0.05)
            
            num_positions = len(pos)
            if num_positions == 0:
                abs_inv_count = 0
                side = 0
                avg_spread_entry = 0
                ## break from main loop and continue algo
                break            
            else:
                m1 = {}
                m2 = {}
                for x in pos:
                    if x["instrument"] == self.var.m1_sym:
                        m1, = [x for x in pos if x["instrument"]==self.var.m1_sym]
                    elif x["instrument"] == self.var.m2_sym:
                        m2, = [x for x in pos if x["instrument"]==self.var.m2_sym]
                if len(m1) == 0:
                    m1["size"] = 0
                if len(m2) == 0:
                    m2["size"] = 0
                
                ### if we are hedged, then calculate inventory and what side we are trading
                if (m1["size"] == -m2["size"]) or (-m1["size"] == m2["size"]):
                    ## abs_inv_count (total size/ trade size)
                    abs_inv_count = int(abs(m1["size"]/self.var.size))
                    partial = m1["size"] % self.var.size
                    ## side, 1 = long, -1 = short
                    if m1["direction"] == "buy":
                        side = 1
                    else:
                        side = -1
                    ## current average spread price
                    avg_spread_entry = round(m1["averagePrice"] - m2["averagePrice"],2)
                    ## break from main loop and continue algo
                    break                
                #### if we are not hedged, then send hegdge!
                else:                    
                    if abs(m1["size"]) > abs(m2["size"]):
                        emergency_hedge_inst = self.var.m1_sym
                        emrg_price = self.data.d["m1"]["bid_price"] ##any side will do for now
                    else:
                        emergency_hedge_inst = self.var.m2_sym
                        emrg_price = self.data.d["m2"]["bid_price"] ##any side will do for now
                    emrg_hedge_qty = m1["size"] + m2["size"] 
                    
                    if emrg_hedge_qty > 0:
                        emrg_side = "sell"
                        new_hedge = {"inst":emergency_hedge_inst,"side":emrg_side,"order":"market","qty":emrg_hedge_qty,"price":round(emrg_price/1.2,0)}
                    else:
                        emrg_side = "buy"
                        new_hedge = {"inst":emergency_hedge_inst,"side":emrg_side,"order":"market","qty":abs(emrg_hedge_qty),"price":round(emrg_price*1.2,0)}
                    
                    self.logging("emergency hedge is needed!","m1 size",m1["size"],"m2 size",m2["size"])
                    
                    if first_pass == True:
                        while True:
                            print()
                            print("m1 instrument:",self.var.m1_sym)
                            print("m2 instrument:",self.var.m2_sym)
                            print("emergency hedge is needed!")
                            print("m1 open position:",m1["size"])
                            print("m2 open position:",m2["size"])
                            print("Sending hedge for:",emergency_hedge_inst,"of size:",emrg_hedge_qty,"direction:",emrg_side)
                            user_input = input("Do you wish to continue (y/n)? - make sure all of the details are correct!\n")
                            if user_input == "y":
                                break
                            else:
                                self.logging("user declined emergency hedge")
                                print("press 'CTRL+C' to exit session")
                                raise KeyboardInterrupt
                    
                    while True:
                        try:
                            ############# Send Orders #################################################################
                            self.logging("sending emergency hedge",new_hedge)
                            new_order, service_error = self.deri_order(new_hedge)   
                            self.logging("finished sending emergency hedge","new_order",new_order,"service_error",service_error)
                            ######### Errors and Assignments ###################################         
                            if service_error == True:  
                                self.logging("emergency hedge serive error")
                                #time.sleep(0.1)
                            else:
                                break
                        except Exception as e:
                            self.logging("problem sending emergency hedge",e)
                            if h_error >10:
                                raise Exception ("error sending emergency hedge")
                            h_error +=1
                            time.sleep(0.05)           
                                
        return abs_inv_count, side, avg_spread_entry            

     
    def main_algo(self):
        ####### Initialisations and Declarations ##################
        first_pass = True
        terminate = False
        loop_counter = 0
        error_terminate = 0
        old_main_status = ()
        old_time = datetime.datetime.now().second
           
        ### reset variables ###
        self.reset_data_stat()
        self.var.change_contracts_max_inv = self.var.max_inv
            
        self.set_logging_file()
        
        ### START WEBSOCKET ##
        self.start_websocket_thread()
        #################################################
        while True:
            end = False
            heartbeat = False
            try:
                #### starting algo ###
                self.logging("\n","STARTING ALGO")
                #### reset data and strat variables ####
                self.reset_data_stat()
                ### cancelling any stading orders, stale that could have been left over ##
                self.deri_cancel_all()
                ### get huge 1min data file on first pass and save time
                self.set_minute_data()
                self.price_data()
                ### just for first pass, to visually check data look okay
                self.assign_prices(conn="ws")
                ### positions ##
                abs_inv_count, side, avg_spread_entry  = self.auto_positions(first_pass)
                first_pass = False
                self.logging("auto_positions","abs_inv_count",abs_inv_count,"side",side,"avg_spread_entry",avg_spread_entry)
                ## overrides ###
                #side = 1 #minus 1 for negative inventory/ short
                #inv_count = 7 #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                #avg_spread_entry = 5
                self.stat.inv_count = abs_inv_count*side
                self.stat.inv_qtys = [side*self.var.size]*abs_inv_count
                self.stat.inv_spreads = [avg_spread_entry]*abs_inv_count #  <<<<<<<<<<<<<< average spread/entry price
                #self.stat.inv_spreads = [-9,-9,-10,-8] # <<<<<<<<<<<<<<  individual spread prices
                self.stat.inv_band_sizes = [round(self.data.d["m1"]["bid_price"]*((self.var.margin*2)+(self.var.fee*2))+self.var.slippage,1)]*abs_inv_count   # <<<<<<<<<<<<< profit target based on btc price $
                self.logging("inv band sizes:",self.stat.inv_band_sizes)
                #self.stat.inv_band_sizes = [5]*len(self.stat.inv_qtys)  #<<<<<<<<<<<<< gross profit target from average price $ (remember fees + slippage are around $4) and remember its *fixed target size
                ############################################
                self.logging("stat:",vars(self.stat))
                ###############################################
                ### just for first pass, to visually check data look okay, working levels and bands ##
                self.assign_prices(conn="ws")
                self.logging("buy_band",self.data.band["buy"],"sell_band",self.data.band["sell"])
                ##############################################   
                self.logging("m1 working levels",self.data.d["m1"]["working_levels"],"m2 working levels",self.data.d["m2"]["working_levels"],"test",self.var.margin,"margin")
                ############
                
                ## start ###
                while True: 
                    loop_counter+=1
                    if self.var.settings_file == True:
                        self.get_settings()
                    
                    main_status = ("\n"+" var.quote_m1: "+str(self.var.quote_m1)+" var.quote_m2: "+str(self.var.quote_m2)+"\n"+
                                        " stat.quote['m1']: "+str(self.stat.quote["m1"])+" stat.quote['m2']: "+str(self.stat.quote["m2"])+"\n"+
                                        " m1 working orders: "+str(self.data.d["m1"]["working_orders"])+"\n"+
                                        " m1 working orders info: "+str(self.data.d["m1"]["working_orders_info"])+"\n"+
                                        " m2 working orders: "+str(self.data.d["m2"]["working_orders"])+"\n"+
                                        " m2 working orders info: "+str(self.data.d["m2"]["working_orders_info"])+"\n"+
                                        " bands: "+str(self.data.band)+" funding rate: "+str(self.data.funding_rate_avg)+" manual trading: "+str(self.var.manual))
                    
                    if main_status != old_main_status:
                        old_main_status = main_status
                        self.logging(main_status)
                    
                    ########### Get account data (margin..etc) ####
                    self.deri_account_and_positions()
                    
                    ########### Dont quote filter ########
                    self.dont_quote(best_bid_offered=False)   
                    
                    ######## check for any fills #########
                    self.master_check_for_fills(all_orders=True)                     
                    
                    if len(self.stat.order_action["send_hedge"]) > 0:
                        self.send_hedges()
                        self.assign_prices(conn="ws") ## get price data here to get more accurate requote
                        continue ### loop around to don't quote because max inventory might have been hit
                        
                    ######### get price data for use when needed ####
                    self.assign_prices(conn="ws") 
                    
                    ####### re_quote if any filters have dropped out from before or working levels have changed ####
                    self.re_quote_and_working_levels()                   
                        
                    ########### Order actions ###########     
                    if len(self.stat.order_action["new_quote"]) > 0:
                        self.new_quotes()
                        
                    if len(self.stat.order_action["amend_quote"]) > 0:
                        self.amend_quotes()
                        ## if sending hedges
                        if len(self.stat.order_action["send_hedge"]) > 0:
                            self.send_hedges()
                    
                    time.sleep(0.001)
                    ########### heartbeat 1min ######## 
                    now_time = datetime.datetime.utcnow()
                    if now_time.second >= 30:
                        heartbeat = True   
                    ### gives 10 seconds for 1min data to save out incase ws is slow
                    if now_time.second > 10 and now_time.second < 30 and heartbeat == True:
                        heartbeat = False
                        ### print the loop counter to see how manny loops you averaged a second ##
                        self.logging("loop counter!: ",loop_counter/60," loops a second on average")
                        loop_counter = 0
                        ### add/append another minutes worth of data to the dataframe and only keep the last week and assign data
                        self.get_minute_data()
                        self.price_data()
                        ## print and reset poll count
                        #print("number of REST polls last minute",self.stat.poll)
                        self.stat.poll = 0
                        ######## change contracts for trading #######
                        if self.var.manual == False:
                            if self.var.change_contracts == False:
                                new_fut, new_back =self.contract_sym(now_time,roll_buffer=24)
                                if (fut_sym,back_sym) != (new_fut,new_back):
                                    print("begin flatteneing positions to change contracts, max_inv is set to 0 effectively")
                                    self.logging("begin flatteneing positions to change contracts, max_inv is set to 0 effectively")
                                    self.var.change_contracts = True
                                    self.var.change_contracts_max_inv = 0

                            elif self.var.change_contracts == True:
                                if self.stat.inv_count == 0:
                                    new_fut, new_back =self.contract_sym(now_time,roll_buffer=24)
                                    self.var.m2_sym = new_fut
                                    print("positions have been flattened and contract changeover has commenced, new:",new_fut)
                                    self.logging("positions have been flattened and contract changeover has begun, new fut sym:",new_fut)
                                    self.var.change_contracts = False
                                    self.var.change_contracts_max_inv = self.var.max_inv
                                    raise KeyboardInterrupt
                                    #raise Exception("restarting algo so we can delete all orders and change contracts, new fut sym:",new_fut)
                                    
                                #### need to write a flatten all function ####
                                #else:     
                                #    new_fut, new_back =self.contract_sym(now_time,roll_buffer=2)
                                #    if (fut_sym,back_sym) != (new_fut,new_back):
                                #        print("emergency flatteneing positions to change contracts")
                                #        self.var.change_contracts = False                                
                            
                        ######## heartbeat 15min algo ################
                        if now_time.minute in [15,30,45,59]:
                            print("algo alive",self.account_name,now_time)
                        ######## create new hourly log file #######
                        elif now_time.minute == 0:
                            log = re.sub("[_][0-9]+.txt$","",self.var.log)
                            self.var.log = log+"_"+str(now_time.hour)+".txt"
                            print(now_time,"setting new log file",self.var.log)
                            self.set_logging_file()
                            self.clear_log()
                        ######## If monday morning, read in new 1min file #####
                        if now_time.minute == 0 and now_time.hour == 0 and now_time.weekday() == 0:
                            self.set_minute_data()
                            self.price_data()
                        ######### clear output #######################
                        elif now_time.minute == 0 and now_time.hour == 17 and now_time.date().weekday() in [0,1,2,3,4]:
                            clear_output(wait=True)
                        ####### log heartbeat #######################
                        self.logging("1min check >","buy/sell bands",(self.data.band["buy"],self.data.band["sell"]),
                                     "working levels m1",self.data.d["m1"]["working_levels"],
                                     "working levels m2",self.data.d["m2"]["working_levels"],
                                     "perp bid/ask",(self.data.d["m1"]["bid_price"],self.data.d["m1"]["ask_price"]),
                                     "fut bid/ask",(self.data.d["m2"]["bid_price"],self.data.d["m2"]["ask_price"]),
                                     "inv_count",self.stat.inv_count,"order_action",self.stat.order_action,"self.stat.quote_type",
                                     self.stat.quote_type,"self.stat.inv_band_sizes",self.stat.inv_band_sizes)
                        
                    ######## change to new data files ######
                    if now_time.minute == 0 and now_time.hour == 0 and now_time.second <=2:
                        self.set_opened_files()
                                
            ###############################################################################
            except KeyboardInterrupt:
                self.logging("\n","maunally stopping algo...")
                end = True
                terminate = True
                
            except Exception as e:
                try:
                    self.logging("\n","Uncaught Error occoured in algo, stopping algo...:"+str(e))
                    traceback.print_exc(file=open(self.var.log,"a+"))
                    traceback.print_exc(file=sys.stdout)
                    self.logging("\n","relooping and restarting algo !!!!!!!!!!!!!!!","\n")
                    if self.testing == False:
                        print("\n","relooping and restarting algo !!!!!!!!!!!!!!!","\n")
                    end = True
                except Expection as e:
                    print("main algo major exception",str(e))
                    self.logging("main algo major exception",str(e))
                    error_terminate +=1
                    if error_terminate > 4:
                        terminate = True
                    
            
            if end == True:
                ## close all positions and all orders
                ## cancel all orders
                self.deri_cancel_all()                        
                ### flatten all positions
                #terminate = flatten_positions(passive_sym,aggressive_sym)
                toaster.show_toast("Algo restarting!!!","LOOK AT ME!",duration=1,threaded=True)
                    
            if terminate == True:
                toaster.show_toast("Algo terminated!!!")
                self.logging("\n","ALGO HAS FINISHED")
                print("\n","ALGO HAS FINISHED")
                ## stop websocket thread ##
                self.t.join()
                self.t.stop()
                break  
                
    
        
    def send_test_message(self):
        print("This is a test message")
    def test_class(self):
        date = datetime.datetime.now()
        self.m2_sym, self.back_sym =self.contract_sym(date,roll_buffer=3)
        print(self.m2_sym, self.back_sym,"testing complete!")
        self.send_test_message()
                  
#first = deri_algo(client)
#first.start_algo()


# %%
account_name = "fixed_levels"
algo = deri_algo(account_name)
algo.test_class()

# %% [markdown]
# ### Start websocket

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#algo.start_websocket_thread()

# %% [markdown]
# ### Display websocket data (run after above)

# %%
#algo.print_ws_data()

# %% [markdown]
# ## Time it!

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#algo.set_minute_data() # for asssign prices
#fut_data = algo.deri_ob("BTC-28DEC18",conn="ws").to_dict('records') # for deri_weighted
#times = []
#for i in range(1000):
#    before = datetime.datetime.now()
#    ##############
#    ### write to log file 200 times ## 0.004s for 200 writes! So quick
#    #for i in range(200):
#    #    algo.logging("this is the life aljf saldkjfald",[1,2,3,4,5,6])
#        
#    ### read a file ####
#    #pd.read_csv(arb_algo_path+"\data\deri_data\\"+str(datetime.datetime.now().date()-datetime.timedelta(days=3))+"_btc_perpetual_quote.csv")
#    
#    ### price data ##
#    #algo.var.m1_sym = fut_sym
#    #algo.var.m2_sym = "BTC-PERPETUAL"
#    #algo.var.margin=(0.09/100)
#    #algo.price_data()
#   
#    ## assign prices ####
#    #algo.assign_prices(conn="ws")
#    
#    ## weighted price 
#    #algo.deri_weighted_price(fut_data,100)
#    
#    ### get all deri obs ####
#    #data = algo.deri_ob("all",conn="ws")
#    
#    ### all orders (seems slow) ###
#    #info = algo.deri_all_orders(50)
#     
#    ## open tailed file and convert to data frame(ob)
#    now = datetime.datetime.now().date()
#    opened_file = open(arb_algo_path+"\data\deri_data\\"+str(now)+"_btc_perpetual_quote.csv",'r',newline='')
#    columns = ['instrument', 'level', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'timestamp', 'my_utc_time']
#    lastLines = tl.tail(opened_file,20)
#    
#    #lastLines = [x[0].split(",") for x in [x.split("\n") for x in lastLines]]
#    #ws_file_data = []
#    #for line in lastLines:
#    #    if line == ['']:
#    #        continue
#    #    level = {}
#    #    level["instrument"] = line[0]
#    #    level["level"] = line[1]
#    #    level["bid_price"] = float(line[2])
#    #    level["bid_size"] = int(line[3])
#    #    level["ask_price"] = float(line[4])
#    #    level["ask_size"] = int(line[5])
#    #    level["timestamp"] = pd.to_datetime(line[6])
#    #    level["my_utc_time"] = pd.to_datetime(line[7])
#    #    ws_file_data.append(level)
#    #ws_file_data
#    ws_file_data = pd.read_csv(io.StringIO('\n'.join(lastLines)), header=None)
#    ws_file_data.columns = columns
#            
#    #############
#    #print(i)
#    after = datetime.datetime.now()
#    times.append(after-before)
#clear_output()
#str(np.mean(times)),str(max(times))

# %% [markdown]
# ### clear logging file

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#algo.var.log = "log_Deribit_Algo_Main.txt"
#algo.set_logging_file()
#algo.clear_log()

# %% [markdown]
# ### get settings

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=False)
#algo.var.settings_file = True
#algo.get_settings()
#display(algo.var.settings)

# %% [markdown]
# ### New order

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
##info = {"inst":"BTC-PERPETUAL","side":"buy","order":"market","qty":1,"price":6700} 
#info = {"inst":"BTC-PERPETUAL","side":"buy","order":"limit","qty":1,"price":6400} 
#order, service_error = algo.deri_order(info)
#ids = order["orderId"]
#print(ids,order)

# %% [markdown]
# ### Cancel order

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#ids = ids 
#cancel = algo.deri_cancel(ids)    
#print(cancel)

# %% [markdown]
# ### Cancel all orders

# %%
### not very useful as it doesn't return and order list ##
#
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#cancel = algo.deri_cancel_all()    
#print(cancel)

# %% [markdown]
# ### Amend order

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#amend = {"order_id":ids,"qty":2,"price":6400} 
#order, service_error, order_error = algo.deri_amend_order(amend)
#print(order)

# %% [markdown]
# ### Time to send order, amend order and cancel order

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#
#now = datetime.datetime.now()
###########################
#### first order
### new order
#info = {"inst":"BTC-PERPETUAL","side":"buy","order":"limit","qty":1,"price":3100} 
#order, service_error = algo.deri_order(info)
#ids = order["orderId"]
### amend order
#amend = {"order_id":ids,"qty":2,"price":3111} 
#order, service_error, order_error = algo.deri_amend_order(amend)
### cancel order
#cancel = algo.deri_cancel(ids) 
###########################
#after = datetime.datetime.now()
#
#print(cancel)
#print(after-now)

# %% [markdown]
# ### Get positions

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#algo.deri_positions()

# %% [markdown]
# ### Account and positions
# %% [markdown]
# #### loop

# %%
#account_name = "fixed_levels"
#algo = deri_algo(account_name,testing=True)
#while True:
#    account, positions = algo.deri_account_and_positions()
#    print(account)
#    display(pd.DataFrame(positions).transpose())
#    print(positions)
#    clear_output(wait=True)
#    time.sleep(0.05)

# %% [markdown]
# #### normal

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#account, positions = algo.deri_account_and_positions()
#print(account)
#positions
#
#pd.DataFrame(positions).transpose()

# %% [markdown]
# ### Get active orders

# %%
#### active orders really are only active, once one is filled is dissapears immediatly and therefore does not update fill quantity, open status...etc from here, the same goes for if an order is cancelled ###
## you will not know from polling this. ###
## Even polling with the individual trade ID does not show any cancelled orders ##

#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#algo.deri_active_orders()
##algo.deri_active_orders(inst = "BTC-PERPETUAL")
##algo.deri_active_orders(order_id = "8221687833")

# %% [markdown]
# ### Get order status

# %%
## best for polling to check if orders have been filled, partial, or cancelled. A filled order will get sent here with a "state" of "filled". A partially filled order with still have a state ##
## of "open" and a "qunatity" of 200 and "filledQuantity" of 100 for example. A cancelled order will show up cancelled, even if it has been paritally filled before ###
## Also you can tell if an order has not been submitted via the API, which could come in handy at some point. Only issue is that you can only poll one order at a time. ###
## No other function shows you cancelled orders however, you could only deduce that if you order isnt open, and hasnt been filled, that it has been cancelled, maybe that would mean less polls in the long run? ###

#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#ids = 8829145793
#order, cancelled = algo.deri_order_status(ids)
#print(order)

# %% [markdown]
# ### Get closed orders history

# %%
#account_name = "fixed_levels"
#algo = deri_algo(account_name)
#data = algo.deri_hist_orders(50)
#data = pd.DataFrame(data)
#data["modified"] = pd.to_datetime(data["modified"])
#data = data.sort_values("modified",ascending=False)
#data
##algo.deri_hist_orders(50,order_id=8300558939,testing=True)

# %% [markdown]
# ### Unpickle

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#all_orders = algo.unpickle("deribit_all_orders.pkl")
#all_orders

# %% [markdown]
# ### All orders (indexed by tradeid)

# %%
#all_orders = {'id': 8795236615, 'orderId': 8795236615, 'type': 'limit', 'instrument': 'BTC-28DEC18', 'direction': 'sell', 'price': 3822.0, 'quantity': 1, 'filledQuantity': 0,
#              'avgPrice': 0.0, 'label': '', 'state': 'cancelled', 'created': 1544016596460, 'api': True, 'modified': 1544016596460, 'amount': 10.0, 'filledAmount': 0.0}
#
#new =        {'orderId': 8795105700, 'type': 'limit', 'instrument': 'BTC-PERPETUAL', 'direction': 'buy', 'price': 3854.0, 'label': '', 'amount': 10.0, 'quantity': 1, 'filledQuantity': 0,
#              'filledAmount': 0.0, 'avgPrice': 0.0, 'commission': 0.0, 'created': 1544015793896, 'lastUpdate': 1544015793896, 'state': 'open', 'postOnly': True, 'api': True,
#              'max_show': 1, 'maxShowAmount': 10, 'adv': False}
#
#working =    {'id': 8795020638, 'orderId': 8795020638, 'type': 'limit', 'instrument': 'BTC-PERPETUAL', 'direction': 'buy', 'price': 3852.5, 'quantity': 1, 'filledQuantity': 0, 
#              'avgPrice': 0.0, 'label': '', 'state': 'open', 'created': 1544015283516, 'api': True, 'modified': 1544015454100, 'amount': 10.0, 'filledAmount': 0.0}


# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#
#### put in new order to moniter it
##info = {"inst":"BTC-PERPETUAL","side":"buy","order":"market","qty":1,"price":6700} 
#info = {"inst":"BTC-PERPETUAL","side":"buy","order":"limit","qty":1,"price":3240.50} 
#order, service_error = algo.deri_order(info)
#ids = order["orderId"]
##print(ids,order)
#
#while True:
#    orders = algo.deri_all_orders(20)
#    print("orderId",ids)
#    print(orders.keys())
#    if ids in orders.keys():
#        print("Found it!")
#        tracked_order = orders[ids]
#        if tracked_order["state"] == "open":
#            print("open!!",tracked_order)
#        elif tracked_order["state"] == "filled":
#            print("filled!!",datetime.datetime.now(),tracked_order)
#            break
#        else:
#            print("unknown state",tracked_order)
#    #data_orders = pd.DataFrame(orders).transpose()
#    #data_orders["modified"] = pd.to_datetime(data_orders["modified"],unit="ms")
#    #display(data_orders)
#    #display("open orders",algo.data.open_orders)
#    #total_order_size = sum([abs(o["quantity"]) for o in algo.data.open_orders.values()])
#    #print(total_order_size)
#    clear_output(wait=True)
#    time.sleep(0.05)
##account_and_positions =  algo.read_dict(arb_algo_path+"\python_scripts\deribit_all_orders_"+algo.account_name+".json")
##account_and_positions = pd.DataFrame(account_and_positions).transpose()
##display("raw pickle",account_and_positions)

# %% [markdown]
# ### Single orderbook

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#algo.deri_ob("BTC-28DEC18",conn="rest",df=True)
#algo.deri_ob("BTC-PERPETUAL",conn="rest",df=True)

# %% [markdown]
# ### single orderbook (websocket)

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=True)
#algo.deri_ob("BTC-28DEC18",conn="ws")

# %% [markdown]
# ### All orderbooks (websocket) (more efficient as its one call and gives dicts of results)

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#data = algo.deri_ob("all",conn="ws")
#print(data.keys())
#perp,fut,back = data.values()
#display(perp)
#display(fut)
#display(back)

# %% [markdown]
# ### All orderbooks (rest) (more efficient as its one call and gives dicts of results)

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#data = algo.deri_ob("all",conn="rest")
#print(data.keys())
#perp,fut = data.values()
#perp
#fut

# %% [markdown]
# ### Merge orderbooks (many)

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#fut_data = algo.deri_ob("BTC-28DEC18")
#perp_data = algo.deri_ob("BTC-PERPETUAL")
#obs = [perp_data,fut_data]
#algo.deri_ob_merge(obs)        

# %% [markdown]
# ### Weighted bid/ask price for certain size

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#fut_data = algo.deri_ob("BTC-28DEC18",conn="ws")
##display(fut_data.head())
#algo.deri_weighted_price(fut_data,10000)

# %% [markdown]
# # Data functions
# %% [markdown]
# ### Set minute data

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#algo.set_minute_data()
#data = algo.data.minute_data
#data = data[data["instrument"]=="BTC-PERPETUAL"]
#data.tail()

# %% [markdown]
# ### Get minute data (run after above function)

# %%
##run with above set minute function
#account_name = "trading_hucks"
#algo.get_minute_data()
#print(len(algo.data.minute_data))
#algo.data.minute_data.tail(9)

# %% [markdown]
# ### Assign data/price to class variables for easy access

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#algo.assign_prices(conn="ws")
#
#print(algo.data.d["m1"])
#print(algo.data.d["m2"])
#display(algo.data.d["m1"]["ob"])
#display(algo.data.d["m2"]["ob"])

# %% [markdown]
# ### Arb Price

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=True)
#
#algo.var.margin = round(0.09/100,8)
#algo.var.fee = round((0.05-0.025)/100,8)
#algo.var.m1_sym = "BTC-PERPETUAL"
#algo.var.m2_sym = fut_sym
#
#algo.set_minute_data()
#algo.get_minute_data()
#algo.price_data()
#algo.assign_prices(conn="ws") ## get data 
#
##algo.arb_price(algo.data.d["m1"],algo.data.d["m2"],algo.data.band)
#print(algo.data.d["m1"]["working_levels"])

# %% [markdown]
# ### Price data (pulls prices, works out targets and arb price)

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#algo.start_websocket_thread() ### comment out websocket after first run
#algo.var.mov_avg = 80
#algo.set_minute_data()
#algo.var.m1_sym = "BTC-PERPETUAL"
#algo.var.m2_sym = fut_sym
#algo.var.margin=(0.09/100)
#algo.price_data()
#algo.assign_prices()
#print(algo.data.d["m1"]["working_levels"], algo.data.band["buy"], algo.data.band["sell"])
#print("beyond_mid_avg_rolling %",algo.data.beyond_mid_avg_rolling)

# %% [markdown]
# ## Orders, Fills, Positions: functions
# %% [markdown]
# ### Flatten positions

# %%
#def flatten_positions(passive_sym,aggressive_sym):
#    flatten_qty = {"passive":0,"aggressive":0}
#    flatten_side = {"passive":"","aggressive":""}
#    terminate = False
#    while True:   
#        passive_pos = bitmex_algo.positions(market =passive_sym,conn="websocket",info="simple",ws_data=bitmex_positions)
#        display(passive_pos["buy"])
#        display(passive_pos["sell"])
#        aggressive_pos = bitmex_algo.positions(market =aggressive_sym,conn="websocket",info="simple",ws_data=bitmex_positions)
#        display(aggressive_pos["buy"])
#        display(aggressive_pos["sell"])
#        
#        ## any passive sym positions??? ####
#        if len(passive_pos["buy"]) > 0:
#            flatten_side["passive"] = "Sell"
#            pas_side = "sell"
#            flatten_qty["passive"] = int(abs(passive_pos["buy"]["currentQty"].values[0]))
#            self.logging("\n","need to flatten passive buy positions",flatten_qty["passive"])
#            
#        elif len(passive_pos["sell"]) > 0:
#            flatten_side["passive"] = "Buy"
#            pas_side = "buy"
#            flatten_qty["passive"] = int(abs(passive_pos["sell"]["currentQty"].values[0])) 
#            self.logging("\n","need to flatten passive sell positions",flatten_qty["passive"])
#        
#        ## any aggressive sym positions??? ####
#        if len(aggressive_pos["buy"]) > 0:
#            flatten_side["aggressive"] = "Sell"
#            agres_side = "sell"
#            flatten_qty["aggressive"] = int(abs(aggressive_pos["buy"]["currentQty"].values[0]))
#            self.logging("\n","need to flatten aggressive buy positions",flatten_qty["aggressive"])
#            
#        elif len(aggressive_pos["sell"]) > 0:
#            flatten_side["aggressive"] = "Buy"
#            agres_side = "buy"
#            flatten_qty["aggressive"] = int(abs(aggressive_pos["sell"]["currentQty"].values[0])) 
#            self.logging("\n","need to flatten aggressive sell positions",flatten_qty["aggressive"])
#            
#        ###################
#        # if we have a position in the passive market, then hedge it
#        if flatten_qty["passive"] > 0:
#            passive_flatten = [{'ordType':"Market",'orderQty': flatten_qty["passive"], 'side': flatten_side["passive"], 'symbol': passive_sym}]
#            new_orders, order_error,service_error, post_only, error_list_order_pos_new_orders = bitmex_algo.bulk_new_order(passive_flatten)   
#            check_order_exist(new_orders[pas_side],passive_sym,pas_side+"_passive_flatten","market")
#            self.logging("flattening passive")
#            
#        # if we have a position in the aggressive market, then hedge it            
#        if flatten_qty["aggressive"] > 0:
#            aggressive_flatten = [{'ordType':"Market",'orderQty': flatten_qty["aggressive"], 'side': flatten_side["aggressive"], 'symbol': aggressive_sym}]
#            new_orders, order_error,service_error, post_only, error_list_order_pos_new_orders = bitmex_algo.bulk_new_order(aggressive_flatten)   
#            check_order_exist(new_orders[agres_side],aggressive_sym,agres_side+"_aggressive_flatten","market")
#            self.logging("flattening aggressive")
#        
#        time.sleep(0.2)
#        passive_pos = bitmex_algo.positions(market =passive_sym,conn="websocket",info="simple",ws_data=bitmex_positions)
#        display(passive_pos["buy"])
#        display(passive_pos["sell"])
#        aggressive_pos = bitmex_algo.positions(market =aggressive_sym,conn="websocket",info="simple",ws_data=bitmex_positions)
#        display(aggressive_pos["buy"])
#        display(aggressive_pos["sell"])
#        
#        if sum([len(passive_pos["buy"]),len(passive_pos["sell"]),len(aggressive_pos["buy"]),len(aggressive_pos["sell"])]) == 0:
#            self.logging("\n","positions and orders have been flattened :)")
#            terminate = True
#            return terminate
#        else:
#            self.logging("\n","THERE STILL SEEMS TO BE POSITIONS IN THE MARKET, TRYING TO FLATTEN AGAIN","\n")
#            continue
#            
    

# %% [markdown]
# ### Check for fills

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=True)
#
### new order ####
#print("---- new order -----")
##info = {"inst":"BTC-PERPETUAL","side":"buy","order":"market","qty":1,"price":6700} 
#info = {"inst":"BTC-PERPETUAL","side":"buy","order":"limit","qty":1,"price":6400} 
#order, service_error = algo.deri_order(info)
#ids = order["orderId"]
#print(ids,order)
#
### assign to a known order
#algo.data.d["m1"]["working_orders"]["buy"] = order
#algo.data.d["m1"]["working_orders_info"]["buy"]["quote_qty_left"] = algo.var.size
#algo.data.d["m1"]["working_orders_info"]["buy"]["hedge_qty"] = 0
#
### bring in order data ##
#all_orders = algo.deri_all_orders(10)
#
#### check for fills ####
#print()
#print("---- check order with orderID -----")
#order, hedge, hedge_qty, quote_qty_left, cancelled = algo.check_for_fills(algo.data.d["m1"]["working_orders_info"]["buy"],ids,all_orders,msg="")
#print(order)
#print(hedge, hedge_qty, quote_qty_left, cancelled)
#print()
#print("---- check order by passing message -----")
#order, hedge, hedge_qty, quote_qty_left, cancelled = algo.check_for_fills(algo.data.d["m1"]["working_orders_info"]["buy"],ids,all_orders="",msg=order)
#print(order)
#print(hedge, hedge_qty, quote_qty_left, cancelled)

# %% [markdown]
# ## Trading Functions
# %% [markdown]
# ### List known working orders algo

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name)
#algo.data.d["m1"]["working_orders"] = {"buy":"bla bla bla","sell":{}}
#algo.list_known_working_orders()

# %% [markdown]
# ### Remove duplicates from any order action

# %%
##### run the above list know wokring orders algo to populate info  
#info = [(algo.data.d["m1"],"buy")]
#algo.remove_order_action_dups(info)

# %% [markdown]
# ### Master check for fills

# %%
### new order ####
#def new_order_test():
#    #info = {"inst":"BTC-PERPETUAL","side":"buy","order":"market","qty":1,"price":6700} 
#    order_info = {"inst":"BTC-PERPETUAL","side":"buy","order":"limit","qty":1,"price":6200} 
#    order, service_error = algo.deri_order(order_info)
#    ids = order["orderId"]
#    print(ids,order)
#    ## assign to a known order
#    algo.data.d["m1"]["working_orders"]["buy"] = order
#    algo.data.d["m1"]["working_orders_info"]["buy"]["quote_qty_left"] = algo.var.size
#    algo.data.d["m1"]["working_orders_info"]["buy"]["hedge_qty"] = 0
#    print()
#
### custom function for this test
#def print_info_test():
#    print("send_hedge:",algo.stat.order_action["send_hedge"])
#    print("new_quote:",algo.stat.order_action["new_quote"])
#    print("working_orders:",algo.data.d["m1"]["working_orders"])
#    print("working_orders_info:",algo.data.d["m1"]["working_orders_info"])
#    print()
#  
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=False)
#### master check for fills ####
#
#print("---- new order -----")
#new_order_test()
#print()
#
#print("---- check order with message (no poll) -----")
#info = [("m1","buy",order)]
#algo.master_check_for_fills(info)
#print_info_test()
#
#print("---- check order with poll all orders -----")
#info = [("m1","buy","")]
#algo.master_check_for_fills(info,all_orders=True)
#print_info_test()
#
#print("---- cancel order -----")
#ids = algo.data.d["m1"]["working_orders"]["buy"]["orderId"]
#cancel, hedge = algo.deri_cancel(ids) 
#print(cancel)
#print()
#
#print("---- check cancelled order with message (no poll) -----")
#info = [("m1","buy",cancel)]
#algo.master_check_for_fills(info)
#print_info_test()
#
#print("---- new order -----")
#new_order_test()
#
#print("---- cancel order -----")
#ids = algo.data.d["m1"]["working_orders"]["buy"]["orderId"]
#cancel, hedge = algo.deri_cancel(ids) 
#print(cancel)
#print()
#
#print("---- check cancelled order with poll all orders -----")
#info = [("m1","buy","")]
#algo.master_check_for_fills(info,all_orders=True)
#print_info_test()

# %% [markdown]
# ### New quotes async

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=True)
#algo.reset_var_data_stat()
#algo.data.d["m1"]["working_orders_info"]["buy"]["quote_qty_left"] = algo.var.size
#algo.data.d["m1"]["working_orders_info"]["buy"]["hedge_qty"] = 0
#algo.data.d["m1"]["working_levels"]["buy"] = 6200
#algo.stat.order_action["new_quote"] = [("m1","buy")]
#
#algo.new_quotes()

# %% [markdown]
# ### Cancel quotes async

# %%
### run new quotes function above! ####
#algo.stat.order_action["cancel_quote"] = [("m1","buy")]
### DONT check fills, and dont call master_check_for_fills ##
###algo.cancel_quotes(all_known_orders=True,check_fills=False)
### Rerun again and check for check fills, and call master_check_for_fills ##
#algo.cancel_quotes(all_known_orders=True,check_fills=True)

# %% [markdown]
# ### Amend orders async

# %%
#### run new quotes function above! ####
#algo.stat.order_action["amend_quote"] = [("m1","buy")]
#algo.data.d["m1"]["working_orders_info"]["buy"]["quote_qty_left"] = 9
#algo.amend_quotes()

# %% [markdown]
# ### Setup hedge orders

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=True)
#algo.reset_var_data_stat()
#algo.setup_hedge_orders("m1","buy")
#display("send_hedge:",algo.stat.order_action["send_hedge"])

# %% [markdown]
# ### Send hedges async

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=True)
#algo.reset_var_data_stat()
#algo.price_data()
#algo.assign_prices()
#side = "buy"
#algo.data.d["m1"]["working_orders"][side]["price"] = algo.data.d["m1"]["bid_price"]
#algo.data.d["m1"]["working_orders"][side]["quantity"] = 1
#algo.data.d["m1"]["working_orders_info"][side]["hedge_qty"] = 1
#algo.setup_hedge_orders("m1",side)
#print("send_hedge:",len(algo.stat.order_action["send_hedge"]))
#
#algo.send_hedges()

# %% [markdown]
# ### Dont quote function

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=True)
#algo.dont_quote(best_bid_offered=True)

# %% [markdown]
# ### Re-quote and working levels 

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=True)
#algo.stat.quote["m1"] = {"buy":True, "sell":True}
#algo.stat.quote["m2"] = {"buy":True, "sell":True}
#algo.re_quote_and_working_levels()
#print(algo.stat.order_action["new_quote"])

# %% [markdown]
# ## auto position calc

# %%
#account_name = "fixed_levels" #"trading_hucks"
#fut_sym, back_sym = contract_sym(date,roll_buffer=3)
#algo = deri_algo(account_name,testing=True)
#algo.set_minute_data()
#algo.var.m1_sym = fut_sym
#algo.var.m2_sym = back_sym
#algo.var.margin=(0.09/100)
#algo.price_data()
#algo.assign_prices()
#inv_count, side, avg_spread_entry  = algo.auto_positions()
#print("inv_count:",inv_count)
#print("side:",side)
#print("avg_spread_entry:",avg_spread_entry)

# %% [markdown]
# ## reset variables on algo reset or start

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=True)
#algo.reset_data_stat()

# %% [markdown]
# ## Main Algo

# %%
#account_name = "trading_hucks"
#algo = deri_algo(account_name,testing=True)
#
### Important vars ##
#algo.var.m1_sym = "BTC-PERPETUAL"
#algo.var.m2_sym = fut_sym
#algo.var.m1_quote = {"buy":True, "sell":True}
#algo.var.m2_quote = {"buy":True, "sell":True}
#algo.var.pay_up_ticks = 2000
##algo.var.account_max_lots = 1000
#algo.var.log = "log_Deribit_Algo_testing.txt"
#
############################################
#trading_type = "algo" #"algo" #"manual"
############################################
#
#if trading_type == "manual":
#    algo.var.manual = True
#    ######## Variables to change #############
#    algo.var.settings_file = True
#    #########################################  
#    if algo.var.settings_file == False:
#        algo.var.size = 1
#        algo.var.sell_price = 110
#        algo.var.buy_price = 90
#        quote_m1 = True
#        quote_m2 = True
#        algo.var.sell_reload = 1
#        algo.var.buy_reload = 1
#        #########################################
#        algo.var.m1_quote = {"buy":quote_m1, "sell":quote_m1}
#        algo.var.m2_quote = {"buy":quote_m2, "sell":quote_m2}
#        if algo.var.sell_price == "":
#            algo.var.m1_quote = {"buy":quote_m1, "sell":False}
#            algo.var.m2_quote = {"buy":False, "sell":quote_m2}
#        if algo.var.buy_price == "":
#            algo.var.m1_quote = {"buy":False, "sell":quote_m1}
#            algo.var.m2_quote = {"buy":quote_m2, "sell":False}
#        
#    
#elif trading_type == "algo":
#    algo.var.abs_funding_filter = 0.01/100 #0.0001 -0.0006
#    algo.var.funding_8h_mov_avg = 5
#    algo.var.mov_avg = 200
#    algo.var.margin = 0.14/100 
#    algo.var.size = 5 
#    algo.var.max_inv = 15 
#    ####
#    algo.var.fixed_target = False
#    algo.var.fixed_target_size = 1
#    ####
#    algo.var.boll = 2
#    algo.var.taker_fee = np.mean([0.05,0.075])/100
#    algo.var.maker_fee = np.mean([-0.025,-0.02])/100
#    algo.var.fee = round(algo.var.taker_fee + algo.var.maker_fee,8)
#    algo.var.slippage = 0.5
#    ####
#    algo.var.max_buy_price = 1000
#    algo.var.min_sell_price = -1000
#    algo.var.max_m1_bid_ask_spread = 50
#    algo.var.max_m2_bid_ask_spread = 100
#
#algo.data = data_variables(algo.var)
#
#algo.main_algo()

