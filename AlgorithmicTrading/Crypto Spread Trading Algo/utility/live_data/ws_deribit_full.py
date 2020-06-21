
# coding: utf-8

# In[1]:


from IPython.display import clear_output,display,HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


import json
import pickle
import ast
import time
import pandas as pd
import numpy as np
import datetime
import os, sys
import csv
from dateutil.relativedelta import relativedelta, FR

import matplotlib.pyplot as plt


# In[3]:


#%run logging.ipynb
def write(dic_data,speed,file_name,opened_file=False,header=False,rows="single"):
    
    file_exists = os.path.isfile(file_name)
    if rows == "single":
        cols = [key for key,value in dic_data.items()]
    elif rows == "multi":
        cols = [key for key,value in dic_data[0].items()]
    
    if speed=="slow":
        with open(file_name,"a+",newline="") as opened_file:
            writer = csv.DictWriter(opened_file, fieldnames=cols)
            if not file_exists or header==True:
                writer.writeheader()
                
            if rows == "single":
                writer.writerow(dic_data)
            elif rows == "multi":
                writer.writerows(dic_data)
            opened_file.flush()
    
    elif speed=="fast":
        if opened_file == False:
            raise Exception("must pass the actual opened file if using fast method..silly")
        try:
            writer = csv.DictWriter(opened_file, fieldnames=cols)
        except NameError as e:
            opened_file = open(file_name, 'a+',newline='')
            writer = csv.DictWriter(opened_file, fieldnames=cols)           
        if not file_exists or header==True:
            writer.writeheader()
            
        if rows == "single":
            writer.writerow(dic_data)
        elif rows == "multi":
            writer.writerows(dic_data)
        opened_file.flush()


# In[4]:


#%run deribit..somehing/...sadfa
def contract_sym(date,roll_buffer=0): ## roll_buffer is how many days you want to switch to the seconds contract before it really expires
    expiry = [datetime.datetime(date.year,3,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1)),
              datetime.datetime(date.year,6,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1)),
              datetime.datetime(date.year,9,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1)),
              datetime.datetime(date.year,12,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1)),
              datetime.datetime(date.year+1,3,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1)),
              datetime.datetime(date.year+1,6,1,8,0)+relativedelta(day=32)+relativedelta(weekday=FR(-1))]    
    
    contracts = [[expiry[0]-datetime.timedelta(days=roll_buffer),"BTC-"+expiry[0].date().strftime("%d%b").upper()],
                 [expiry[1]-datetime.timedelta(days=roll_buffer),"BTC-"+expiry[1].date().strftime("%d%b").upper()],
                 [expiry[2]-datetime.timedelta(days=roll_buffer),"BTC-"+expiry[2].date().strftime("%d%b").upper()],
                 [expiry[3]-datetime.timedelta(days=roll_buffer),"BTC-"+expiry[3].date().strftime("%d%b").upper()],
                 [expiry[4]-datetime.timedelta(days=roll_buffer),"BTC-"+expiry[4].date().strftime("%d%b").upper()],
                 [expiry[5]-datetime.timedelta(days=roll_buffer),"BTC-"+expiry[5].date().strftime("%d%b").upper()]]
    
    for x in range(len(contracts)):
        if date < contracts[x][0]:
            front_sym = contracts[x][1] + str(contracts[x][0].year)[2:4]
            second_sym = contracts[x+1][1] + str(contracts[x+1][0].year)[2:4]
            break
        else:
            continue
            
    return front_sym,second_sym


# In[5]:


arb_algo_path = os.getcwd().replace("\\python_scripts","")
arb_algo_path


# In[6]:


from deribit_api import RestClient

## spread account
url = "https://www.deribit.com"
key = "5WmL4PVQDkYsr"
secret = "SCUMOH5HNLQ6DBVD7LYJJQDIGHRSHWQO"

client = RestClient(key, secret, url)


# In[7]:


import json, hmac, hashlib, time, requests, base64
from collections import OrderedDict

def generate_signature(key,secret, action, data):
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

generate_signature(key,secret, "/api/v1/private/subscribe", {"instrument":"BTC-PERPETUAL","event":"order_book"})


# In[8]:


# gives you top 20 levels
ob = {'notifications': 
     [{'success': True, 'testnet': True, 'message': 'order_book_event', 'result': 
       {'state': 'open', 'settlementPrice': 6636.89, 'instrument': 'BTC-PERPETUAL', 
        'bids': 
        [{'quantity': 4627, 'amount': 46270.0, 'price': 6707.5, 'cm': 4627, 'cm_amount': 46270.0},
         {'quantity': 1560, 'amount': 15600.0, 'price': 6707.0, 'cm': 6187, 'cm_amount': 61870.0},
         {'quantity': 12283, 'amount': 122830.0, 'price': 6706.5, 'cm': 18470, 'cm_amount': 184700.0},
         {'quantity': 2000, 'amount': 20000.0, 'price': 6706.0, 'cm': 20470, 'cm_amount': 204700.0},
         {'quantity': 7067, 'amount': 70670.0, 'price': 6705.0, 'cm': 27537, 'cm_amount': 275370.0},
         {'quantity': 2, 'amount': 20.0, 'price': 6704.5, 'cm': 27539, 'cm_amount': 275390.0},
         {'quantity': 10000, 'amount': 100000.0, 'price': 6704.0, 'cm': 37539, 'cm_amount': 375390.0},
         {'quantity': 300, 'amount': 3000.0, 'price': 6703.5, 'cm': 37839, 'cm_amount': 378390.0},
         {'quantity': 1869, 'amount': 18690.0, 'price': 6703.0, 'cm': 39708, 'cm_amount': 397080.0},
         {'quantity': 1780, 'amount': 17800.0, 'price': 6702.5, 'cm': 41488, 'cm_amount': 414880.0},
         {'quantity': 600, 'amount': 6000.0, 'price': 6702.0, 'cm': 42088, 'cm_amount': 420880.0},
         {'quantity': 1600, 'amount': 16000.0, 'price': 6701.5, 'cm': 43688, 'cm_amount': 436880.0},
         {'quantity': 1550, 'amount': 15500.0, 'price': 6701.0, 'cm': 45238, 'cm_amount': 452380.0},
         {'quantity': 300, 'amount': 3000.0, 'price': 6700.0, 'cm': 45538, 'cm_amount': 455380.0},
         {'quantity': 1960, 'amount': 19600.0, 'price': 6699.5, 'cm': 47498, 'cm_amount': 474980.0},
         {'quantity': 300, 'amount': 3000.0, 'price': 6699.0, 'cm': 47798, 'cm_amount': 477980.0},
         {'quantity': 800, 'amount': 8000.0, 'price': 6698.5, 'cm': 48598, 'cm_amount': 485980.0},
         {'quantity': 300, 'amount': 3000.0, 'price': 6698.0, 'cm': 48898, 'cm_amount': 488980.0},
         {'quantity': 2379, 'amount': 23790.0, 'price': 6697.5, 'cm': 51277, 'cm_amount': 512770.0},
         {'quantity': 1, 'amount': 10.0, 'price': 6685.0, 'cm': 51278, 'cm_amount': 512780.0}],
        'asks': 
        [{'quantity': 1000, 'amount': 10000.0, 'price': 6708.0, 'cm': 1000, 'cm_amount': 10000.0},
         {'quantity': 77, 'amount': 770.0, 'price': 6708.5, 'cm': 77, 'cm_amount': 10770.0},
         {'quantity': 450, 'amount': 4500.0, 'price': 6709.0, 'cm': 450, 'cm_amount': 15270.0},
         {'quantity': 7300, 'amount': 73000.0, 'price': 6709.5, 'cm': 7300, 'cm_amount': 88270.0},
         {'quantity': 1870, 'amount': 18700.0, 'price': 6710.0, 'cm': 1870, 'cm_amount': 106970.0},
         {'quantity': 4000, 'amount': 40000.0, 'price': 6710.5, 'cm': 4000, 'cm_amount': 146970.0},
         {'quantity': 8940, 'amount': 89400.0, 'price': 6711.0, 'cm': 8940, 'cm_amount': 236370.0},
         {'quantity': 5867, 'amount': 58670.0, 'price': 6711.5, 'cm': 5867, 'cm_amount': 295040.0},
         {'quantity': 300, 'amount': 3000.0, 'price': 6712.0, 'cm': 300, 'cm_amount': 298040.0},
         {'quantity': 300, 'amount': 3000.0, 'price': 6712.5, 'cm': 300, 'cm_amount': 301040.0},
         {'quantity': 2126, 'amount': 21260.0, 'price': 6713.0, 'cm': 2126, 'cm_amount': 322300.0},
         {'quantity': 8600, 'amount': 86000.0, 'price': 6713.5, 'cm': 8600, 'cm_amount': 408300.0},
         {'quantity': 7300, 'amount': 73000.0, 'price': 6714.0, 'cm': 7300, 'cm_amount': 481300.0},
         {'quantity': 400, 'amount': 4000.0, 'price': 6714.5, 'cm': 400, 'cm_amount': 485300.0},
         {'quantity': 600, 'amount': 6000.0, 'price': 6715.0, 'cm': 600, 'cm_amount': 491300.0},
         {'quantity': 300, 'amount': 3000.0, 'price': 6715.5, 'cm': 300, 'cm_amount': 494300.0},
         {'quantity': 2020, 'amount': 20200.0, 'price': 6716.0, 'cm': 2020, 'cm_amount': 514500.0},
         {'quantity': 300, 'amount': 3000.0, 'price': 6716.5, 'cm': 300, 'cm_amount': 517500.0},
         {'quantity': 900, 'amount': 9000.0, 'price': 6717.0, 'cm': 900, 'cm_amount': 526500.0},
         {'quantity': 300, 'amount': 3000.0, 'price': 6717.5, 'cm': 300, 'cm_amount': 529500.0}],
        'tstamp': 1537523736180, 'last': 6700.5, 'low': 6331.0, 'high': 6750.0, 'mark': 6706.74, 'min': 6639.62, 'max': 6773.76}}]
     ,'usOut': 1537523736209587}

message = ob["notifications"][0]["message"]
instrument = ob["notifications"][0]["result"]["instrument"]
bid_price = ob["notifications"][0]["result"]["bids"][0]["price"]
bid_size = ob["notifications"][0]["result"]["bids"][0]["quantity"]
ask_price = ob["notifications"][0]["result"]["asks"][0]["price"]
ask_size = ob["notifications"][0]["result"]["asks"][0]["quantity"]
timestamp = pd.to_datetime(ob["notifications"][0]["result"]["tstamp"],unit="ms")

data = {"message" : ob["notifications"][0]["message"],
        "instrument" : ob["notifications"][0]["result"]["instrument"],
        "bid_price" : ob["notifications"][0]["result"]["bids"][0]["price"],
        "bid_size" : ob["notifications"][0]["result"]["bids"][0]["quantity"],
        "ask_price" : ob["notifications"][0]["result"]["asks"][0]["price"],
        "ask_size" : ob["notifications"][0]["result"]["asks"][0]["quantity"],
        "timestamp" : str(pd.to_datetime(ob["notifications"][0]["result"]["tstamp"],unit="ms"))}
print(data)


# In[9]:



date = datetime.datetime.utcnow()
fut_sym, back_sym = contract_sym(date)
fut_sym, back_sym


# # ws lomond template

# In[10]:


#client.getsummary("BTC-PERPETUAL")


# In[11]:


from lomond import WebSocket
from lomond.persist import persist

def ws():  
    nicknames = {}
    file_names = {}
    headers = {}
    trades = {}
    opened_files = {}
    old_ob = {}
    send_ob = {}
    global deri_ob
    deri_ob = {}
    high = {}
    low = {}
    
    time_now = datetime.datetime.utcnow()
    fut_sym, back_sym = contract_sym(time_now)    
    instruments = ["BTC-PERPETUAL",fut_sym,back_sym]
    short = ["_btc_perpetual_","_btc_front_quarter_","_btc_back_quarter_"]
    channels = ["quote","trade"]  
    directory = arb_algo_path+"\data\deri_data\\"
    m1_directory = arb_algo_path+"\data\deri_1min\\"
                     
    for chan in channels:
        nicknames[chan] = {}
        headers[chan] = {}
        file_names[chan] = {}
        opened_files[chan] = {}
        for idx in range(len(instruments)):
            nicknames[chan][instruments[idx]] = short[idx]+chan
        
    for chan in channels:    
        for inst in instruments:            
            file_names[chan][inst] = directory+str(time_now.date())+nicknames[chan][inst]+".csv"
            if os.path.isfile(file_names[chan][inst]) == False:
                headers[chan][inst] = True
            else:
                headers[chan][inst] = False
            ### open all files, creating any new files names for those that dont exist
            opened_files[chan][inst] = open(file_names[chan][inst],'a+',newline='')
            
    for inst in instruments:
        ### create data stores ###
        #deri_ob[inst] = [] dont create as it creates itself
        old_ob[inst] = {} 
        high[inst] = -np.inf
        low[inst] = np.inf
    spread_high = -np.inf
    spread_low = np.inf
        
    ### files just for 1min quote ###
    file_name_min = m1_directory+"week_"+str(time_now.isocalendar()[1])+"_1min"+".csv"
    headers_min = False
    
    
    save_min = False
    ws_fund = False
    heartbeat1h = False
    old_date = datetime.datetime(2008,1,1)
    ### Start websocket ###
    while True:
        ws_delay = False
        websocket = WebSocket('wss://www.deribit.com/ws/api/v1')
        now_time = datetime.datetime.utcnow()
        for event in persist(websocket):
            
            time_now = datetime.datetime.utcnow()
            ### check if file exisits (either because you run the ws for the first time, or its a new date) ####
            if time_now.date() != old_date:
                for chan in channels:
                    for inst,nickname in nicknames[chan].items():
                        file_names[chan][inst] = directory+str(time_now.date())+nickname+".csv"
                        if os.path.isfile(file_names[chan][inst]) == False:
                            ## only creates a new file if one doenst exist
                            opened_files[chan][inst] = open(file_names[chan][inst],'a+',newline='')
                            headers[chan][inst] = True
                file_name_min = m1_directory+"week_"+str(time_now.isocalendar()[1])+"_1min"+".csv"
                if os.path.isfile(file_name_min) == False:
                    headers_min = True
                
            old_date = time_now.date()
            
            #################################################################################################### 
            
            if event.name == "ready":
                ############ need to code up way to auto subscribe when instruments change ##############
                #fut_sym, back_sym = contract_sym(time_now)
                ################################
                ws_args = {"instrument":instruments,"event":["order_book","trade"]}
                signature = generate_signature(key,secret, "/api/v1/private/subscribe",ws_args)
                print("connecting",datetime.datetime.now(),signature)
                websocket.send_json(id = 5533,action="/api/v1/private/subscribe",arguments= ws_args,sig = signature)                
                ws_args = {"instrument":"BTC-PERPETUAL"}
                signature = generate_signature(key,secret, "/api/v1/public/getsummary",ws_args)  
                ws_funding = websocket.send_json(id = 62,action="/api/v1/public/getsummary",arguments= ws_args,sig = signature)                 
            elif event.name == "text":
                now_time = datetime.datetime.utcnow()
                result = event.json
                #print("actual new websocket object coming through")
    ###############################################################
                ######### funding if requested ####
                if "id" in result:
                    if result["id"] == 62:
                        current_funding = result["result"]["currentFunding"]
                        funding_8h = result["result"]["funding8h"]
                ######################################
                if "notifications" in result:
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

                                write(send_ob[inst],"fast",file_names[chan][inst],opened_files[chan][inst],headers[chan][inst],rows="multi")                         
                                headers[chan][inst]=False
                            
        
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
                            
                            ## write
                            write(min_data,"slow",file_name_min,rows="multi")
                            headers_min = False
                            save_min = False                           
                            
                        if now_time.minute == 0 and now_time.second == 0:
                            heartbeat1h = True
                        elif now_time.minute == 0 and now_time.second > 0 and heartbeat1h == True:
                            print("ws full alive",now_time)
                            heartbeat1h = False
                                
                    if result["notifications"][0]["message"] == "trade_event":
                        inst = result["notifications"][0]["result"][0]["instrument"]
                        chan = "trade"
                        trades[inst] = []                    
                        for trade in result["notifications"][0]["result"]:
                            trades_dict = {}
                            trades_dict["instrument"] = trade["instrument"]
                            trades_dict["qty"] = trade["quantity"]
                            trades_dict["timestamp"] = str(pd.to_datetime(trade["timeStamp"],unit="ms"))
                            trades_dict["my_utc_time"] = str(now_time)
                            trades_dict["price"] = trade["price"]
                            trades_dict["direction"] = trade["direction"] 
                            
                            trades[inst].append(trades_dict)
                            
                        write(trades[inst],"fast",file_names[chan][inst],opened_files[chan][inst],headers[chan][inst],rows="multi")                         
                        headers[chan][inst]=False
                                                    
    ###############################################################               
    
            elif event.name == "pong":
                continue
            elif event.name == "poll": # happens every 5 seconds
                new_fut, new_back = contract_sym(now_time)
                if (fut_sym,back_sym) != (new_fut,new_back):
                    ## break websocket so it can change contracts
                    print("breaking websocket to change contracts")
                    break
                if now_time+datetime.timedelta(seconds=5) < datetime.datetime.utcnow():
                    print("websocket has been stale for 5 seconds","ws_time",now_time,"now",datetime.datetime.utcnow())
                    break
ws()


# ## testing

# In[ ]:


#import threading
#
#t = threading.Thread(target = ws,args = ())
#t.start()
#
#while True:
#    print(deri_ob.get("BTC-PERPETUAL")[0])
#    clear_output(wait=True)
#    time.sleep(0.1)

