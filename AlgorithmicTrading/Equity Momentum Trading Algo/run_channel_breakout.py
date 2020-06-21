import os
import sys
import ast
import time
import subprocess

args = sys.argv[1:]
num_processes = int(args[0]) if len(args) >= 1 else 18

batch_processes = []
for stop in [0.3,0.4,0.5,0.6,0.7]:
    for target in [3,4,5,6,7,8,9,10,12,15,20]:
        for lookback in [120]: #120
            process = subprocess.Popen(['python',
                                        '-u',
                                        'channel_breakout.py',
                                        str(lookback),
                                        str(stop),
                                        str(target),
                                        ])
            batch_processes.append(process)
            # keep the number of processes running at a consistent number
            while len(batch_processes) == num_processes:
                time.sleep(5)
                # process.poll() will return None if it's still running
                batch_processes = [process for process in batch_processes if process.poll() is None]