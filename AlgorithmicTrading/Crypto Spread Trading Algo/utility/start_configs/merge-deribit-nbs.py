#!/usr/bin/env python
# Note, updated version of 
# https://github.com/ipython/ipython-in-depth/blob/master/tools/nbmerge.py
"""
usage:
python nbmerge.py A.ipynb B.ipynb C.ipynb > merged.ipynb
"""

import io
import os
import sys

import nbformat

def merge_notebooks(filenames,end_file_name):
	merged = None
	filenames = filenames.split(",")
	for fname in filenames:
		with io.open(fname, 'r', encoding='utf-8') as f:
			nb = nbformat.read(f, as_version=4)
		if merged is None:
			merged = nb
		else:
            # TODO: add an optional marker between joined notebooks
            # like an horizontal rule, for example, or some other arbitrary
            # (user specified) markdown cell)
			merged.cells.extend(nb.cells)
	if not hasattr(merged.metadata, 'name'):
		merged.metadata.name = ''
	merged.metadata.name += "_merged"
	print(nbformat.write(merged,end_file_name))

if __name__ == '__main__':     
	#notebooks = [r"C:\Users\PC\GS Trading Dropbox\Bitcoin\ArbAlgo\python_scripts\logging.ipynb",
	#r"C:\Users\PC\GS Trading Dropbox\Bitcoin\ArbAlgo\python_scripts\"Deribit_Algo_testing.ipynb",
	#r"C:\Users\PC\GS Trading Dropbox\Bitcoin\ArbAlgo\python_scripts\"Deribit Algo Account Main.ipynb"] 
	#end_file_name = r"C:\Users\PC\GS Trading Dropbox\Bitcoin\ArbAlgo\python_scripts\Deribit_Main_Runner.ipynb"	
	merge_notebooks(sys.argv[1],sys.argv[2])