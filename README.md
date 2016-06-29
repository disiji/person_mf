## Python code for the current work on Personalizing Matrix Factorization.
###There are multiple methods available for evaluation. All are accessed through the 'main' code which is run_eval.py.

###In order to see all options run 'python run_eval.py -h' or alternatively from iPython console '%run run_eval.py -h'.

## To reproduce the results from the email run: ##

* For 'no-mixing': 'python run_eval.py -m single'
* For 'global weights': 'python run_eval.py -m mix_global'
* for 'indiv weights': 'python run_eval.py -m mix_indiv'

The other options are: 

* data folders (default is twitter Orange County) 
* debug level (default is INFO)
* number of dimensions for the MFs (default is 30)