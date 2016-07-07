"""
Running script for the evaluation.
It includes all methods, the args will control which one is running.

Run %run run_eval.py -h for full description of the options.

Author: Moshe Lichman
"""
from __future__ import division
import argparse
import logging

from utils import log_utils as log
from utils import file_utils as fu

from evaluation import method_factory as mf
from smoothing import method_factory as mf_smoothing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-area', type=str, help='Defines where to load files from',
                        default='tw_oc', choices=fu.valid_areas())
    parser.add_argument('-dim', type=int, help='Number of dimensions for MFs', default=30)
    parser.add_argument('-m', type=str, help='Smoothing method',
                        default='non_smoothing', choices=list(mf_smoothing.keys()))
    parser.add_argument('-v', type=bool, help='Debug logging level', default=False)

    args = parser.parse_args()

    if args.v:
        log._ch.setLevel(logging.DEBUG)

    train, val, test = fu.load_data(args.area)
    #eval_method = mf[args.m]()

    #results = eval_method.evaluate(train, val, test, args.dim, args.area)
    smooth_method = mf_smoothing[args.m]()
    results = smooth_method.evaluate(train, val, test, args.dim, args.area)