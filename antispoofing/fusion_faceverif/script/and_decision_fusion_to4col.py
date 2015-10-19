#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Tue Feb  4 11:50:30 CET 2014

"""
This script performs AND decision level fusion of face verification (FV) and anti-spoofing (AS system and writes down a 4-column score file with the scores. The fusion is done in the following way. 

Development set: 
  - if the sample passes the AS system, it is added to the dev 4-col file with its FV score. If not, there are three options: 1. it is ommited in the dev 4-col file (not written at all); 2. it is written in the dev 4-col file with the lowest score minus 1; 3. it is written in the dev 4-col file with its FV score. The required option can be selected as command line argument
  
Test set:
  - if the sample passes the AS system, it is added to the dev 4-col file with its FV score. If not, it is added to the dev 4-col file with the lowest dev score minus 1.

NOTE: While the script can receive more face verification and anti-spoofing systems as arguments (this is true also for the thresholds), the functionality to fuse more then one face verificaiton and one anti-spooing system is not supported yet.

"""

import os, sys
import argparse
import bob
import numpy

import antispoofing

from antispoofing.utils.db import *
from antispoofing.utils.ml import *
from antispoofing.utils.helpers import *
from antispoofing.fusion_faceverif.helpers import *

import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as mpl

def ensure_dir(f):
  d = os.path.dirname(f)
  if not os.path.exists(d):
    os.makedirs(d)

def main():

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-s', '--fv-scores-dir', type=str, dest='fv_scoresdir', default='', help='Base directory containing the scores of one or more face verification algorithms (without the protocol dir)', nargs='+') #nargs='+'

  parser.add_argument('-a', '--as-scores-dir', type=str, dest='as_scoresdir', default='', help='Base directory containing the scores of one or more antispoofing algorithms', nargs='+') #nargs='+'

  parser.add_argument('--at', '--as-threshold', type=float, dest='as_threshold', default=None, help='The anti-spoofing threshold', nargs='+') #nargs='+'
  
  parser.add_argument('--sp', '--save_params', action='store_true', dest='save_params', default=False, help='Save the decision thresholds in the outputdir for future use')

  parser.add_argument('--do', '--dev-out-option', dest='dev_out_option', type=int, default=3, help='The option for the output of the development file', choices=(1,2,3))

  parser.add_argument('-o', '--outdir', metavar='DIR', type=str, dest='outdir', default='tmp', help='Directory where the resulting 4-col files will be written')

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')
 
  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()
 
  #######################
  # Loading the database objects
  #######################
  database = args.cls(args)
  
  if len(args.as_threshold) != len(args.as_scoresdir):
    raise ValueError("Thresholds must be specified for all the input score sets\n")
  else:
    as_thr = args.as_threshold[0] # multiple thresholds are still not supported

  # Read faceverif and antispoofing scores for all samples
  devel_scores, devel_labels = gather_fvas_scores(database, 'devel', args.fv_scoresdir, args.as_scoresdir, binary_labels=False, fv_protocol='both', normalize=False)
  test_scores, test_labels = gather_fvas_scores(database, 'test', args.fv_scoresdir, args.as_scoresdir, binary_labels=False, fv_protocol='both', normalize=False)
  
  # Separate the scores of valid users, impostors and spoofing attacks
  valid_devel_fv_scores = devel_scores[devel_labels == 1,0];   valid_devel_as_scores = devel_scores[devel_labels == 1,1]
  valid_test_fv_scores = test_scores[test_labels == 1,0];   valid_test_as_scores = test_scores[test_labels == 1,1]
  
  impostors_devel_fv_scores = devel_scores[devel_labels == 0,0];   impostors_devel_as_scores = devel_scores[devel_labels == 0,1]
  impostors_test_fv_scores = test_scores[test_labels == 0,0];   impostors_test_as_scores = test_scores[test_labels == 0,1]
  
  spoof_devel_fv_scores = devel_scores[devel_labels == -1,0];   spoof_devel_as_scores = devel_scores[devel_labels == -1,1]
  spoof_test_fv_scores = test_scores[test_labels == -1,0];   spoof_test_as_scores = test_scores[test_labels == -1,1]
  
  # Determine minimum fv score of dev set and subtract -1
  min_fv_dev = min(devel_scores[:,0]) - 1

  # Reorganize score according to the selected option for dev score output
  if args.dev_out_option == 1:
    valid_devel_fv_scores = valid_devel_fv_scores[valid_devel_as_scores >= as_thr]
    impostors_devel_fv_scores = impostors_devel_fv_scores[impostors_devel_as_scores >= as_thr]
    spoof_devel_fv_scores = impostors_devel_fv_scores[spoof_devel_as_scores >= as_thr]
  elif args.dev_out_option == 2:
    valid_devel_fv_scores[valid_devel_as_scores < as_thr] = min_fv_dev
    impostors_devel_fv_scores[impostors_devel_as_scores < as_thr] = min_fv_dev
    spoof_devel_fv_scores[spoof_devel_as_scores < as_thr] = min_fv_dev
  #else: if this option is 3, nothing is changed in the valid_devel_fv_scores
  
  valid_test_fv_scores[valid_test_as_scores < as_thr] = min_fv_dev
  impostors_test_fv_scores[impostors_test_as_scores < as_thr] = min_fv_dev  
  spoof_test_fv_scores[spoof_test_as_scores < as_thr] = min_fv_dev    
  
  # Organize the lines that need to be written in each file    
  valid_devel_lines = '\n'.join(["x x foo %f" % i for i in valid_devel_fv_scores]) + '\n'
  impostors_devel_lines = '\n'.join(["x y foo %f" % i for i in impostors_devel_fv_scores]) + '\n'
  spoof_devel_lines = '\n'.join(["x y foo %f" % i for i in spoof_devel_fv_scores]) + '\n'
  
  valid_test_lines = '\n'.join(["x x foo %f" % i for i in valid_test_fv_scores]) + '\n'
  impostors_test_lines = '\n'.join(["x y foo %f" % i for i in impostors_test_fv_scores]) + '\n'
  spoof_test_lines = '\n'.join(["x y foo %f" % i for i in spoof_test_fv_scores]) + '\n'
  
  # First write the 4-col for licit protocol
  ensure_dir(os.path.join(args.outdir, '10_licit', 'scores-dev'))
  f_licit_dev = open(os.path.join(args.outdir, '10_licit', 'scores-dev'), 'w')
  f_licit_test = open(os.path.join(args.outdir, '10_licit', 'scores-eval'), 'w')
  f_licit_dev.write(valid_devel_lines); f_licit_dev.write(impostors_devel_lines)
  f_licit_test.write(valid_test_lines); f_licit_test.write(impostors_test_lines)
  f_licit_dev.close()
  f_licit_test.close()
  
  # then write the 4-col for spoof protocol
  ensure_dir(os.path.join(args.outdir, '10_spoof', 'scores-dev'))
  f_spoof_dev = open(os.path.join(args.outdir, '10_spoof', 'scores-dev'), 'w')
  f_spoof_test = open(os.path.join(args.outdir, '10_spoof', 'scores-eval'), 'w')
  f_spoof_dev.write(valid_devel_lines); f_spoof_dev.write(spoof_devel_lines)
  f_spoof_test.write(valid_test_lines); f_spoof_test.write(spoof_test_lines)
  f_spoof_dev.close()
  f_spoof_test.close()
  
  
if __name__ == "__main__":
  main()




