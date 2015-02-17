#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Mon Mar 25 18:32:57 CET 2013

"""
This script performs AND decision level fusion of face verification and anti-spoofing system. It determines the decision boundaries of the two systems and reports the error rate (FAR, FRR, HTER, SFAR).

NOTE: While the script can receive more face verification and anti-spoofing systems as arguments (this is true also for the thresholds), the functionality to fuse more then one face verificaiton and one anti-spooing system is not supported yet.

"""

import os, sys
import argparse
import numpy

import antispoofing

from antispoofing.utils.db import *
from antispoofing.utils.ml import *
from antispoofing.utils.helpers import *
from antispoofing.fusion_faceverif.helpers import *

import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as mpl



def main():

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-s', '--fv-scores-dir', type=str, dest='fv_scoresdir', default='', help='Base directory containing the scores of one or more face verification algorithms (without the protocol dir)', nargs='+') #nargs='+'

  parser.add_argument('-a', '--as-scores-dir', type=str, dest='as_scoresdir', default='', help='Base directory containing the scores of one or more antispoofing algorithms', nargs='+') #nargs='+'

  parser.add_argument('--ft', '--fv-threshold', type=float, dest='fv_threshold', default=None, help='The face verification threshold', nargs='+') #nargs='+'

  parser.add_argument('--at', '--as-threshold', type=float, dest='as_threshold', default=None, help='The anti-spoofing threshold', nargs='+') #nargs='+'

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
  
  if len(args.as_threshold) != len(args.as_scoresdir) or len(args.fv_threshold) != len(args.fv_scoresdir): 
    raise ValueError("Thresholds must be specified for all the input score sets\n")
  else:
    as_thr = args.as_threshold[0] # multiple thresholds are still not supported
    fv_thr = args.fv_threshold[0]

  # read faceverif and antispoofing scores for all samples
  devel_scores, devel_labels = gather_fvas_scores(database, 'devel', args.fv_scoresdir, args.as_scoresdir, binary_labels=False, fv_protocol='both', normalize=False)
  test_scores, test_labels = gather_fvas_scores(database, 'test', args.fv_scoresdir, args.as_scoresdir, binary_labels=False, fv_protocol='both', normalize=False)
  
  # separate the scores of valid users, impostors and spoofing attacks
  valid_devel_fv_scores = devel_scores[devel_labels == 1,0];   valid_devel_as_scores = devel_scores[devel_labels == 1,1]
  valid_test_fv_scores = test_scores[test_labels == 1,0];   valid_test_as_scores = test_scores[test_labels == 1,1]
  
  impostors_devel_fv_scores = devel_scores[devel_labels == 0,0];   impostors_devel_as_scores = devel_scores[devel_labels == 0,1]
  impostors_test_fv_scores = test_scores[test_labels == 0,0];   impostors_test_as_scores = test_scores[test_labels == 0,1]
  
  spoof_devel_fv_scores = devel_scores[devel_labels == -1,0];   spoof_devel_as_scores = devel_scores[devel_labels == -1,1]
  spoof_test_fv_scores = test_scores[test_labels == -1,0];   spoof_test_as_scores = test_scores[test_labels == -1,1]
  
  # determine the wrongly classified samples by AND fusion system
  devel_far_ind = [ind for ind in range(len(impostors_devel_fv_scores)) if impostors_devel_fv_scores[ind] > fv_thr and impostors_devel_as_scores[ind] > as_thr];
  devel_frr_ind = [ind for ind in range(len(valid_devel_fv_scores)) if (valid_devel_fv_scores[ind] < fv_thr or valid_devel_as_scores[ind] < as_thr)];
  devel_sfar_ind = [ind for ind in range(len(spoof_devel_fv_scores)) if spoof_devel_fv_scores[ind] > fv_thr and spoof_devel_as_scores[ind] > as_thr];
    
  test_far_ind = [ind for ind in range(len(impostors_test_fv_scores)) if impostors_test_fv_scores[ind] > fv_thr and impostors_test_as_scores[ind] > as_thr];
  test_frr_ind = [ind for ind in range(len(valid_test_fv_scores)) if (valid_test_fv_scores[ind] < fv_thr or valid_test_as_scores[ind] < as_thr)];
  test_sfar_ind = [ind for ind in range(len(spoof_test_fv_scores)) if spoof_test_fv_scores[ind] > fv_thr and spoof_test_as_scores[ind] > as_thr];  
  
  # calculate performance of AND fusion system
  devel_far = len(devel_far_ind) / float(len(impostors_devel_fv_scores)); devel_frr = len(devel_frr_ind) / float(len(valid_devel_fv_scores)); devel_sfar = len(devel_sfar_ind) / float(len(spoof_devel_fv_scores)); 
  test_far = len(test_far_ind) / float(len(impostors_test_fv_scores)); test_frr = len(test_frr_ind) / float(len(valid_test_fv_scores)); test_sfar = len(test_sfar_ind) / float(len(spoof_test_fv_scores)); 

  # print results
  sys.stdout.write("FV threshold: %f, AS threshold: %f\n" % (fv_thr, as_thr))
  sys.stdout.write("----------------------------------------------------------\n")
  sys.stdout.write("AND fused system results:\n")
  sys.stdout.write("Devel: FAR=%.3f, FRR=%.3f, HTER=%.3f, SFAR=%.3f\n" % (devel_far*100, devel_frr*100, devel_far*50 + devel_frr*50, devel_sfar*100))
  sys.stdout.write("Test: FAR=%.3f, FRR=%.3f, HTER=%.3f, SFAR=%.3f\n" % (test_far*100, test_frr*100, test_far*50 + test_frr*50, test_sfar*100))
  
if __name__ == "__main__":
  main()




