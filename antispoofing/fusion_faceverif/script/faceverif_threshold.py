#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Wed Mar 27 10:26:35 CET 2013

"""
This script calculates the threshold of a faceverificaiton algorithm on EER on the devel set and calculates the error rates (FAR, FRR, HTER, SFAR) on the development and test set

"""

import os, sys
import argparse
import bob.measure
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

  parser.add_argument('fv_scoresdir', type=str, help='Base directory containing the scores of a face verification algorithms (without the protocol dir)')
 
  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()
 
  #######################
  # Loading the database objects
  #######################
  database = args.cls(args)
  
  # read faceverif and antispoofing scores for all samples
  devel_scores, devel_labels = gather_fvas_scores(database, 'devel', [args.fv_scoresdir,], as_dirs = None, binary_labels=False, fv_protocol='both', normalize=False)
  test_scores, test_labels = gather_fvas_scores(database, 'test', [args.fv_scoresdir,], as_dirs = None, binary_labels=False, fv_protocol='both', normalize=False)
  
  # separate the scores of valid users, impostors and spoofing attacks
  valid_devel_fv_scores = devel_scores[devel_labels == 1,0];  
  valid_test_fv_scores = test_scores[test_labels == 1,0];  
  
  impostors_devel_fv_scores = devel_scores[devel_labels == 0,0];
  impostors_test_fv_scores = test_scores[test_labels == 0,0];  
  
  spoof_devel_fv_scores = devel_scores[devel_labels == -1,0];  
  spoof_test_fv_scores = test_scores[test_labels == -1,0]; 
  
  # determine face verif threshold
  fv_thr = bob.measure.eer_threshold(impostors_devel_fv_scores, valid_devel_fv_scores)
  #as_thr = bob.measure.eer_threshold(spoof_devel_as_scores, valid_devel_as_scores)   
  
  # calculate faceverif baseline performance
  devel_far_fv, devel_frr_fv = bob.measure.farfrr(impostors_devel_fv_scores, valid_devel_fv_scores, fv_thr)
  devel_sfar_fv, devel_frr_fv = bob.measure.farfrr(spoof_devel_fv_scores, valid_devel_fv_scores, fv_thr)
  
  test_far_fv, test_frr_fv = bob.measure.farfrr(impostors_test_fv_scores, valid_test_fv_scores, fv_thr)
  test_sfar_fv, test_frr_fv = bob.measure.farfrr(spoof_test_fv_scores, valid_test_fv_scores, fv_thr)
  
  # print results
  sys.stdout.write("FV threshold: %f\n" % fv_thr)
  sys.stdout.write("FV system results:\n")
  sys.stdout.write("Devel: FAR=%.3f, FRR=%.3f, HTER=%.3f, SFAR=%.3f\n" % (devel_far_fv*100, devel_frr_fv*100, devel_far_fv*50 + devel_frr_fv*50, devel_sfar_fv*100))
  sys.stdout.write("Test: FAR=%.3f, FRR=%.3f, HTER=%.3f, SFAR=%.3f\n" % (test_far_fv*100, test_frr_fv*100, test_far_fv*50 + test_frr_fv*50, test_sfar_fv*100))
  sys.stdout.write("----------------------------------------------------------\n")
  
if __name__ == "__main__":
  main()




