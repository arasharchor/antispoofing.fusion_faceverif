#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Wed Mar 27 10:14:57 CET 2013

"""
This script calculates the threshold of an anti-spoofing algorithm on the EER of the devel set, as well as FAR and FRR for the devel and test set.

"""

import os, sys
import argparse
import bob.measure
import numpy

import antispoofing

from antispoofing.utils.db import *
from antispoofing.utils.helpers import *
from antispoofing.fusion_faceverif.helpers import *

import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as mpl


def read_as_data(real_files, attack_files, as_scoresdir):
  '''Reads all the scores for the files given in the lists real_files and attack_files'''

  real_reader = score_reader.ScoreReader(real_files, as_scoresdir)
  attack_reader = score_reader.ScoreReader(attack_files, as_scoresdir)
  return real_reader.getScores(), attack_reader.getScores()

def main():

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('as_scoresdir', type=str, help='Base directory containing the scores an antispoofing algorithms')
 
  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()
 
  #######################
  # Loading the database objects
  #######################
  database = args.cls(args)

  # read spoofing scores to determine anti-spoofing threshold
  real_devel, attack_devel = database.get_devel_data()
  real_test, attack_test = database.get_test_data()
  
  real_devel_as, attack_devel_as = read_as_data(real_devel, attack_devel, args.as_scoresdir) #True
  real_test_as, attack_test_as = read_as_data(real_test, attack_test, args.as_scoresdir) #True

  as_thr = bob.measure.eer_threshold(attack_devel_as.flatten(), real_devel_as.flatten())
  
  # calculate antispoofing system performance
  devel_far_as, devel_frr_as = bob.measure.farfrr(attack_devel_as.flatten(), real_devel_as.flatten(), as_thr)
  test_far_as, test_frr_as = bob.measure.farfrr(attack_test_as.flatten(), real_test_as.flatten(), as_thr)

  sys.stdout.write("AS system results:\n")
  sys.stdout.write("AS threshold: %f\n" % (as_thr))
  sys.stdout.write("Devel: FAR=%.3f, FRR=%.3f, HTER=%.3f\n" % (devel_far_as*100, devel_frr_as*100, devel_far_as*50 + devel_frr_as*50))
  sys.stdout.write("Test: FAR=%.3f, FRR=%.3f, HTER=%.3f\n" % (test_far_as*100, test_frr_as*100, test_far_as*50 + test_frr_as*50))
  sys.stdout.write("----------------------------------------------------------\n")
  
if __name__ == "__main__":
  main()




