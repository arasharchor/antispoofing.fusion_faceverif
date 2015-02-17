#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Tue Nov 20 18:11:15 CET 2012

"""
This script performs fusion of the scores of two or more face verification systems and antispoofing counter measure. It writes the fused scores into a specified output directory

"""

import os, sys
import argparse
import numpy
import string

import antispoofing

from antispoofing.fusion_faceverif.helpers import *
from antispoofing.utils.db import *
from antispoofing.utils.ml import *
from antispoofing.utils.helpers import *
from antispoofing.fusion.score_fusion import *

import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as mpl

def main():

  #basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
  #FEATURES_DIR = os.path.join(basedir, 'features')

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-s', '--fv-scores-dir', type=str, dest='fv_scoresdir', default='', help='Base directory containing the scores of different face verification algorithms (without the protocol dir and the stdnorm dir; they are added automatically within the program)', nargs='+')

  parser.add_argument('-a', '--as-scores-dir', type=str, dest='as_scoresdir', default='', help='Base directory containing the scores of different antispoofing algorithms', nargs='+')

  parser.add_argument('-p', '--fv-protocol', type=str, dest='faceverif_protocol', default='licit', choices=('licit','spoof'), help='The face verification protocol whose score you want to fuse')
  
  parser.add_argument('-f', '--score-fusion', type=str, dest='fusionalg', default='LLR', choices=('LLR','SUM','LLR_P'), help='The score fusion algorithm. LLR - Linear Logistic Regression fusion. LLR-P - Polynomial Logistic regression')

  parser.add_argument('-o', '--output-dir', type=str, dest='outputdir', default='', help='Base directory that will be used to save the fused scores.')
  
  parser.add_argument('--sp', '--save_params', action='store_true', dest='save_params', default=False, help='Save the LLR machine and normalization parameters in the outputdir for future use')

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

  sys.stdout.write(args.fusionalg + " FUSION of ANTISPOOFING and FACEVERIFICATION systems!!! " + args.faceverif_protocol + " protocol\n")
  sys.stdout.write('---------------------------------------------------------\n')    

  if args.fusionalg == "LLR_P":
    pol_augment=True
  else:
    pol_augment=False

  # read the training data for LLR training and normalization purposes
  sys.stdout.write("Reading training scores...\n")
  all_train_pos, all_train_neg = fusion_utils.organize_llrtraining_scores(database, args.fv_scoresdir, args.as_scoresdir, normalize=True, pol_augment=pol_augment)
  all_train_scores_nonorm, all_train_labels_nonorm = fusion_utils.gather_fvas_scores(database, 'train', args.fv_scoresdir, args.as_scoresdir, normalize=False, pol_augment=pol_augment)
  score_norm = ScoreNormalization(all_train_scores_nonorm)
  sys.stdout.write("-------------------------------------\n")

   
  # Process both devel and test data
  
  subset_alias = {'devel':'dev', 'test':'eval'}
  for subset in ('devel', 'test'):  
    sys.stdout.write("Processing " + string.upper(subset) + " scores, " +  args.faceverif_protocol + " protocol...\n")
    all_scores, all_labels = fusion_utils.gather_fvas_scores(database, subset, args.fv_scoresdir, args.as_scoresdir, binary_labels=True, fv_protocol=args.faceverif_protocol, normalize=True, score_norm = score_norm, pol_augment=pol_augment)

    if args.fusionalg == "LLR" or args.fusionalg == "LLR_P":
      score_fusion = LLRFusion()
    else:
      score_fusion = SUMFusion()  
    
    score_fusion.train(trainer_scores = (all_train_pos, all_train_neg))
    fused = score_fusion(all_scores)
  
    if args.save_params == True:
      if args.fusionalg == "LLR" or args.fusionalg == "LLR_P":
        fusion_utils.save_fusion_machine(score_fusion.get_machine(), args.outputdir, protocol = args.faceverif_protocol, subset = subset_alias[subset])
        fusion_utils.save_norm_params(ScoreNormalization(all_train_scores_nonorm), args.outputdir, protocol = args.faceverif_protocol, subset = subset_alias[subset])
      else:
        score_fusion = SUMFusion() 
        fusion_utils.save_norm_params(score_norm, args.outputdir, protocol = args.faceverif_protocol, subset = subset_alias[subset])    

    fusion_utils.save_fused_scores(fused, all_labels, args.outputdir, protocol = args.faceverif_protocol, subset = subset_alias[subset]) # save the scores in 4-column format
    sys.stdout.write('---------------------------------------------------------\n')    

if __name__ == "__main__":
  main()




