#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Mon 14 Sep 15:20:22 CEST 2015

"""
This script performs fusion of the scores of two or more face verification systems, antispoofing counter measure and Image Quality Measures, using GMM modeling. It writes the fused scores into a specified output directory. The original algorithm is published in "Biometric System Design Under Zero and Non-Zero Effort Attacks" - Rattani et al.
"""

import os, sys
import argparse
import numpy
import string

import antispoofing

from antispoofing.fusion_faceverif.helpers import *
from antispoofing.utils.db import *
from antispoofing.utils.helpers import *
from antispoofing.fusion.score_fusion import *

def main():

  #basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
  #FEATURES_DIR = os.path.join(basedir, 'features')

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-s', '--fv-scores-dir', type=str, dest='fv_scoresdir', default='', help='Base directory containing the scores of different face verification algorithms (without the protocol dir and the stdnorm dir; they are added automatically within the program)', nargs='+')

  parser.add_argument('-a', '--as-scores-dir', type=str, dest='as_scoresdir', default='', help='Base directory containing the scores of different antispoofing algorithms', nargs='+')

  parser.add_argument('-q', '--iqa-scores-dir', type=str, dest='iqa_scoresdir', default=None, help='Base directory containing the IQA scores. Can be the IQA features directly, or IQA scores obtained in a trained way')

  parser.add_argument('-p', '--fv-protocol', type=str, dest='faceverif_protocol', default='licit', choices=('licit','spoof','both'), help='The face verification protocol whose score you want to fuse')

  parser.add_argument('-f', '--score-fusion', type=str, dest='fusionalg', default='LLR', choices=('LLR','SUM','LLR_P','SVM','GMM'), help='The score fusion algorithm. LLR - Linear Logistic Regression fusion. LLR-P - Polynomial Logistic regression')

  parser.add_argument('--rc', '--remove_cols', type=int, dest='remove_cols', default=None, help='Indices of columns to remove from the IQA features', nargs='+')

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

  sys.stdout.write("GMM FUSION of ANTISPOOFING, FACEVERIFICATION and IQA systems!!! " + args.faceverif_protocol + " protocol\n")
  sys.stdout.write('---------------------------------------------------------\n')    

  if args.fusionalg == "LLR_P":
    pol_augment=True
  else:
    pol_augment=False

  # read the training data for training and normalization purposes
  sys.stdout.write("Reading training scores...\n")
  all_train_pos, all_train_neg = fusion_iqa_utils.organize_llrtraining_iqa_scores(database, args.fv_scoresdir, args.as_scoresdir, args.iqa_scoresdir, normalize=True, pol_augment=pol_augment, remove_cols=args.remove_cols)
  all_train_scores_nonorm, all_train_labels_nonorm = fusion_iqa_utils.gather_fvas_iqa_scores(database, 'train', args.fv_scoresdir, args.as_scoresdir, args.iqa_scoresdir, normalize=False, pol_augment=pol_augment, remove_cols=args.remove_cols)
  score_norm = ScoreNormalization(all_train_scores_nonorm)
  sys.stdout.write("-------------------------------------\n")

   
  # Process both devel and test data
  
  subset_alias = {'devel':'dev', 'test':'eval'}
  for subset in ('devel', 'test'):  
    sys.stdout.write("Processing " + string.upper(subset) + " scores, " +  args.faceverif_protocol + " protocol...\n")
    all_scores, all_labels = fusion_iqa_utils.gather_fvas_iqa_scores(database, subset, args.fv_scoresdir, args.as_scoresdir, args.iqa_scoresdir, binary_labels=True, fv_protocol=args.faceverif_protocol, normalize=True, score_norm = score_norm, pol_augment=pol_augment, remove_cols=args.remove_cols)

    #import ipdb; ipdb.set_trace()
    if args.fusionalg == "LLR" or args.fusionalg == "LLR_P":
      score_fusion = LLRFusion()
    elif args.fusionalg == "SVM":
      score_fusion = SVMFusion()
    elif args.fusionalg == "GMM":
      score_fusion = GMMFusion()
    else: #SUM
      score_fusion = SUMFusion() 

    
    score_fusion.train(trainer_scores = (all_train_pos, all_train_neg))
    fused = score_fusion(all_scores)
  
    fusion_utils.save_fused_scores(fused, all_labels, args.outputdir, protocol = args.faceverif_protocol, subset = subset_alias[subset]) # save the scores in 4-column format
    sys.stdout.write('---------------------------------------------------------\n')    

if __name__ == "__main__":
  main()




