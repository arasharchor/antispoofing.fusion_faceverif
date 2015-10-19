#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Mon 14 Sep 18:09:56 CEST 2015

import os, sys
import argparse
import bob.io.base
import numpy

import antispoofing

from antispoofing.utils.db import *
from antispoofing.utils.ml import *
from antispoofing.utils.helpers import *
from antispoofing.fusion.readers import *

from .fusion_utils import *

  
def organize_llrtraining_iqa_scores(database, fv_dirs, as_dirs, iqa_dirs = None, normalize=True, pol_augment=False, remove_cols=None):
  all_scores, all_labels = gather_train_fvas_iqa_scores(database, fv_dirs, as_dirs, iqa_dirs, binary_labels=True, normalize=normalize, pol_augment=pol_augment, remove_cols=remove_cols)
  all_pos = all_scores[all_labels == 1,:]
  all_neg = all_scores[all_labels == 0,:]
  return all_pos, all_neg
  
def simple_feat_gather(indir, objects):
  """Collects the features from all objects together"""
  dataset = None 
  for obj in objects:
    filename = os.path.expanduser(obj.make_path(indir, '.hdf5'))
    fvs = bob.io.base.load(filename)
    if dataset is None:
      dataset = fvs
    else:
      dataset = numpy.append(dataset, fvs, axis = 0)
  return dataset     

def gather_train_fvas_iqa_scores(database, fv_dirs, as_dirs, iqa_dirs=None, binary_labels=True, fv_protocol='both', normalize=True, pol_augment=False, remove_cols=None):
  """Populates a numpy.ndarray with the nor normalized training scores of face verification and anti-spoofing algorithm(s) and a numpy.array with their corresponding labels. Each column of the arrays correspond to a face verification / anti-spoofing algorithm (with face verification algorithms coming first). The returning result canbe used for normalization purposes
  
  @param database The database (replay)
  @param fv_dirs List of directories of the scores of face verification algorithms
  @param as_dirs List of directories of the scores of anti-spoofing algorithms
  @param iqa_dirs Directory containing the IQA scores or features
  @param binary_labels If True, binary labels will be returned: both impostors and spoofing attacks will be labeled with 0, while real accesses with 1. If False, ternary labels will be returned: impostors: 0, real accesses: 1, spoofing attacks: -1
  @param fv_protocol Specifies the face verification protocol for the returned scores. Can be 'licit', 'spoof' or 'both'
  @param normalize if True, the returned data will be normalized with regards to the training set
  @param pol_augment If True, the data will be polinomially augmented (columns with quadratic values will be added to the data
  @param remove_cols Columns to be removed from the IQA features
  """

  real, attack = database.get_train_data()

  clients = list(set(["client%03d" % x.get_client_id() for x in real]))
  
  sys.stdout.write('Organizing faceverif and antispoofing scores\n')
  
  # read the IQA scores
  if iqa_dirs is not None:
    real_iqa = simple_feat_gather(iqa_dirs, real)
    attack_iqa = simple_feat_gather(iqa_dirs, attack)

    if remove_cols is not None:
      cols_to_keep = list(set(range(real_iqa.shape[1])) - set(remove_cols))
      real_iqa = real_iqa[:,cols_to_keep]
      attack_iqa = attack_iqa[:,cols_to_keep]

  # create empty array for all scores
  all_scores = numpy.ndarray((0, len(fv_dirs) + len(as_dirs) + real_iqa.shape[1]), 'float') if iqa_dirs is not None else numpy.ndarray((0, len(fv_dirs) + len(as_dirs)), 'float');
  all_labels = numpy.array([], 'int');
  
  # reading the anti-spoofing data
  real_scorereader_as = ScoreFusionReader(real, as_dirs)
  attack_scorereader_as = ScoreFusionReader(attack, as_dirs)

  real_as = real_scorereader_as.getConcatenetedScores(onlyValidScores=False)
  attack_as = attack_scorereader_as.getConcatenetedScores(onlyValidScores=False)

  # reading the face verification data
  if fv_protocol == 'licit' or fv_protocol == 'both':
    sys.stdout.write('Processing face verif scores: LICIT protocol\n')
    #dir_precise = os.path.join('10_licit', 'nonorm')
    dir_precise = os.path.join('licit')
    for cl in clients:
      sys.stdout.write("Processing [%s/%d] in training set\n" % (cl, len(clients)))
      # creating the scores readers from different face verification algorithms
      # joining the scores for face verfication with the anti-spoofing scores. The face verifications scores dirs need to have the client labels
      real_scorereader_fv = ScoreFusionReader(real, [os.path.join(sd, dir_precise, cl) for sd in fv_dirs])
      real_fv = real_scorereader_fv.getConcatenetedScores(onlyValidScores=False) # raw scores as numpy.array (the scores should be already normalized in the input files)
      labels = get_labels(os.path.join(fv_dirs[0], dir_precise), real, protocol='licit', client_id=cl, onlyValidScores=False, binary_labels=binary_labels) # labels for the face verification queries. The samples in the 'licit' protocol are real accesses => their label depends only on the identities
      scores = numpy.append(real_fv, numpy.append(real_as, real_iqa, axis=1), axis=1) if iqa_dirs is not None else numpy.append(real_fv, real_as, axis=1)
      all_labels = numpy.append(all_labels, labels)
      all_scores = numpy.append(all_scores, scores, axis=0)
  
  if fv_protocol == 'spoof' or fv_protocol == 'both':
    sys.stdout.write('Processing face verif scores: SPOOF protocol\n')
    #dir_precise = os.path.join('10_spoof', 'nonorm')
    dir_precise = os.path.join('spoof')
    real_scorereader_fv = ScoreFusionReader(real, [os.path.join(sd, dir_precise) for sd in fv_dirs])
    attack_scorereader_fv = ScoreFusionReader(attack, [os.path.join(sd, dir_precise) for sd in fv_dirs])  

    # get raw scores as numpy.array (the scores should be already normalized in the input files)
    real_fv = real_scorereader_fv.getConcatenetedScores(onlyValidScores=False)
    attack_fv = attack_scorereader_fv.getConcatenetedScores(onlyValidScores=False) 
    #labels for the queries. Depends not on the identity, but on whether it is a real access or spoofing attack
    real_labels = get_labels(os.path.join(fv_dirs[0], dir_precise), real, protocol='spoof', onlyValidScores=False, binary_labels=binary_labels)
    attack_labels = get_labels(os.path.join(fv_dirs[0], dir_precise), attack, protocol='spoof', onlyValidScores=False, binary_labels=binary_labels)
    real_scores = numpy.append(real_fv, numpy.append(real_as, real_iqa, axis=1), axis=1) if iqa_dirs is not None else numpy.append(real_fv, real_as, axis=1)
    all_labels = numpy.append(all_labels, real_labels)
    all_scores = numpy.append(all_scores, real_scores, axis=0)
    attack_scores = numpy.append(attack_fv, numpy.append(attack_as, attack_iqa, axis=1), axis=1) if iqa_dirs is not None else numpy.append(attack_fv, attack_as, axis=1)
    all_labels = numpy.append(all_labels, attack_labels)
    all_scores = numpy.append(all_scores, attack_scores, axis=0)

  # remove rows with scores with nan values
  all_labels = all_labels[~numpy.isnan(all_scores).any(axis=1)]
  all_scores = all_scores[~numpy.isnan(all_scores).any(axis=1)]

  # do polinomial augmentation
  if pol_augment == True:
    all_scores = polinomial_augmentation(all_scores)

  # standard normalization of the data if it is required
  if normalize == True:
    score_norm = ScoreNormalization(all_scores) # training data are normalized by themselves
    all_scores = score_norm.calculateZNorm(all_scores)

  sys.stdout.write('---------------------------------------------------------\n')
  return all_scores, all_labels


def gather_fvas_iqa_scores(database, subset, fv_dirs, as_dirs, iqa_dirs=None, binary_labels=True, fv_protocol='both', normalize=True, score_norm=None, pol_augment=False, remove_cols=None):
  """Populates a numpy.ndarray with the scores of face verification and anti-spoofing algorithm(s) and a numpy.array with their corresponding labels. Each column of the arrays correspond to a face verification / anti-spoofing algorithm (with face verification algorithms coming first).
  
    @param database The database (replay)
    @param subset 'devel', 'test' or 'train'
    @param fv_dirs List of directories of the scores of face verification algorithms
    @param as_dirs List of directories of the scores of anti-spoofing algorithms
    @param iqa_dirs Directory containing the IQA scores or features
    @param binary_labels If True, binary labels will be returned: both impostors and spoofing attacks will be labeled with 0, while real accesses with 1. If False, ternary labels will be returned: impostors: 0, real accesses: 1, spoofing attacks: -1
    @param fv_protocol Specifies the face verification protocol for the returned scores. Can be 'licit', 'spoof' or 'both'
    @param normalize If True, the returned data will be normalized with regards to the training set
    @param score_norm Instance of the antispoofing.utils.ml.ScoreNormalization class, containing the normalization parameters computed over some training data
    @param pol_augment If True, the data will be polinomially augmented (columns with quadratic values will be added to the data
    @param remove_cols Columns to be removed from the IQA features
  """

  if subset == 'devel':
    real, attack = database.get_devel_data()
  elif subset == 'test':
    real, attack   = database.get_test_data()
  else:
    real, attack   = database.get_train_data()

  clients = list(set(["client%03d" % x.get_client_id() for x in real]))
  
  sys.stdout.write('Organizing faceverif and antispoofing scores: %s set\n' % (subset))
  
  # read the IQA scores
  if iqa_dirs is not None:
    real_iqa = simple_feat_gather(iqa_dirs, real)
    attack_iqa = simple_feat_gather(iqa_dirs, attack)

    if remove_cols is not None:
      cols_to_keep = list(set(range(real_iqa.shape[1])) - set(remove_cols))
      real_iqa = real_iqa[:,cols_to_keep]
      attack_iqa = attack_iqa[:,cols_to_keep]

  # create empty array for all scores
  all_scores = numpy.ndarray((0, len(fv_dirs) + len(as_dirs) + real_iqa.shape[1]), 'float') if iqa_dirs is not None else numpy.ndarray((0, len(fv_dirs) + len(as_dirs)), 'float');  
  all_labels = numpy.array([], 'int');
  
  # reading the anti-spoofing data
  if as_dirs != None:
    real_scorereader_as = ScoreFusionReader(real, as_dirs)
    attack_scorereader_as = ScoreFusionReader(attack, as_dirs)
    real_as = real_scorereader_as.getConcatenetedScores(onlyValidScores=False)
    attack_as = attack_scorereader_as.getConcatenetedScores(onlyValidScores=False)

  # reading the face verification data
  if fv_protocol == 'licit' or fv_protocol == 'both':
    sys.stdout.write('Processing face verif scores: LICIT protocol\n')
    #dir_precise = os.path.join('10_licit', 'nonorm')
    dir_precise = os.path.join('licit')
    for cl in clients:
      sys.stdout.write("Processing [%s/%d] in  %s set\n" % (cl, len(clients), subset))
      # creating the scores readers from different face verification algorithms
      # joining the scores for face verfication with the anti-spoofing scores. The face verifications scores dirs need to have the client labels
      real_scorereader_fv = ScoreFusionReader(real, [os.path.join(sd, dir_precise, cl) for sd in fv_dirs])
      real_fv = real_scorereader_fv.getConcatenetedScores(onlyValidScores=False) # raw scores as numpy.array (the scores should be already normalized in the input files)
      labels = get_labels(os.path.join(fv_dirs[0], dir_precise), real, protocol='licit', client_id=cl, onlyValidScores=False, binary_labels=binary_labels) # labels for the face verification queries. The samples in the 'licit' protocol are real accesses => their label depends only on the identities
      scores = numpy.append(real_fv, numpy.append(real_as, real_iqa, axis=1), axis=1) if iqa_dirs is not None else numpy.append(real_fv, real_as, axis=1)
      all_labels = numpy.append(all_labels, labels)
      all_scores = numpy.append(all_scores, scores, axis=0)
  
  if fv_protocol == 'spoof' or fv_protocol == 'both':
    sys.stdout.write('Processing face verif scores: SPOOF protocol\n')
    #dir_precise = os.path.join('10_spoof', 'nonorm')
    dir_precise = os.path.join('spoof')
    real_scorereader_fv = ScoreFusionReader(real, [os.path.join(sd, dir_precise) for sd in fv_dirs])
    attack_scorereader_fv = ScoreFusionReader(attack, [os.path.join(sd, dir_precise) for sd in fv_dirs])  

    # get raw scores as numpy.array (the scores should be already normalized in the input files)
    real_fv = real_scorereader_fv.getConcatenetedScores(onlyValidScores=False)
    attack_fv = attack_scorereader_fv.getConcatenetedScores(onlyValidScores=False) 
    #labels for the queries. Depends not on the identity, but on whether it is a real access or spoofing attack
    real_labels = get_labels(os.path.join(fv_dirs[0], dir_precise), real, protocol='spoof', onlyValidScores=False, binary_labels=binary_labels)
    attack_labels = get_labels(os.path.join(fv_dirs[0], dir_precise), attack, protocol='spoof', onlyValidScores=False, binary_labels=binary_labels)
    real_scores = numpy.append(real_fv, numpy.append(real_as, real_iqa, axis=1), axis=1) if iqa_dirs is not None else numpy.append(real_fv, real_as, axis=1)
    all_labels = numpy.append(all_labels, real_labels)
    all_scores = numpy.append(all_scores, real_scores, axis=0)
    attack_scores = numpy.append(attack_fv, numpy.append(attack_as, attack_iqa, axis=1), axis=1) if iqa_dirs is not None else numpy.append(attack_fv, attack_as, axis=1)
    all_labels = numpy.append(all_labels, attack_labels)
    all_scores = numpy.append(all_scores, attack_scores, axis=0)
    
  # remove rows with scores with nan values
  all_labels = all_labels[~numpy.isnan(all_scores).any(axis=1)]
  all_scores = all_scores[~numpy.isnan(all_scores).any(axis=1)]

  # do polinomial augmentation
  if pol_augment == True:
    all_scores = polinomial_augmentation(all_scores)

  # standard normalization of the data if it is required
  if normalize == True:
    if score_norm == None:
      sys.stderr.write('Error: Normalization can not be done: no normalization parameters specified!\n')    
      sys.exit(1)
      #train_scores, train_labels = gather_train_fvas_scores(database, fv_dirs, as_dirs, fv_procotol=fv_protocol, normalized=False, pol_augment=pol_augment)
      #score_norm = ScoreNormalization(train_scores)
    all_scores = score_norm.calculateZNorm(all_scores)
    
  sys.stdout.write('---------------------------------------------------------\n')
  return all_scores, all_labels
