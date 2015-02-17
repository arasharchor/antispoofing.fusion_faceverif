#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Tue Nov 20 17:53:32 CET 2012

import os, sys
import argparse
import bob.io.base
import numpy

import antispoofing

from antispoofing.utils.db import *
from antispoofing.utils.ml import *
from antispoofing.utils.helpers import *
from antispoofing.fusion.readers import *

def polinomial_augmentation(scores):
  num_dim = scores.shape[1] * (scores.shape[1] + 1) / 2  + scores.shape[1]# number of dimensions in the augmented feature space
  augscores = numpy.ndarray([scores.shape[0], num_dim], 'float64')

  augscores[:,range(scores.shape[1])] = scores
  ind = scores.shape[1]
  for i in range(scores.shape[1]):
    for j in range(i, scores.shape[1]):
      augscores[:,ind] = scores[:,i] * scores[:,j]
      ind += 1

  return augscores


def save_fused_scores(all_scores, all_labels, dirname, protocol, subset):
  """Saves the fused scores in a 4 column format. Since the exact identity of the user is not known, as well as the filename of the used file, we will put dummy identities and dummy filename in the first three columns"""

  # remove nan scores
  #outdir = os.path.join(dirname, "10_" + protocol, "scores")
  outdir = os.path.join(dirname, protocol, "scores")
  filename = "scores-" + subset
  ensure_dir(outdir)
  f = open(os.path.join(outdir, filename), 'w')
  for i in range(0, len(all_scores)):
    if numpy.isnan(all_scores[i]):
      continue
    if all_labels[i] == 0: # negative (imposter or spoof, depending on the protocol)
      f.write("x y foo %f\n" % all_scores[i])
    else: # positive sample, real access
      f.write("x x foo %f\n" % all_scores[i])
  f.close()

def save_fusion_machine(machine, dirname, protocol, subset):
  """ Saves a trained fusion machine in an .hdf5 file for future use
  """
  #outdir = os.path.join(dirname, "10_" + protocol, "scores")
  outdir = os.path.join(dirname, protocol, "scores")
  ensure_dir(outdir)
  outfile = bob.io.base.HDF5File(os.path.join(outdir, 'llrmachine-' + subset + '.hdf5'), 'w')
  machine.save(outfile)

def save_norm_params(norm_params, dirname, protocol, subset):
  """ Saves parameters of normalization in an .hdf5 file for future use
  """
  #outdir = os.path.join(dirname, "10_" + protocol, "scores")
  outdir = os.path.join(dirname, protocol, "scores")
  ensure_dir(outdir)
  outfile = os.path.join(outdir, 'normparams-' + subset + '.hdf5')
  mins = numpy.reshape(norm_params.mins, [1, len(norm_params.mins)])
  maxs = numpy.reshape(norm_params.maxs, [1, len(norm_params.maxs)])
  avg = numpy.reshape(norm_params.avg, [1, len(norm_params.avg)])
  std = numpy.reshape(norm_params.std, [1, len(norm_params.std)])

  norm_params_array = numpy.append(mins, maxs, axis=0)
  norm_params_array = numpy.append(norm_params_array, avg, axis=0)
  norm_params_array = numpy.append(norm_params_array, std, axis=0) 
  bob.io.base.save(norm_params_array, outfile)

def get_labels(indir, files, protocol, client_id=None, onlyValidScores=True, binary_labels=True):
  """
  Return a numpy.array with the labels of all bob.db.replay.File

  @param indir The input directory where the score files are stored. They need to be checked for labeling together with the filename, in order to determine the number of valid scores in each file 
  @param files The files that need to be labeled
  @param protocol The protocol: "licit" or "spoof"
  @param client_id The id of the client if the protocol is "licit", otherwise can be left to None
  @param onlyValidScores if True will return list of labels of only the valid scores in the files
  @param binary_labels If True, binary labels will be returned: both impostors and spoofing attacks will be labeled with 0, while real accesses with 1. If False, ternary labels will be returned: impostors: 0, real accesses: 1, spoofing attacks: -1
  """

  def reshape(scores):
    if(scores.shape[1]==1):
      scores = numpy.reshape(scores,(scores.shape[1],scores.shape[0]))

    return scores  

  #Finding the number of elements (total number of scores in all the files)
  totalScores = 0
  for f in files:
    if protocol == 'licit':
      fileName = str(f.make_path(os.path.join(indir, client_id), extension='.hdf5'))
    else: #'spoof' protocol
      fileName = str(f.make_path(indir,extension='.hdf5'))
    scores = bob.io.base.load(fileName)
    scores = reshape(scores)
    totalScores =totalScores + scores.shape[1]
    
  # calculating the labels
  allLabels = numpy.ndarray((totalScores), 'int')
  allScores = numpy.zeros(shape=(totalScores))
  offset = 0
  for f in files:
    if protocol == 'licit':
      fileName = str(f.make_path(os.path.join(indir, client_id),extension='.hdf5'))
    else: #'spoof' protocol
      fileName = str(f.make_path(indir,extension='.hdf5'))
    scores = bob.io.base.load(fileName)
    scores = reshape(scores)
    labels = numpy.ndarray(scores.shape, 'int')
    if protocol == 'licit':
      if client_id == "client%03d" % f.get_client_id():
        if f.is_real():
          labels[0,:] = 1
        else:
          if binary_labels == True: 
            labels[0,:] = 0 
          else: 
            labels[0,:] = -1
      else:
        labels[0,:] = 0
    else: # 'spoof' protocol
      if f.is_real():
        labels[0,:] = 1
      else:
        if binary_labels == True: 
          labels[0,:] = 0
        else: 
          labels[0,:] = -1
    
    allScores[offset:offset+scores.shape[1]] = numpy.reshape(scores,(scores.shape[1]))
    allLabels[offset:offset+labels.shape[1]] = numpy.reshape(labels,(labels.shape[1]))
    offset = offset + labels.shape[1]

  if(onlyValidScores):
    allLabels = allLabels[(numpy.where(numpy.isnan(allScores)==False))[0]]

  return allLabels


def gather_train_fvas_scores(database, fv_dirs, as_dirs, binary_labels=True, fv_protocol='both', normalize=True, pol_augment=False):
  """Populates a numpy.ndarray with the nor normalized training scores of face verification and anti-spoofing algorithm(s) and a numpy.array with their corresponding labels. Each column of the arrays correspond to a face verification / anti-spoofing algorithm (with face verification algorithms coming first). The returning result canbe used for normalization purposes
  
  @param database The database (replay)
  @param fv_dirs List of directories of the scores of face verification algorithms
  @param as_dirs List of directories of the scores of anti-spoofing algorithms
  @param binary_labels If True, binary labels will be returned: both impostors and spoofing attacks will be labeled with 0, while real accesses with 1. If False, ternary labels will be returned: impostors: 0, real accesses: 1, spoofing attacks: -1
  @param fv_protocol Specifies the face verification protocol for the returned scores. Can be 'licit', 'spoof' or 'both'
  @param normalize if True, the returned data will be normalized with regards to the training set
  @param pol_augment If True, the data will be polinomially augmented (columns with quadratic values will be added to the data
"""

  real, attack = database.get_train_data()

  clients = list(set(["client%03d" % x.get_client_id() for x in real]))
  
  sys.stdout.write('Organizing faceverif and antispoofing scores\n')
  
  all_scores = numpy.ndarray((0, len(fv_dirs) + len(as_dirs)), 'float');
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
      scores = numpy.append(real_fv, real_as, axis=1)
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
    real_scores = numpy.append(real_fv, real_as, axis=1)
    all_labels = numpy.append(all_labels, real_labels)
    all_scores = numpy.append(all_scores, real_scores, axis=0)
    attack_scores = numpy.append(attack_fv, attack_as, axis=1)
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


def gather_fvas_scores(database, subset, fv_dirs, as_dirs=None, binary_labels=True, fv_protocol='both', normalize=True, score_norm=None, pol_augment=False):
  """Populates a numpy.ndarray with the scores of face verification and anti-spoofing algorithm(s) and a numpy.array with their corresponding labels. Each column of the arrays correspond to a face verification / anti-spoofing algorithm (with face verification algorithms coming first).
  
  @param database The database (replay)
  @param subset 'devel', 'test' or 'train'
  @param fv_dirs List of directories of the scores of face verification algorithms
  @param as_dirs List of directories of the scores of anti-spoofing algorithms
  @param binary_labels If True, binary labels will be returned: both impostors and spoofing attacks will be labeled with 0, while real accesses with 1. If False, ternary labels will be returned: impostors: 0, real accesses: 1, spoofing attacks: -1
  @param fv_protocol Specifies the face verification protocol for the returned scores. Can be 'licit', 'spoof' or 'both'
  @param normalize If True, the returned data will be normalized with regards to the training set
  @param score_norm Instance of the antispoofing.utils.ml.ScoreNormalization class, containing the normalization parameters computed over some training data
  @param pol_augment If True, the data will be polinomially augmented (columns with quadratic values will be added to the data
"""

  if subset == 'devel':
    real, attack = database.get_devel_data()
  elif subset == 'test':
    real, attack   = database.get_test_data()
  else:
    real, attack   = database.get_train_data()

  clients = list(set(["client%03d" % x.get_client_id() for x in real]))
  
  sys.stdout.write('Organizing faceverif and antispoofing scores: %s set\n' % (subset))
  
  if as_dirs == None:
    all_scores = numpy.ndarray((0, len(fv_dirs)), 'float');
  else:
    all_scores = numpy.ndarray((0, len(fv_dirs) + len(as_dirs)), 'float');  
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
      if as_dirs != None:
        scores = numpy.append(real_fv, real_as, axis=1)
      else:
        scores = real_fv  
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
    if as_dirs != None:
      real_scores = numpy.append(real_fv, real_as, axis=1)
      attack_scores = numpy.append(attack_fv, attack_as, axis=1)
    else:
      real_scores = real_fv
      attack_scores = attack_fv
      
    all_labels = numpy.append(all_labels, real_labels)
    all_scores = numpy.append(all_scores, real_scores, axis=0)
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
  
  
def organize_llrtraining_scores(database, fv_dirs, as_dirs, normalize=True, pol_augment=False):
  all_scores, all_labels = gather_train_fvas_scores(database, fv_dirs, as_dirs, binary_labels=True, normalize=normalize, pol_augment=pol_augment)
  all_pos = all_scores[all_labels == 1,:]
  all_neg = all_scores[all_labels == 0,:]
  return all_pos, all_neg
  

