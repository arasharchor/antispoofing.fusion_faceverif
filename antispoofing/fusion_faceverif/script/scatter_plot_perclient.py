#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Sat 10 Oct 12:03:43 CEST 2015

"""
This script plots scores of a face verification system and an antispoofing counter measure in 2D. 

"""

import os, sys
import argparse
import bob.learn.linear
import bob.io.base
import numpy
import string

import antispoofing

from antispoofing.utils.db import *
from antispoofing.utils.ml import *
from antispoofing.utils.helpers import *
from antispoofing.fusion_faceverif.helpers import * 

import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx


from matplotlib.lines import Line2D                
import matplotlib.font_manager as fm

def main():

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-s', '--fv-scores-dir', type=str, dest='fv_scoresdir', default='', help='Base directory containing the scores of different face verification algorithms (without the protocol dir and the stdnorm dir; they are added automatically within the program)', nargs='+')

  parser.add_argument('-a', '--as-scores-dir', type=str, dest='as_scoresdir', default='', help='Base directory containing the scores of different antispoofing algorithms', nargs='+')

  parser.add_argument('-o', '--output', type=str, dest='output', metavar='FILE', default='plots.pdf', help='Set the name of the output file prefixed with output directory (defaults to "%(default)s")')

  parser.add_argument('-t', '--title', metavar='STR', type=str, dest='title', default="", help='Plot title')  

  parser.add_argument('-m', '--machine-input', type=str, dest='machine_input', default=None, help='Base file containing the LLR or LLR_P machine to be plotted')

  parser.add_argument('-n', '--norm-input', type=str, dest='norm_input', default=None, help='Base file containing normalization parameters')

  parser.add_argument('-d', '--devel-thres', type=str, dest='devel_thres', default=None, help='EER/HTER threshold on the devel set (will be used to plot the decision boundary). If the fusion algorithm is AND, two thresholds need to be specified: first faceverif, then antispoofing', nargs='+')

  parser.add_argument('--cs', '--client_spec', action='store_true', dest='clientspec', default=False, help='Should be set to TRue if you want to use the fusion with client-specific anti-spoofing algorithm')

  parser.add_argument('-f', '--fusion-alg', type=str, dest='fusion_alg', default=None, choices=('LLR', 'SUM', 'LLR_P', 'AND'), help='The fusion algorithm used (based on this the decision boundary will be plotted)')

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')

  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()
  
  #ensure_dir(os.path.dirname(args.output))

  pp = PdfPages(args.output) 
  #######################
  # Loading the database objects
  #######################
  database = args.cls(args)
  
  # read normalization parameters
  
  if args.norm_input != None:
    norm_params = bob.io.base.load(args.norm_input)
    score_norm = ScoreNormalization()
    score_norm.set_norm_params(norm_params[0,:], norm_params[1,:], norm_params[2,:], norm_params[3,:])  
    normalize = True
  else:
    score_norm = None
    normalize = False


  groups_to_plot = ['devel', 'test']
  
  for group in groups_to_plot:
    if args.clientspec == True:
      real_scores, _ = fusion_utils.gather_fvas_clsp_scores_perclient(database, group, args.fv_scoresdir, args.as_scoresdir, binary_labels=False, normalize = normalize, score_norm = score_norm) 
      all_scores, all_labels, _ = fusion_utils.gather_fvas_clsp_scores(database, group, args.fv_scoresdir, args.as_scoresdir, fv_protocol='spoof', binary_labels=False, normalize = normalize, score_norm = score_norm)
    else:   
      real_scores, _ = fusion_utils.gather_fvas_scores_perclient(database, group, args.fv_scoresdir, args.as_scoresdir, binary_labels=False, normalize = normalize, score_norm = score_norm)
      all_scores, all_labels = fusion_utils.gather_fvas_scores(database, group, args.fv_scoresdir, args.as_scoresdir, fv_protocol='spoof', binary_labels=False, normalize = normalize, score_norm = score_norm)

    #all_scores = all_scores[:,0:2]
    #real_accesses = all_scores[all_labels == 1,:]
    impostor_scores = all_scores[all_labels == 0,:]
    attack_scores = all_scores[all_labels == -1,:] 
    
    values = range(len(real_scores))
    #define the colormap
    #set1 = cm = mpl.get_cmap('Paired') 
    set1 = cm = mpl.get_cmap('nipy_spectral') 
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=set1)
    
    #print scalarMap.get_clim()


    fig = mpl.figure()
    mpl.rcParams.update({'font.size': 18})
    # because of the large number of samples, we plot only each 10th sample (or 5th or so...)
    imp_range = range(0, impostor_scores.shape[0], 10)
    att_range = range(0, attack_scores.shape[0], 10)
    #racc_range = range(0, real_accesses.shape[0], 10)

    idx=0
    for clid, scores in real_scores.items():
      scores_range = range(0, scores.shape[0], 4)
      colorVal = scalarMap.to_rgba(values[idx])
      print(colorVal)
      mpl.plot(scores[scores_range,0], scores[scores_range,1], marker='o', color=colorVal, linestyle='None', label=clid) # alpha = 0.2  
      #mpl.plot(scores[:,0], scores[:,1], marker='o', color=colorVal, linestyle='None') # alpha = 0.2  
      idx+=1

    #mpl.plot(impostor_scores[imp_range,0], impostor_scores[imp_range,1], marker='^', color='#257bd4', label = 'impostors', linestyle='None')
    #mpl.plot(attack_scores[att_range,0], attack_scores[att_range,1], marker='s', color='#d4257b', label = 'spoofing attacks', linestyle='None')
    #mpl.plot(real_accesses[racc_range,0], real_accesses[racc_range,1], marker='o', color='#7bd425', label = 'genuine users', alpha = 0.8, linestyle='None') # alpha = 0.2
    #pp.save_fig()

        
    mpl.xlabel('Face verification scores')
    mpl.ylabel('Anti-spoofing scores')
    #mpl.title(args.title + " - " + string.upper(group) + " set") 
    #mpl.legend(loc='lower right', prop=fm.FontProperties(size=16))
    mpl.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0., prop=fm.FontProperties(size=4))
    mpl.grid()
    pp.savefig()
  
  pp.close()
  #mpl.show()    


if __name__ == "__main__":
  main()
  
