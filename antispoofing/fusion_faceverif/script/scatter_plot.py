#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Wed Nov 14 17:01:47 CET 2012

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

  if args.fusion_alg == "LLR_P":
    pol_augment=True
  else:
    pol_augment=False

  groups_to_plot = ['devel', 'test']
  for group in groups_to_plot:
    
    all_scores, all_labels = fusion_utils.gather_fvas_scores(database, group, args.fv_scoresdir, args.as_scoresdir, binary_labels=False, normalize = normalize, score_norm = score_norm, pol_augment = pol_augment) 

    all_scores = all_scores[:,0:2]
    real_accesses = all_scores[all_labels == 1,:]
    impostors = all_scores[all_labels == 0,:]
    attacks = all_scores[all_labels == -1,:] 
    
    fig = mpl.figure()
    mpl.rcParams.update({'font.size': 18})
    # because of the large number of samples, we plot only each 10th sample (or 5th or so...)
    imp_range = range(0, impostors.shape[0], 10)
    att_range = range(0, attacks.shape[0], 10)
    racc_range = range(0, real_accesses.shape[0], 10)

    mpl.plot(impostors[imp_range,0], impostors[imp_range,1], 'b^', label = 'impostors')
    mpl.plot(attacks[att_range,0], attacks[att_range,1], 'rs', label = 'spoofing attacks')
    mpl.plot(real_accesses[racc_range,0], real_accesses[racc_range,1], 'go', label = 'real accesses', alpha = 0.5) # alpha = 0.2
    #pp.save_fig()
     
    if args.fusion_alg == 'LLR': #plot LLR decision boundary
      if args.machine_input == None or score_norm == None or args.devel_thres == None:
        sys.stdout.write('Decision boundary can not be plotted: no input file for LLR machine or for score normalization or decision threshold specified!')
        
      else:
        llr_machine = bob.learn.linear.Machine(bob.io.base.HDF5File(args.machine_input))
        n_points = 1000
        xlim = mpl.xlim()
        x = [xlim[0]+(xlim[1]-xlim[0])/3.0, xlim[1]-(xlim[1]-xlim[0])/3]
        y = mpl.ylim()
        yrange = numpy.arange(y[0], y[1], (y[1]-y[0])/n_points)
        yrange = numpy.reshape(yrange, [len(yrange), 1])
        x1 = numpy.array([x[0]]*yrange.size); x1 = numpy.reshape(x1, [x1.size,1])
        x2 = numpy.array([x[1]]*yrange.size); x2 = numpy.reshape(x2, [x2.size,1])

        xy1 = numpy.append(x1, yrange, axis=1); xy2 = numpy.append(x2, yrange, axis=1)
        #xy1 = score_norm.inverseZNorm(xy1, (0,1)); xy2 = score_norm.inverseZNorm(xy2, (0,1))
        xy1 = score_norm.calculateZNorm(xy1); xy2 = score_norm.calculateZNorm(xy2) #!!!!!!!!!!!!

        yrangex1 = float(args.devel_thres[0]) - llr_machine(xy1)
        yrangex2 = float(args.devel_thres[0]) - llr_machine(xy2)
        y = [yrange[numpy.where(numpy.abs(yrangex1) == numpy.min(numpy.abs(yrangex1)))], yrange[numpy.where(numpy.abs(yrangex2) == numpy.min(numpy.abs(yrangex2)))]]

        ylim = (numpy.array(xlim) - x[0])*(y[1]-y[0])/(x[1]-x[0]) + y[0] # calculating full line (that spans the full plot)

        ax = fig.axes
        #line = Line2D(x, y, color='black', label = 'decision boundary', linewidth=2)
        line = Line2D(xlim, ylim, color='black', label = 'decision boundary', linewidth=3)
        ax[0].add_line(line)

    elif args.fusion_alg == 'LLR_P': #plot PLR decision boundary
      if args.machine_input == None or score_norm == None or args.devel_thres == None:
        sys.stdout.write('Decision boundary can not be plotted: no input file for LLR machine or for score normalization or decision threshold specified!')
      else:
        llr_machine = bob.learn.linear.Machine(bob.io.base.HDF5File(args.machine_input))
        
        ax = mpl.axis()
        grid_width = 100 
        grid = numpy.mgrid[
          ax[0]:ax[1]:(abs(ax[1]-ax[0])/grid_width),
          ax[2]:ax[3]:(abs(ax[3]-ax[2])/grid_width)
          ]

        #import pdb; pdb.set_trace()
        # Then, you create your "fake" input set by stacking the grid points:
        xy = numpy.vstack(grid.T)

        # Then, you pass your "fake" input set by the polynomial making function
        xy = score_norm.inverseZNorm(xy, (0,1)) ####!!! Why was I doing this step anyway???
        xy = fusion_utils.polinomial_augmentation(xy)
        xy = score_norm.calculateZNorm(xy)
        xy_res = llr_machine(xy)

        # You then pass the polynomial terms through your LLR machine
        # You reshape the output to match the original grid
        xy_res = xy_res.reshape(grid.shape[2], grid.shape[1])
        #xy_thr = float(args.devel_thres) - llr_machine(xy)

        # Now, you only need to draw the contour with this grid. To draw only the
        # line with Z=0.5 (which is the separation threshold for the LLR):
        mpl.contour(grid[0,:,0], grid[1,0,:], xy_res, levels=[float(args.devel_thres[0])], colors="k", linewidths=3, linestyles = 'solid', zorder=10)
      
    elif args.fusion_alg == 'SUM': # plot SUM decision boundary
      if args.devel_thres == None or score_norm == None:
        sys.stdout.write('Decision boundary can not be plotted: no decision threshold specified!')      
      else:
        x = mpl.xlim()
        y = [float(args.devel_thres[0]) - i for i in x]
        ax = fig.axes
        line = Line2D(x, y, color='black', label = 'decision boundary', linewidth=3)
        ax[0].add_line(line)
        
    elif args.fusion_alg == 'AND': # plot AND decision boundary
      
      if args.devel_thres == None or len(args.devel_thres) < 2:
        sys.stdout.write('Decision boundary can not be plotted: at least two decision thresholds are expected!')      
      else:
        fv_thr = float(args.devel_thres[0])
        as_thr = float(args.devel_thres[1])
        #fv_thr, as_thr = fusion_utils.normalize_points([fv_thr, as_thr], database, args.fv_scoresdir, args.as_scoresdir)
        #x = [fv_thr, mpl.xlim()[1]] 
        #y = [as_thr, as_thr]
        ax = fig.axes
        mpl.hlines(y=as_thr, xmin=fv_thr, xmax=mpl.xlim()[1], color='black', label = 'decision boundary', linewidth=3, zorder=10)
        mpl.vlines(x=fv_thr, ymin=as_thr, ymax=mpl.ylim()[1], color='black', label = 'decision boundary', linewidth=3, zorder=10)
        
    mpl.xlabel('Face verification scores')
    mpl.ylabel('Anti-spoofing scores')
    #mpl.title(args.title + " - " + string.upper(group) + " set") 
    #mpl.legend(loc='lower right', prop=fm.FontProperties(size=16))
    #mpl.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., prop=fm.FontProperties(size=10))
    mpl.grid()
    pp.savefig()
  
  pp.close()
  #mpl.show()    


if __name__ == "__main__":
  main()
  
