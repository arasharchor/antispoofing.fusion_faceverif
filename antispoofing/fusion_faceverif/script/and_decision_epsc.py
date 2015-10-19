#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Tue Feb 11 10:54:37 CET 2014

"""
This script performs AND decision level fusion of face verification and anti-spoofing system and plots the EPSC for the decision

NOTE: While the script can receive more face verification and anti-spoofing systems as arguments (this is true also for the thresholds), the functionality to fuse more then one face verificaiton and one anti-spoofing system is not supported yet.

"""

import os, sys
import argparse
import bob.io.base
import numpy

import antispoofing

from antispoofing.utils.db import *
from antispoofing.utils.ml import *
from antispoofing.utils.helpers import *
from antispoofing.fusion_faceverif.helpers import *

import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as mpl

from matplotlib import rc
rc('text',usetex=1)



def main():

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-s', '--fv-scores-dir', type=str, dest='fv_scoresdir', default='', help='Base directory containing the scores of one or more face verification algorithms (without the protocol dir)', nargs='+') #nargs='+'

  parser.add_argument('-a', '--as-scores-dir', type=str, dest='as_scoresdir', default='', help='Base directory containing the scores of one or more antispoofing algorithms', nargs='+') #nargs='+'

  parser.add_argument('--ft', '--fv-threshold', type=float, dest='fv_threshold', default=None, help='The face verification threshold', nargs='+') #nargs='+'

  parser.add_argument('--at', '--as-threshold', type=float, dest='as_threshold', default=None, help='The anti-spoofing threshold', nargs='+') #nargs='+'
  
  parser.add_argument('--sp', '--save_params', action='store_true', dest='save_params', default=False, help='Save the decision thresholds in the outputdir for future use')
  
  parser.add_argument('--op', '--output', metavar='FILE', type=str, default='plots.pdf', dest='output', help='Set the name of the output plot file (defaults to "%(default)s")')
  parser.add_argument('--oh', '--outhdf', metavar='FILE', type=str, default='toplot_and.pdf', dest='outhdf', help='Set the name of the output hdf file (defaults to "%(default)s")')

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

  # plot EPSC for HTER_w
  points = 100
  step_size = 1 / float(points)
  omega = numpy.array([(i * step_size) for i in range(points+1)])
  far_w = [(1-w) * test_far + w * test_sfar for w in omega]
  hter = [(far + test_frr) / 2 for far in far_w]
  
  from scipy import integrate
  aue = integrate.cumtrapz(hter, omega)


  pp = PdfPages(args.output)
  fig = mpl.figure()
  
  mpl.plot(omega, 100. * numpy.array(hter), color='red', label = "AND", linewidth=4)
      
  mpl.xlabel("Weight $\omega$")
  mpl.ylabel(r"HTER$_{\omega}$ (\%)")

  #mpl.title(r"EPSC with %s criteria: WER$_{\omega,\beta}$" % (criteria.upper()) if args.title == "" else args.title)

  #mpl.legend(prop=fm.FontProperties(size=18), loc = 4)
  fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'], 'weight' : 'normal'}
  ax1 = mpl.subplot(111) # EPC like curves for FVAS fused scores for weighted error rates between the negatives (impostors and spoofing attacks)
  ax1.set_xticklabels(ax1.get_xticks(), fontProperties)
  ax1.set_yticklabels(ax1.get_yticks(), fontProperties)
  mpl.grid()  

  pp.savefig()  
  pp.close() # close multi-page PDF writer
  
  # write the values that need to be printed in hdf5file 
  f = bob.io.base.HDF5File(args.outhdf, 'w')
  f.set('omega', numpy.array(omega))
  f.set('hter_w', numpy.array(hter))
  f.set('sfar', numpy.array((points+1)*[test_sfar]))
  del f
  
  
  # print results
  sys.stdout.write("FV threshold: %f, AS threshold: %f\n" % (fv_thr, as_thr))
  sys.stdout.write("----------------------------------------------------------\n")
  sys.stdout.write("AND fused system results:\n")
  sys.stdout.write("Devel: FAR=%.3f, FRR=%.3f, HTER=%.3f, SFAR=%.3f\n" % (devel_far*100, devel_frr*100, devel_far*50 + devel_frr*50, devel_sfar*100))
  sys.stdout.write("Test: FAR=%.3f, FRR=%.3f, HTER=%.3f, SFAR=%.3f\n" % (test_far*100, test_frr*100, test_far*50 + test_frr*50, test_sfar*100))

  sys.stdout.write("AUE = %.4f \n" % aue[-1]) # for indexing purposes, aue is cumulative integration
  
if __name__ == "__main__":
  main()




