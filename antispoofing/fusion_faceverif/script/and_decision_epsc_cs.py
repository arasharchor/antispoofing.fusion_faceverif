#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Mon 12 Oct 13:00:33 CEST 2015

"""
This script performs AND decision level fusion of face verification and anti-spoofing system and plots the EPSC for the decision. The anti-spoofing systems should be client-specific

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

from antispoofing.evaluation.utils import error_utils
import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as mpl

from matplotlib import rc
rc('text',usetex=1)



def main():

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('-s', '--fv-scores-dir', type=str, dest='fv_scoresdir', default='', help='Base directory containing the scores of one or more face verification algorithms (without the protocol dir)', nargs='+') #nargs='+'

  parser.add_argument('-a', '--as-scores-dir', type=str, dest='as_scoresdir', default='', help='Base directory containing the scores of one or more antispoofing algorithms', nargs='+') #nargs='+'

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
  
  # read faceverif and antispoofing scores for all samples
  devel_scores, devel_labels, _ = gather_fvas_clsp_scores(database, 'devel', args.fv_scoresdir, args.as_scoresdir, binary_labels=False, fv_protocol='both', normalize=False)
  test_scores, test_labels, _ = gather_fvas_clsp_scores(database, 'test', args.fv_scoresdir, args.as_scoresdir, binary_labels=False, fv_protocol='both', normalize=False)

  # separate the scores of valid users, impostors and spoofing attacks
  valid_devel_fv_scores = devel_scores[devel_labels == 1,0];  valid_devel_as_scores = devel_scores[devel_labels == 1,1]
  valid_test_fv_scores = test_scores[test_labels == 1,0];   valid_test_as_scores = test_scores[test_labels == 1,1]
  
  impostors_devel_fv_scores = devel_scores[devel_labels == 0,0];   impostors_devel_as_scores = devel_scores[devel_labels == 0,1]
  impostors_test_fv_scores = test_scores[test_labels == 0,0];   impostors_test_as_scores = test_scores[test_labels == 0,1]
  
  spoof_devel_fv_scores = devel_scores[devel_labels == -1,0];   spoof_devel_as_scores = devel_scores[devel_labels == -1,1]
  spoof_test_fv_scores = test_scores[test_labels == -1,0];   spoof_test_as_scores = test_scores[test_labels == -1,1]
  
  points = 100
  criteria = 'eer'

  omega, beta, thrs_fv = error_utils.epsc_thresholds(impostors_devel_fv_scores, valid_devel_fv_scores, spoof_devel_fv_scores, valid_devel_fv_scores, points=points, criteria=criteria, beta=0.5)
  omega, beta, thrs_as = error_utils.epsc_thresholds(impostors_devel_as_scores, valid_devel_as_scores, spoof_devel_as_scores, valid_devel_as_scores, points=points, criteria=criteria, beta=0.5) 

  test_far = numpy.ndarray(omega.shape, dtype='float64')
  test_sfar = numpy.ndarray(omega.shape, dtype='float64')
  test_frr = numpy.ndarray(omega.shape, dtype='float64')

  for i in range(len(omega)):
    test_fa = numpy.logical_and(impostors_test_fv_scores >= thrs_fv[0][i], impostors_test_as_scores >= thrs_as[0][i])
    test_sfa = numpy.logical_and(spoof_test_fv_scores >= thrs_fv[0][i], spoof_test_as_scores >= thrs_as[0][i])
    test_fr = numpy.logical_or(valid_test_fv_scores < thrs_fv[0][i], valid_test_as_scores < thrs_as[0][i])

    test_far[i] = sum(test_fa) / float(len(test_fa))
    test_sfar[i] = sum(test_sfa) / float(len(test_sfa))
    test_frr[i] = sum(test_fr) / float(len(test_fr))

  #import ipdb; ipdb.set_trace()

  far_w = [(1-omega[i]) * test_far[i] + omega[i] * test_sfar[i] for i in range(len(omega))]
  hter = [(far_w[i] + test_frr[i]) / 2 for i in range(len(omega))]
  
  from scipy import integrate
  aue = integrate.cumtrapz(hter, omega)

  pp = PdfPages(args.output)
  fig = mpl.figure()
  
  mpl.plot(omega, 100. * numpy.array(hter), color='red', label = "AND", linewidth=4)
  mpl.plot(omega, 100. * numpy.array(test_sfar), color='blue', label = "AND", linewidth=4)
      
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
  f.set('sfar', numpy.array(test_sfar))
  del f
  
  sys.stdout.write("AUE = %.4f \n" % aue[-1]) # for indexing purposes, aue is cumulative integration
  
if __name__ == "__main__":
  main()




