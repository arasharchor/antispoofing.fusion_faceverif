#!/usr/bin/env python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Sun 11 Oct 21:22:02 CEST 2015

"""
This script converts scores which are saved as per Replay-Attack directory structure, into 4-column score files. Note that the input scores need to be separated by three classes: real accesses, zero-effort impostors and spoofing attacks. Can be used for face verification or anti-spoofing scores alike.
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

  parser.add_argument('scoresdir', type=str, help='Base directory containing the scores of an antispoofing system (without the protocol dir)')
  parser.add_argument('outputdir', type=str, help='Output directory to save the 4-columns score files)')
 
  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()
 
  #######################
  # Loading the database objects
  #######################
  database = args.cls(args)
  
  #import ipdb; ipdb.set_trace()
  # read faceverif and antispoofing scores for all samples

  subset_dict = {'devel':'dev', 'test':'eval'}
  for subset in ('devel', 'test'):
    licit_scores, licit_labels, _ = gather_fvas_clsp_scores(database, subset, fv_dirs=[], as_dirs = (args.scoresdir,), binary_labels=False, fv_protocol='licit', normalize=False)
    spoof_scores, spoof_labels, _ = gather_fvas_clsp_scores(database, subset, fv_dirs=[], as_dirs = (args.scoresdir,), binary_labels=False, fv_protocol='spoof', normalize=False)

    fusion_utils.save_fused_scores(licit_scores, licit_labels, args.outputdir, protocol = 'licit', subset = subset_dict[subset])
    fusion_utils.save_fused_scores(spoof_scores, spoof_labels, args.outputdir, protocol = 'spoof', subset = subset_dict[subset])

if __name__ == "__main__":
  main()
