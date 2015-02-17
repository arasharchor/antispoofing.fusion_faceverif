#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Mon Feb 18 16:12:41 CET 2013

"""Plot different plots as demanded by the user
"""

import os
import sys
from matplotlib import rc
rc('text',usetex=1)
import matplotlib.pyplot as mpl
import bob.measure
import numpy as np
import argparse

import matplotlib.font_manager as fm

def calc_pass_rate(threshold, attacks):
  """Calculates the rate of attacks that are after a certain threshold"""

  return sum(1 for i in attacks if i >= threshold)/float(attacks.size)

def main():

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('baseline_dev', metavar='FILE', type=str,
      default="", help='Name of the scores file (4-column) containing the scores for the baseline face verification i.e. LICIT protocol (development set)')
  parser.add_argument('baseline_test', metavar='FILE', type=str,
      default="", help='Name of the scores file (4-column) containing the scores for the baseline face verification i.e. LICIT protocol (test set)')
  parser.add_argument('overlay_dev', metavar='FILE', type=str,
      default="", help='Name of the scores file (4-column) containing the scores for the overlaid negatives i.e. SPOOF protocol (spoofing attacks; development set)')
  parser.add_argument('overlay_test', metavar='FILE', type=str,
      default="", help='Name of the scores file (4-column) containing the scores for the overlaid negatives i.e. SPOOF protocol (spoofing attacks; test set)')
  parser.add_argument('-b', '--type-of-threshold', metavar='STR', type=str,
      dest='threshold', default="hter", help='Type of threshold', choices=('eer', 'hter'))
  parser.add_argument('-t', '--title', metavar='STR', type=str,
      dest='title', default="", help='Plot title')
  parser.add_argument('-o', '--output', metavar='FILE', type=str,
      default='plots.pdf', dest='output',
      help='Set the name of the output file (defaults to "%(default)s")')
  parser.add_argument('-i', '--demandedplot', metavar='STR', type=int, dest='demandedplot', default=1, help='The number of plot that is needed. Possible options: 1-DET for LICIT protocol; 2-DET for LICIT and SPOOF protocol; 3-DET for LICIT and SPOOF protocol and threshold line; 4-score histogram for LICIT; 5-score histogram for LICIT and threshold line; 6-score histogram for LICIT and SPOOF and threshold line; 7-score histogram for LICIT and SPOOF and probability of success line.')

  args = parser.parse_args()

  if args.threshold == 'eer':
    report_text = "EER"
  else:
    report_text = "Min.HTER"

  [base_neg, base_pos] = bob.measure.load.split_four_column(args.baseline_test)
  [over_neg, over_pos] = bob.measure.load.split_four_column(args.overlay_test)
  [base_neg_dev, base_pos_dev] = bob.measure.load.split_four_column(args.baseline_dev)
  [over_neg_dev, over_pos_dev] = bob.measure.load.split_four_column(args.overlay_dev)

  from matplotlib.backends.backend_pdf import PdfPages

  outdir = os.path.dirname(args.output)
  if outdir and not os.path.exists(outdir): os.makedirs(outdir)

  pp = PdfPages(args.output)


  # Plot 1: DET baseline only
  # -------------------------
  if args.demandedplot == 1:
    fig = mpl.figure()
    mpl.rcParams.update({'font.size': 16})
    bob.measure.plot.det(base_neg, base_pos, 100, color='blue', linewidth=2)
    bob.measure.plot.det_axis([0.1, 99, 0.1, 99])
    #mpl.title("DET: baseline verification system")
    mpl.xlabel("False Rejection Rate (\\%)")
    mpl.ylabel("False Acceptance Rate (\\%)")
    mpl.grid()
    pp.savefig()


  # Plot 2: DET (LICIT + SPOOF PROTOCOL)
  # ------------------------------------

  if args.demandedplot == 2:
    fig = mpl.figure()
    bob.measure.plot.det(base_neg, base_pos, 100, color='blue', label="LICIT", linewidth=2)
    bob.measure.plot.det(over_neg, over_pos, 100, color='red', label="SPOOF", linewidth=2)
    bob.measure.plot.det_axis([0.1, 99, 0.1, 99])
    #mpl.title("DET: LICIT and overlaid SPOOF protocol")
    mpl.xlabel("False Rejection Rate (\%)")
    mpl.ylabel("False Acceptance Rate (\%)")
    mpl.legend()
    mpl.grid()
    pp.savefig()


  # Plot 3: DET (LICIT + SPOOF PROTOCOL) + threshold at fixed FRR
  # ------------------------------------

  if args.demandedplot == 3:
    fig = mpl.figure()
    mpl.rcParams.update({'font.size': 16})
    bob.measure.plot.det(base_neg, base_pos, 100, color='blue', label="licit", linewidth=3)
    bob.measure.plot.det(over_neg, over_pos, 100, color='red', label="spoof", linewidth=3)
    bob.measure.plot.det_axis([0.01, 99, 0.01, 99])
    ax = mpl.subplot(111)
    mpl.rcParams.update({'font.size': 18}) 
    if args.threshold == 'eer':
      thres_baseline = bob.measure.eer_threshold(base_neg_dev, base_pos_dev)
    else:
      thres_baseline = bob.measure.min_hter_threshold(base_neg_dev, base_pos_dev)

    axlim = mpl.axis()

    farfrr_licit = bob.measure.farfrr(base_neg, base_pos, thres_baseline) # calculate test frr @ EER (licit protocol)
    farfrr_spoof = bob.measure.farfrr(over_neg, over_pos, thres_baseline) # calculate test frr @ EER (spoof protocol)
    farfrr_licit_det = [bob.measure.ppndf(i) for i in farfrr_licit] # find the FAR and FRR values that need to be plotted on normal deviate scale
    farfrr_spoof_det = [bob.measure.ppndf(i) for i in farfrr_spoof] # find the FAR and FRR values that need to be plotted on normal deviate scale
    mpl.axvline(x=farfrr_licit_det[1], ymin=axlim[2], ymax=axlim[3], color='k', linestyle='--', linewidth = 3, label="FRR = %.2f\\%%" % (farfrr_licit[1]*100)) # vertical FRR threshold
    #mpl.axvline(x=farfrr_licit_det[1], ymin=axlim[2], ymax=axlim[3], color='k', linestyle='--', linewidth = 3, label="FRR @ EER") # vertical FRR threshold
    mpl.plot(farfrr_licit_det[1], farfrr_licit_det[0], 'bo', markersize=7) # FAR point, licit protocol
    mpl.plot(farfrr_spoof_det[1], farfrr_spoof_det[0], 'ro', markersize=7) # FAR point, spoof protocol

    # annotate the FAR points
    xyannotate_licit = [bob.measure.ppndf(1.4*farfrr_licit[0]), bob.measure.ppndf(1.2*farfrr_licit[1])]
    xyannotate_spoof = [bob.measure.ppndf(1*farfrr_spoof[0]), bob.measure.ppndf(1.2*farfrr_licit[1])]
    #mpl.annotate('FAR @ operating point', xy=(farfrr_licit_det[1], farfrr_licit_det[0]),  xycoords='data', xytext=(xyannotate_licit[1], xyannotate_licit[0]), color='blue')
    #mpl.annotate('SFAR @ operating point', xy=(farfrr_spoof_det[1], farfrr_spoof_det[0]),  xycoords='data', xytext=(xyannotate_spoof[1], xyannotate_spoof[0]), color='red')
    mpl.annotate('FAR = %.2f\\%%' % (farfrr_licit[0]*100), xy=(farfrr_licit_det[1], farfrr_licit_det[0]),  xycoords='data', xytext=(xyannotate_licit[1], xyannotate_licit[0]), color='blue', size='large')
    mpl.annotate('SFAR = %.2f\\%%' % (farfrr_spoof[0]*100), xy=(farfrr_spoof_det[1], farfrr_spoof_det[0]),  xycoords='data', xytext=(xyannotate_spoof[1], xyannotate_spoof[0]), color='red', size='large')

    mpl.tick_params(axis='both', which='major', labelsize=4)

    #mpl.title("DET: LICIT and SPOOF protocol")
    mpl.xlabel("False Rejection Rate (\%)")
    mpl.ylabel("False Acceptance Rate (\%)")
    mpl.legend(prop=fm.FontProperties(size=16))
    
    for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(6) 
    for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(6) 
    
    mpl.grid()
    pp.savefig()

  
  # Plot 4: Score histograms (LICIT only)
  # --------------------------

  if args.demandedplot == 4:
    fig = mpl.figure()
    ax1 = mpl.subplot(111)
    mpl.rcParams.update({'font.size': 16})
  
    mpl.hist(base_neg, bins=10, color='red', alpha=0.5, label="Impostors", normed=True)
    mpl.hist(base_pos, bins=10, color='blue', alpha=0.5, label="Valid Users", normed=True)

    mpl.xlabel("Verification Scores")
    mpl.ylabel("Normalized Count")

    mpl.legend()

    #mpl.title("Score distributions of baseline verification system")
    mpl.grid()
    pp.savefig()

  # Plot 5: Score histograms + Threshold (LICIT only)
  # --------------------------------------

  if args.demandedplot == 5:
    fig = mpl.figure()
    ax1 = mpl.subplot(111)
  
    mpl.hist(base_neg, bins=10, color='red', alpha=0.5,
      label="Impostors @ test", normed=True)
    mpl.hist(base_pos, bins=10, color='blue', alpha=0.5, label="Valid Users @ test",
      normed=True)

  
    axlim = mpl.axis()

    if args.threshold == 'eer':
      thres_baseline = bob.measure.eer_threshold(base_neg_dev, base_pos_dev)
    else:
      thres_baseline = bob.measure.min_hter_threshold(base_neg_dev, base_pos_dev)

    # plot the line
    mpl.axvline(x=thres_baseline, ymin=0, ymax=1, linewidth=2, color='green', linestyle='--', label="EER threshold @ dev")
  
    mpl.xlabel("Verification Scores")
    mpl.ylabel("Normalized Count")
    
    mpl.legend(prop=fm.FontProperties(size=10))

    mpl.title("Score distributions of baseline verification system")
    mpl.grid()
    pp.savefig()

  # Plot 6: Score histograms + threshold (LICIT + SPOOF)
  # ------------------------------------------
  if args.demandedplot == 6:
    fig = mpl.figure()
    ax1 = mpl.subplot(111)
  
    mpl.rcParams.update({'font.size': 16})
    mpl.hist(base_neg, bins=10, color='red', alpha=0.5,
      label="Impostors", normed=True)
    mpl.hist(base_pos, bins=10, color='blue', alpha=0.5,
      label="Genuine Accesses", normed=True)
    mpl.hist(over_neg, bins=10, color='black', alpha=0.5,
      label="Spoofing Attacks", normed=True)

    axlim = mpl.axis()

    if args.threshold == 'eer':
      thres_baseline = bob.measure.eer_threshold(base_neg_dev, base_pos_dev)
    else:
      thres_baseline = bob.measure.min_hter_threshold(base_neg_dev, base_pos_dev)

    # plot the line
    #mpl.axvline(x=thres_baseline, ymin=0, ymax=1, linewidth=2, color='green', linestyle='--', label="%s @ dev" % report_text)
  
    mpl.xlabel("Verification Scores")
    mpl.ylabel("Normalized Count")
    mpl.legend(prop=fm.FontProperties(size=14))

    # Reports what is the canonical value for a particular attack
  
    #mhter_thres_baseline = bob.measure.min_hter_threshold(base_neg_dev, base_pos_dev)
    #pass_rate = calc_pass_rate(thres_baseline, over_neg)
    #print "Attack Success Rate on %s: %.2f%%" % (report_text, 100.*pass_rate)

    #mpl.title("Score distributions of baseline verification system")
    frame1 = mpl.gca()
    #frame1.axes.get_xaxis().set_ticks([])
    #frame1.axes.get_yaxis().set_ticks([])

    '''
    for xlabel_i in frame1.axes.get_xticklabels():
      xlabel_i.set_visible(False)
      xlabel_i.set_fontsize(0.0)
    for xlabel_i in frame1.axes.get_yticklabels():
      xlabel_i.set_fontsize(0.0)
      xlabel_i.set_visible(False)  
    '''    
    mpl.grid()
    pp.savefig()

  # Plot 7: Score histogram (LICIT + SPOOF) + Probability of Success line
  # --------------------------------------------

  if args.demandedplot == 7:
    fig = mpl.figure()
    ax1 = mpl.subplot(111)

    mpl.rcParams.update({'font.size': 18}) 
    mpl.hist(base_neg, bins=10, color='red', alpha=0.8,
      label="Impostors", normed=True) # 0.5
    mpl.hist(base_pos, bins=20, color='blue', alpha=0.6,
      label="Valid Users", normed=True)
    mpl.hist(over_neg, bins=20, color='black', alpha=0.4,
      label="Spoofs", normed=True) #0.5

    axlim = mpl.axis()
    
    if args.threshold == 'eer':
      thres_baseline = bob.measure.eer_threshold(base_neg_dev, base_pos_dev)
    else:
      thres_baseline = bob.measure.min_hter_threshold(base_neg_dev, base_pos_dev)
    pass_rate = calc_pass_rate(thres_baseline, over_neg)
    #print "Attack Success Rate on %s: %.2f%%" % (report_text, 100.*pass_rate)
    
    threscolor = "green"
    mpl.axvline(x=thres_baseline, ymin=0, ymax=1, linewidth=3,
      color='green', linestyle='--', label="%s @ dev" % report_text)
  
    mpl.xlabel("Verification Scores")
    mpl.ylabel("Normalized Count")
    mpl.legend(prop=fm.FontProperties(size=16), loc = 1)

    # scan the range of scores, put an axis on the right with spoofing success
    # probabilities that depend on the threshold
    ntick = 100
    step = (axlim[1] - axlim[0])/float(ntick)
    thres = [(k*step)+axlim[0] for k in range(ntick)]
    mix_prob_y = []
    for k in thres: mix_prob_y.append(100.*calc_pass_rate(k, over_neg))

    prob_ax = ax1.twinx() 
    mpl.plot(thres, mix_prob_y, color='green', label="SFAR", linewidth=3)
    prob_ax.set_ylabel("SFAR (\%)", color='green')
    for tl in prob_ax.get_yticklabels(): tl.set_color('green')

    # Inprint the threshold one on the plot:
    prob_ax.plot(thres_baseline, 100.*pass_rate, 'o', color=threscolor)
    #prob_ax.text(thres_baseline-(thres_baseline-axlim[0])/2.5, 95.*pass_rate, '%.1f%%' % (100.*pass_rate,), color='green')
    #prob_ax.annotate('%.2f\\%%' % (100.*pass_rate,), xy=(thres_baseline, 100.*pass_rate),  xycoords='data', xytext=(0.5, 0.3), textcoords='axes fraction', color=threscolor, size='large', arrowprops=dict(facecolor='black', shrink=0.05, width=2), horizontalalignment='right', verticalalignment='top',)
    prob_ax.text(thres_baseline+(abs(thres_baseline-axlim[1]))/15, 95.*pass_rate, '%.2f\\%%' % (100.*pass_rate,), color=threscolor, size='large')
    #prob_ax.text(thres_baseline-(thres_baseline-axlim[0])/2.3, 95.*pass_rate, 'SFAR @ \noperating point', color='green',size='small',multialignment='right')

    #mpl.title("Score distributions of baseline verification system")
    mpl.grid()
    pp.savefig()

    
  pp.close() # close multi-page PDF writer
 

if __name__ == '__main__':
  main()
