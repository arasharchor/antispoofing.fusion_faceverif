#!/usr/bin/env /idiap/group/torch5spro/nightlies/last/install/linux-x86_64-release/bin/python
#Ivana Chingovska <ivana.chingovska@idiap.ch>
#Tue Oct 16 15:51:38 CEST 2012

'''This script takes score files in 4-column format (which are output of FaceRecLib) and arranges the scores as in the directory structure of Replay-Attack'''

import bob.io.video
import numpy, os
import sys

import antispoofing

from antispoofing.utils.db import *

class Score:

  def __init__(self, model):
    self.model = model
    self.total_keys = 0
    self.fileobjs = []
    self.scores_dict = {}

  def add_scorelist(self, fileobj, num_frames):
    self.fileobjs.append(fileobj)
    scores = numpy.ndarray((num_frames, 1), dtype='float64') 
    scores[:] = numpy.NAN
    self.scores_dict[str(fileobj.make_path())] = scores

  def update_scorelist(self, fname, frame_index, score):
    self.scores_dict[fname][frame_index-1] = score

  def save_score_files(self, output_dir, fv_protocol):
    for obj in self.fileobjs:
      if fv_protocol == 'licit': # licit face verification protocol, scores need to be saved in subdirectories with model ids
        obj.save(self.scores_dict[str(obj.make_path())], os.path.join(output_dir, fv_protocol, 'client%03d' % int(self.model)))
      else: # spoof face verification protocol
        obj.save(self.scores_dict[str(obj.make_path())], os.path.join(output_dir, fv_protocol))


def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
  INPUT_DIR = os.path.join(basedir, 'database')
  
  import argparse

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  
  parser.add_argument('infile', type=str, help='The input score file')

  parser.add_argument('outputdir', metavar='DIR', type=str, help='Directory where the extracted scores will be stored')
  
  parser.add_argument('-v', '--input-dir', metavar='DIR', type=str, dest='inputdir', default=INPUT_DIR, help='Base directory containing the videos to be treated by this procedure (defaults to "%(default)s")')
  
  parser.add_argument('-t', '--fv_protocol', metavar='fv_protocol', type=str, dest="fv_protocol", default='licit', help='Specifies whether the score file contains scores for the licit protocol or for spoofing attacks (defaults to "%(default)s")', choices=('licit','spoof'))

  parser.add_argument('-s', '--scoresubset', metavar='scoresubset', type=str, dest="scoresubset", default='devel', help='Specifies whether the score file contains scores for development (devel), test (eval) or train (train) set (defaults to "%(default)s")', choices=('devel','test', 'train'))

  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()

  #######################
  # Loading the database objects
  #######################
  database = args.cls(args)

  name_conv_dict = {'licit':'real', 'spoof':'attack'}

  if args.fv_protocol == 'licit': # licit protocol, only real accesses are in play
    if args.scoresubset == 'train':
      objects = database.get_train_data()[0]
    elif args.scoresubset == 'devel':
      objects = database.get_devel_data()[0] 
    else:
      objects = database.get_test_data()[0]  
      
    #objects = db.objects(cls=name_conv_dict[args.fv_protocol], protocol=args.protocol, groups=args.scoresubset)
  else: # spoof protocol, we need real accesses + attacks
    if args.scoresubset == 'train':
      objects = database.get_train_data()[0] + database.get_train_data()[1]
    elif args.scoresubset == 'devel':
      objects = database.get_devel_data()[0] + database.get_devel_data()[1]
    else:
      objects = database.get_test_data()[0] + database.get_test_data()[1]
    #objects = db.objects(protocol=args.protocol, groups=args.scoresubset)

  def find_object(fname, objects_list):
    for obj in objects_list:
      if str(obj.make_path()) == fname:
        return obj
    return None

  models_dict = {} 

  # load the 4 column file 
  lines = file(args.infile).readlines()
  line_counter = 0

  for line in lines:
    line_counter += 1
    sys.stdout.write("Processing line [%d/%d] \n" % (line_counter, len(lines)))
    words = line.split()
    model = words[0] # the claimed ID # was words[1] for UBMGMM, EBGM, LGBPHS
    sepind = words[2].find('/')
    frame_index = int(words[2].split('/')[0])
    filestem = words[2][sepind+1:len(words[2])]   
  
    obj = find_object(filestem, objects)
    if obj == -1: raise Exception("File " + filestem + " not found in the database\n")

    if models_dict.has_key(model) == False:
      models_dict[model] = Score(model)

    if models_dict[model].scores_dict.has_key(str(obj.make_path())) == False:
      input = bob.io.video.reader(obj.videofile(args.inputdir)) # read the video to see the total number of frames it contains
      models_dict[model].add_scorelist(obj, input.number_of_frames) 
     
    models_dict[model].update_scorelist(str(obj.make_path()), frame_index, words[3])  
  
  for key, item in models_dict.items():
    item.save_score_files(os.path.join(args.outputdir), args.fv_protocol)
    
if __name__ == '__main__':
  main()

