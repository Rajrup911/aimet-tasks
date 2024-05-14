#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

from modules.user import *

#MCW
'''
if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to train with. No Default',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default=os.path.expanduser("~") + '/logs/' +
      datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
      help='Directory to put the predictions. Default: ~/logs/date+time'
  )
  parser.add_argument(
      '--model', '-m',
      type=str,
      required=True,
      default=None,
      help='Directory to get the trained model.'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset", FLAGS.dataset)
  print("log", FLAGS.log)
  print("model", FLAGS.model)
  print("----------\n")
  print("Commit hash (training version): ", str(
      subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
  print("----------\n")

  # open arch config file
'''
def validation(dataset, log, model, config):
    try:
      print("Opening arch config file from %s" % model)
      ARCH = yaml.safe_load(open(model + "/arch_cfg.yaml", 'r'))
    except Exception as e:
      print(e)
      print("Error opening arch yaml file.")
      quit()
  
    # open data config file
    try:
      print("Opening data config file from %s" % model)
      DATA = yaml.safe_load(open(model + "/data_cfg.yaml", 'r'))
    except Exception as e:
      print(e)
      print("Error opening data yaml file.")
      quit()
  
    # create log folder
    try:
      if os.path.isdir(log):
        shutil.rmtree(log)
      os.makedirs(log)
      
      for seq in DATA["split"]["valid"]:
        seq = '{0:02d}'.format(int(seq))
        print("valid", seq)
        os.makedirs(os.path.join(log, "sequences", seq))
        os.makedirs(os.path.join(log, "sequences", seq, "predictions"))
        
      if not os.path.exists(os.path.join(log, "sequences")):
        os.makedirs(os.path.join(log, "sequences"))
      for seq in DATA["split"]["train"]:
        seq = '{0:02d}'.format(int(seq))
        print("train", seq)
        os.makedirs(os.path.join(log, "sequences", seq))
        os.makedirs(os.path.join(log, "sequences", seq, "predictions"))
      '''
      for seq in DATA["split"]["test"]:
        seq = '{0:02d}'.format(int(seq))
        print("test", seq)
        os.makedirs(os.path.join(log, "sequences", seq))
        os.makedirs(os.path.join(log, "sequences", seq, "predictions"))
      '''
    except Exception as e:
      print(e)
      print("Error creating log directory. Check permissions!")
      raise
  
    except Exception as e:
      print(e)
      print("Error creating log directory. Check permissions!")
      quit()
  
    # does model folder exist?
    if os.path.isdir(model):
      print("model folder exists! Using model from %s" % (model))
    else:
      print("model folder doesnt exist! Can't infer...")
      quit()
  
    # create user and infer dataset
    user = User(ARCH, DATA, dataset, log, model, config)
    user.ptq()