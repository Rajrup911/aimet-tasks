#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import sys
import numpy as np

from modules.segmentator import *
from modules.ioueval import iouEval
from common.laserscan import SemLaserScan
from modules.trainer import *
from postproc.KNN import KNN

from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_torch.model_preparer import prepare_model
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.auto_quant import AutoQuant
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_common.defs import QuantScheme
import aimet_torch


class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir, config):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.config = config

    # get the data
    parserModule = imp.load_source("parserModule", 'src/dataset/' + self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"], 
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.ARCH,
                               self.parser.get_n_classes(),
                               self.modeldir)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.dummy_input = torch.rand(1, 5, 64, 2048).to(self.device)
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()
      
    self.input_shape=self.config['input_shape']
    self.dummy_input = torch.rand(self.input_shape).cuda()
  
  def infer(self):
    # do train set
    #self.infer_subset(loader=self.parser.get_train_set(),
    #                 to_orig_fn=self.parser.to_original)

    # do valid set
    self.infer_subset(self.model,loader=self.parser.get_valid_set(),
                      to_orig_fn=self.parser.to_original)
                      
    # do test set
    #self.infer_subset(loader=self.parser.get_test_set(),
    #                  to_orig_fn=self.parser.to_original)
    
    self.evaluate_semantics()

    print('Finished Infering')

    return
    

  def ptq(self):
    print("\nModel Validate 1")
    ModelValidator.validate_model(self.model, model_input=self.dummy_input)
    
    print("\nPrepare Model") 
    self.model = prepare_model(self.model.eval())
    
    print("\nModel Validate 2")
    ModelValidator.validate_model(self.model, model_input=self.dummy_input)
    self.infer()
    
    equalize_model(self.model, self.input_shape)
    
    dataloader=self.parser.get_valid_set()
          
    params = AdaroundParameters(data_loader=dataloader, num_batches= 32, default_num_iterations=10000)
    
    self.model = Adaround.apply_adaround(model = self.model, dummy_input = self.dummy_input, params = params, 
                                         path=self.config['exports_path'], filename_prefix='Adaround', default_param_bw=8, default_quant_scheme="tf_enhanced")
         
    kwargs = {
        "quant_scheme": QuantScheme.training_range_learning_with_tf_init,
        "default_param_bw": self.config["quantization_configuration"]["param_bw"],
        "default_output_bw": self.config["quantization_configuration"]["output_bw"], 
        "dummy_input": self.dummy_input,
    }
    
    sim = QuantizationSimModel(self.model, **kwargs)

    sim.set_and_freeze_param_encodings(encoding_path=self.config['exports_path']+'/Adaround.encodings')
    print("set_and_freeze_param_encodings finished!")


    sim.compute_encodings(self.infer_subset, forward_pass_callback_args=self.parser.get_calib_set())
    print("done")
    self.model=sim.model
    self.infer()
    
    trainer = Trainer(self.ARCH, self.DATA, self.datadir, "logger", sim.model)
    sim.model = trainer.train()
    self.model= sim.model
    print("\nQAT Inference:")
    self.infer()

  def infer_subset(self,model, loader, to_orig_fn=None):
    to_orig_fn=self.parser.to_original
    # switch to evaluate mode 
    self.model.eval()
    

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        # print(proj_in, proj_mask, path_seq, path_name, p_x, p_y, proj_range, unproj_range, npoints)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          proj_mask = proj_mask.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        # compute output
        proj_output = model(proj_in, proj_mask)
        proj_argmax = proj_output[0].argmax(dim=0)

        if self.post:
          # knn postproc
          unproj_argmax = self.post(proj_range,
                                    unproj_range,
                                    proj_argmax,
                                    p_x,
                                    p_y)
        else:
          # put in original pointcloud using indexes
          unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
          torch.cuda.synchronize()

        # print("Infered seq", path_seq, "scan", path_name,
        #       "in", time.time() - end, "sec")
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)
        
  def evaluate_semantics(self):
    DATA = self.DATA
    
    class_strings = DATA["labels"]
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    nr_classes = len(class_inv_remap)
  
    # make lookup table for mapping
    maxkey = max(class_remap.keys())
    
    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(class_remap.keys())] = list(class_remap.values())
    # print(remap_lut)
  
    # create evaluator
    ignore = []
    for cl, ign in class_ignore.items():
      if ign:
        x_cl = int(cl)
        ignore.append(x_cl)
        print("Ignoring xentropy class ", x_cl, " in IoU evaluation")
  
    # create evaluator
    device = torch.device("cpu")
    evaluator = iouEval(nr_classes, device, ignore)
    evaluator.reset()

    test_sequences = DATA["split"]["valid"]
    
    scan_names = []
    for sequence in test_sequences:
      sequence = '{0:02d}'.format(int(sequence))
      scan_paths = os.path.join(self.datadir, "sequences",
                                str(sequence), "velodyne")

      seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
      seq_scan_names.sort()
      scan_names.extend(seq_scan_names)

    label_names = []
    for sequence in test_sequences:
      sequence = '{0:02d}'.format(int(sequence))
      label_paths = os.path.join(self.datadir, "sequences",
                                 str(sequence), "labels")
      # populate the label names
      seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_paths)) for f in fn if ".label" in f]
      seq_label_names.sort()
      label_names.extend(seq_label_names)

    pred_names = []
    for sequence in test_sequences:
      sequence = '{0:02d}'.format(int(sequence))
      pred_paths = os.path.join(self.logdir, "sequences",
                                sequence, "predictions")
      # populate the label names
      seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
      seq_pred_names.sort()
      pred_names.extend(seq_pred_names)

    assert(len(label_names) == len(pred_names))
  
    print("\nEvaluating sequences: ")
    # open each file, get the tensor, and make the iou comparison
    for scan_file, label_file, pred_file in zip(scan_names, label_names, pred_names):
      # print("evaluating label ", label_file, "with", pred_file)
      # open label
      label = SemLaserScan(project=False)
      label.open_scan(scan_file)
      label.open_label(label_file)
      u_label_sem = remap_lut[label.sem_label]  # remap to xentropy format
      

      # open prediction
      pred = SemLaserScan(project=False)
      pred.open_scan(scan_file)
      pred.open_label(pred_file)
      u_pred_sem = remap_lut[pred.sem_label]  # remap to xentropy format
      

      # add single scan to evaluation
      evaluator.addBatch(u_pred_sem, u_label_sem)
  
    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()
  
    print('Validation set:\n'
          'Acc avg {m_accuracy:.3f}\n'
          'IoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy,
                                           m_jaccard=m_jaccard))
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
      if i not in ignore:
        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))
  
    # print for spreadsheet
    print("*" * 80)
    print("below can be copied straight for paper table")
    for i, jacc in enumerate(class_jaccard):
      if i not in ignore:
        sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
        sys.stdout.write(",")
    sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
    sys.stdout.write(",")
    sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
    sys.stdout.write('\n')
    sys.stdout.flush()