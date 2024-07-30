#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a DeepIM network on an image database."""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import random
import scipy.io
from torchinfo import summary
from tqdm import tqdm
from sklearn.preprocessing import normalize
import _init_paths
from fcn.test_dataset import test_segnet, clustering_features, filter_labels_depth, crop_rois, match_label_crop
from utils.evaluation import multilabel_metrics
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_dataset
import networks
from fcn.test_common import _vis_minibatch_segmentation
from featupxdepth import networkFxD, mem_usage

if __name__ == '__main__':
    cfg_from_file("/content/UnseenObjectClustering/experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml")
    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
    np.random.seed(cfg.RNG_SEED)
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    print('GPU device {:d}'.format(cfg.gpu_id))
    if cfg.TEST.VISUALIZE:
        shuffle = True
        np.random.seed()
    else:
        shuffle = False
    cfg.MODE = 'TEST'
    dataset = get_dataset("ocid_object_test")
    worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
    num_workers = 1
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=shuffle,
        num_workers=num_workers, worker_init_fn=worker_init_fn)
    print('Use dataset `{:s}` for training'.format(dataset.name))
    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        dataset._intrinsic_matrix = K    
    output_dir = get_output_dir(dataset, None)
    print('Output will be saved to `{:s}`\n\n'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    DEVICE='cuda'
    network = networkFxD()
    metrics_all=[]
    metrics_all_refined = []

    for i, sample in enumerate(dataloader):
        image = sample['image_color']
        rgb = sample['rgb']
        depth = None
        label = sample['label']
        print(f"\n{image.shape=}\t{rgb.shape=}\t{label.shape=}\n")

        network.load_dino(DEVICE)
        network.load_featup("/content/UnseenObjectClustering/data/checkpoints/dinov2_jbu_stack_cocostuff.ckpt", DEVICE)
        mem_usage(DEVICE)
        lr_feats_rgb = network.dino(rgb.to(DEVICE))
        hr_feats_rgb = network.featupUpsamplper(lr_feats_rgb.to(DEVICE), rgb.to(DEVICE))
        rgb.detach().cpu()
        if hr_feats_rgb.shape[2] != rgb.shape[2]:
            hr_feats_rgb = torch.nn.functional.interpolate(hr_feats_rgb, rgb.shape[2:], mode="bilinear").detach().cpu()
        normalized_hr_features = network.unit_vec_feats(hr_feats_rgb)
        
        del lr_feats_rgb
        del hr_feats_rgb
        torch.cuda.empty_cache()
        mem_usage(DEVICE)

        out_label, selected_pixels = clustering_features(normalized_hr_features, num_seeds=100)
        out_label_refined = None
        rgb_crop, out_label_crop, rois, depth_crop = crop_rois(rgb, out_label.clone(), depth)
        print(f"\n{rgb_crop.shape=}\t{out_label_crop.shape=}\n")
        
        if rgb_crop.shape[0] > 0:
            
            lr_feats_rgb_crop = network.dino(rgb_crop.to(DEVICE))
            network.del_dino()
            mem_usage(DEVICE)
            hr_feats_rgb_crop = network.featupUpsamplper(lr_feats_rgb_crop.to(DEVICE), rgb_crop.to(DEVICE)).detach().cpu()
            del lr_feats_rgb_crop
            rgb_crop.detach().cpu()
            torch.cuda.empty_cache()
            network.del_featup()
            mem_usage(DEVICE)
            if hr_feats_rgb_crop.shape[2] != rgb_crop.shape[2]:
                hr_feats_rgb_crop = torch.nn.functional.interpolate(hr_feats_rgb_crop, rgb_crop.shape[2:], mode="bilinear").detach().cpu()
            normalize_hr_features_crop = network.unit_vec_feats(hr_feats_rgb_crop)
            labels_crop, selected_pixels_crop = clustering_features(normalize_hr_features_crop)
            out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.to(DEVICE), out_label_crop, rois, depth_crop)
        
        _vis_minibatch_segmentation(rgb, depth, label, out_label, out_label_refined, normalized_hr_features, 
                selected_pixels=selected_pixels, bbox=None)
        
        mem_usage(DEVICE)
        break

    # # sum the values with same keys
    # print('========================================================')
    # result = {}
    # num = len(metrics_all)
    # print('%d images' % num)
    # print('========================================================')
    # for metrics in metrics_all:
    #     for k in metrics.keys():
    #         result[k] = result.get(k, 0) + metrics[k]

    # for k in sorted(result.keys()):
    #     result[k] /= num
    #     print('%s: %f' % (k, result[k]))

    # print('%.6f' % (result['Objects Precision']))
    # print('%.6f' % (result['Objects Recall']))
    # print('%.6f' % (result['Objects F-measure']))
    # print('%.6f' % (result['Boundary Precision']))
    # print('%.6f' % (result['Boundary Recall']))
    # print('%.6f' % (result['Boundary F-measure']))
    # print('%.6f' % (result['obj_detected_075_percentage']))

    # print('========================================================')
    # print(result)
    # print('====================Refined=============================')

    # result_refined = {}
    # for metrics in metrics_all_refined:
    #     for k in metrics.keys():
    #         result_refined[k] = result_refined.get(k, 0) + metrics[k]

    # for k in sorted(result_refined.keys()):
    #     result_refined[k] /= num
    #     print('%s: %f' % (k, result_refined[k]))
    # print(result_refined)
    # print('========================================================')
