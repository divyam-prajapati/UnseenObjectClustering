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
from featupxdepth import networkFxD

def parse_args():
    """
    Parse input arguments
    ./tools/test_net.py \
    --network seg_resnet34_8s_embedding \
    --pretrained output/tabletop_object/tabletop_object_train/seg_resnet34_8s_embedding_cosine_color_sampling_epoch_$2.checkpoint.pth  \
    --dataset ocid_object_test \
    --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml \
    --pretrained_crop output/tabletop_object/tabletop_object_train/seg_resnet34_8s_embedding_cosine_color_crop_sampling_epoch_$3.checkpoint.pth
    
    py -W ignore ./tools/test_net.py --network seg_resnet34_8s_embedding --pretrained data/checkpoints/seg_resnet34_8s_embedding_cosine_color_sampling_epoch_16.checkpoint.pth --dataset ocid_object_test --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml --pretrained_crop data/checkpoints/seg_resnet34_8s_embedding_cosine_color_crop_sampling_epoch_16.checkpoint.pth
    py -W ignore ./tools/test_net.py --dataset ocid_object_test --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml
    """
    parser = argparse.ArgumentParser(description='Test a Unseen Clustering Network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--pretrained_crop', dest='pretrained_crop',
                        help='initialize with pretrained checkpoint for crops',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    # print('GPU device {:d}'.format(args.gpu_id))
    # prepare dataset
    if cfg.TEST.VISUALIZE:
        shuffle = True
        np.random.seed()
    else:
        shuffle = False
    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)
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
    # print('Output will be saved to `{:s}`\n\n'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Import featup_X_depthany Network

    # test network
    # test_segnet(dataloader, network, output_dir, network_crop)
    DEVICE='cuda'
    print(f"\n1. Allocated: {torch.cuda.memory_allocated(DEVICE)/(10**9):.2f} Resereved: {torch.cuda.memory_reserved(DEVICE)/(10**9):.2f}\n")
    network = networkFxD()
    network.send_model_to_device(DEVICE)
    print(f"\n2. Allocated: {torch.cuda.memory_allocated(DEVICE)/(10**9):.2f} Resereved: {torch.cuda.memory_reserved(DEVICE)/(10**9):.2f}\n")
    metrics_all=[]
    metrics_all_refined = []
    
    for i, sample in enumerate(dataloader):
        image = sample['image_color']
        rgb = sample['rgb']
        depth = None
        label = sample['label']
        lr_feats_rgb = network.dino(rgb.to(DEVICE))
        hr_feats_rgb = network.featupUpsamplper(lr_feats_rgb.to(DEVICE), rgb.to(DEVICE))
        hr_feats_rgb = torch.nn.functional.interpolate(hr_feats_rgb, rgb.shape[2:], mode="bilinear").detach().cpu()
        print(f"\n{image.shape=}\t{rgb.shape=}\t{lr_feats_rgb.shape=}\t{hr_feats_rgb.shape=}\n")
        print(f"\n3. Allocated: {torch.cuda.memory_allocated(DEVICE)/(10**9):.2f} Resereved: {torch.cuda.memory_reserved(DEVICE)/(10**9):.2f}\n")
        del lr_feats_rgb
        del rgb
        torch.cuda.empty_cache()
        print(f"\n4. Allocated: {torch.cuda.memory_allocated(DEVICE)/(10**9):.2f} Resereved: {torch.cuda.memory_reserved(DEVICE)/(10**9):.2f}\n")

        # def old_code(): # << Just to hide cmts
        #     out_label, selected_pixels = clustering_features(hr_feats_rgb.detach().cpu(), num_seeds=100)
        #     gt = sample['label'].squeeze().numpy()
        #     prediction = out_label.squeeze().detach().cpu().numpy() 
        #     # print(f"\n{gt.shape=}\t{out_label.shape=}\t{prediction.shape=}\n")
        #     metrics = multilabel_metrics(prediction, gt)
        #     metrics_all.append(metrics)
        #     # print(f"Normal: {metrics}")
        #     out_label_refined = None
        #     rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
        #     # print(f"\n{rgb_crop.shape=}\t{out_label_crop.shape=}\t{depth_crop.shape=}\n")
        #     if rgb_crop.shape[0] > 0:
        #         # features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
        #         lr_feats_rgb_crop = network.dino(rgb_crop.cuda())
        #         hr_feats_rgb_crop = network.featupUpsamplper(lr_feats_rgb_crop.cuda(), rgb_crop.cuda())
        #         if hr_feats_rgb_crop.shape[2] != rgb_crop.shape[2]:
        #             hr_feats_rgb_crop = torch.nn.functional.interpolate(hr_feats_rgb_crop, rgb_crop.shape[2:], mode="bilinear")
        #         labels_crop, selected_pixels_crop = clustering_features(hr_feats_rgb_crop.detach().cpu())
        #         out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)

        #     if out_label_refined is not None:
        #         prediction_refined = out_label_refined.squeeze().detach().cpu().numpy()
        #     else:
        #         prediction_refined = prediction.copy()
        #     # print(f"\n{gt.shape=}\t{prediction_refined.shape=}\t{prediction.shape=}\n")
        #     metrics_refined = multilabel_metrics(prediction_refined, gt)
        #     metrics_all_refined.append(metrics_refined)
        #     print(f"Refined: {metrics_refined}")
        #     _vis_minibatch_segmentation(image, depth, label, out_label, out_label_refined, hr_feats_rgb, 
        #             selected_pixels=selected_pixels, bbox=None)
        
        # UNIT VECTOR
        epsilon = 1e-8
        magnitude = torch.norm(hr_feats_rgb, p=2, dim=1, keepdim=True)
        magnitude = magnitude + epsilon # TO avoid division by 0
        normalized_features = hr_feats_rgb / magnitude
        out_label_norm, selected_pixels_norm = clustering_features(normalized_features, num_seeds=100)
        out_label_refined_norm = None
        rgb_crop_norm, out_label_crop_norm, rois_norm, depth_crop_norm = crop_rois(sample['rgb'], out_label_norm.clone(), depth)

        print(f"\n{rgb_crop_norm.shape=}\t{out_label_crop_norm.shape=}\n")
        print(f"\n5. Allocated: {torch.cuda.memory_allocated(DEVICE)/(10**9):.2f} Resereved: {torch.cuda.memory_reserved(DEVICE)/(10**9):.2f}\n")
        if rgb_crop_norm.shape[0] > 0:
            # features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
            lr_feats_rgb_crop_norm = network.dino(rgb_crop_norm.to(DEVICE))
            print(f"\n6. Allocated: {torch.cuda.memory_allocated(DEVICE)/(10**9):.2f} Resereved: {torch.cuda.memory_reserved(DEVICE)/(10**9):.2f}\n")
            network.del_dino()
            print(f"\n7. Allocated: {torch.cuda.memory_allocated(DEVICE)/(10**9):.2f} Resereved: {torch.cuda.memory_reserved(DEVICE)/(10**9):.2f}\n")
            hr_feats_rgb_crop_norm = network.featupUpsamplper(lr_feats_rgb_crop_norm.to(DEVICE), rgb_crop_norm.to(DEVICE))
            print(f"\n8. Allocated: {torch.cuda.memory_allocated(DEVICE)/(10**9):.2f} Resereved: {torch.cuda.memory_reserved(DEVICE)/(10**9):.2f}\n")
            if hr_feats_rgb_crop_norm.shape[2] != rgb_crop_norm.shape[2]:
                hr_feats_rgb_crop_norm = torch.nn.functional.interpolate(hr_feats_rgb_crop_norm, rgb_crop_norm.shape[2:], mode="bilinear")
            magnitude_norm = torch.norm(hr_feats_rgb_crop_norm, p=2, dim=1, keepdim=True)
            magnitude_norm = magnitude_norm + epsilon # TO avoid division by 0
            normalized_features_norm = hr_feats_rgb_crop_norm / magnitude_norm
            labels_crop_norm, selected_pixels_crop_norm = clustering_features(normalized_features_norm.detach().cpu())
            out_label_refined_norm, labels_crop_norm = match_label_crop(out_label_norm, labels_crop_norm.to(DEVICE), out_label_crop_norm, rois_norm, depth_crop_norm)
        print(f"\n{out_label_norm.shape=}\t{out_label_refined_norm.shape=}\n")
        _vis_minibatch_segmentation(sample['rgb'], depth, label, out_label_norm, out_label_refined_norm, normalized_features, 
                selected_pixels=selected_pixels_norm, bbox=None)
        
        break
        if i==100:
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