"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time
import torch

import _init_paths
from core.config import cfg, merge_cfg_from_file, assert_and_infer_cfg
from core.evaluate_engine import run_evaluate
import utils.logging
cv2.ocl.setUseOpenCL(False)


def parseArgs():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--dataset', default='fsod', help='training dataset')
    parser.add_argument('--cfg', dest='cfg_file', required=True, help='optional config file')
    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument('--load_detectron', help='path to the detectron weight pickle file')
    parser.add_argument('--output_dir', help='output directory to save the testing results.')
    parser.add_argument('--detfile_dir', required=True, help='path to saved detection pkl files')
    parser.add_argument('--vis', dest='vis', help='visualize detections', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        sys.exit("CUDA not available!")
    logger = utils.logging.setup_logging(__name__)
    args = parseArgs()
    logger.info('Run with argument:')
    logger.info(args)

    if args.dataset == "fsod":
        cfg.TEST.DATASETS = ('fsod_test',)
        cfg.MODEL.NUM_CLASSES = 201
    
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)

    # checkpoint and detectron pickle file
    assert bool(args.load_ckpt) ^ bool(args.load_detectron), 'Exactly one of --load_ckpt and --load_detectron should be specified.'
    
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'evaluate')
        logger.info('Output directory is %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    assert_and_infer_cfg()
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # manually set args.cuda
    args.cuda = True
    run_evaluate(args)
