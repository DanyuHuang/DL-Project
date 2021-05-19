"""Perform inference on one or more datasets."""

import os
import sys
import pprint
import argparse
import cv2
import torch
import _init_paths
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import test_net
import utils.logging
cv2.ocl.setUseOpenCL(False)


def parseArgs():
    """
    Command line arguments are defined here
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fsod', help='training dataset')  # use fsod dataset for default
    parser.add_argument('--cfg', dest='cfg_file', required=True, help='optional config file')
    parser.add_argument('--load_ckpt', help='path to load checkpoint')
    parser.add_argument('--load_detectron', help='path to load detectron weight pickle file')
    parser.add_argument('--output_dir', help='output directory to save the testing results.')
    parser.add_argument('--range', help='[start, end)', type=int, nargs=2)
    parser.add_argument('--visualize', dest='visualize', help='output images of detection', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    if torch.cuda.is_available() == False:
        sys.exit("CUDA not available!")
    logger = utils.logging.setup_logging(__name__)
    args = parseArgs()
    logger.info('Run with argument:')
    logger.info(args)

    if args.dataset == "fsod":  # argument for dataset
        cfg.TEST.DATASETS = ('fsod_test',)
        cfg.MODEL.NUM_CLASSES = 201
    
    if args.cfg_file is not None:  # argument for config file
        merge_cfg_from_file(args.cfg_file)
    
    # checkpoint and detectron pickle file
    assert bool(args.load_ckpt) ^ bool(args.load_detectron), 'Checkpoint and detectron error! Exactly one of the two should be specified.'
    
    # argument for outpur directory
    if args.output_dir is None:
        if args.load_ckpt:
            ckpt_path = args.load_ckpt
        else:
            ckpt_path = args.load_detectron
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Output directory is %s', args.output_dir)
    if os.path.exists(args.output_dir) == False:  # if there is no such directory, make directory
        os.makedirs(args.output_dir)

    # argument for output images
    cfg.VIS = args.visualize

    # output configuration
    assert_and_infer_cfg()
    logger.info('Test configuration:')
    logger.info(pprint.pformat(cfg))

    # argument for CUDA
    args.cuda = True

    # Now start testing
    logger.info('Start testing...')
    test_net(args, test_range=args.range, check_expected_results=True)
