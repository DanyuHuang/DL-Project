"""Evaluate partial detection files"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import numpy as np

from core.config import cfg
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from datasets.roidb import get_roidb_and_dataset
from utils.timer import Timer
import pandas as pd
np.random.seed(666)
logger = logging.getLogger(__name__)


def run_evaluate(args):
    output_dir = args.output_dir
    detfile_dir = args.detfile_dir
    dataset_name = cfg.TEST.DATASETS[0]  # 'fsod_test'
    evaluate_net_on_dataset(args, dataset_name, output_dir, detfile_dir)


def evaluate_net_on_dataset(
        args,
        dataset_name,
        output_dir,
        detfile_dir):  # detection pkl file path
    """Run evaluation on a dataset."""
    dataset = JsonDataset(dataset_name)
    evaluate_timer = Timer()
    evaluate_timer.tic()

    all_boxes = evaluate_net(args, dataset_name, output_dir, detfile_dir)
    evaluate_timer.toc()
    logger.info('Total evaluation time: {:.3f}s'.format(evaluate_timer.average_time))
    results = task_evaluation.evaluate_all(dataset, all_boxes, output_dir)
    return results


def evaluate_net(
        args,
        dataset_name,
        detfile_dir,
        ind_range=None):
    """Evaluate the detection results and show the results"""
    full_roidb, dataset, start_ind, end_ind, total_num_images, total_num_cls, support_dict = get_roidb_and_dataset(dataset_name, ind_range)
    roidb = full_roidb[start_ind : end_ind]

    index_ls = []
    for item in roidb:
        index_ls.append(item['real_index'])
    num_classes = total_num_cls  # 3001
    num_images = len(list(set(index_ls)))  # 30000
    print('part:', num_images)

    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    # get all_boxes
    all_detection = []
    for entry in os.listdir(detfile_dir):
        if entry.find('.pkl') != -1:
            filename = os.path.join(detfile_dir, entry)
            all_detection.append(filename)
    
    all_boxes = [[] for i in range(num_classes)]  # need to be [3001][4000]
    for detfile in all_detection:
        det_partial = pd.read_pickle(detfile)
        # print('Loaded detection file {}'.format(detfile))
        logger.info('Loaded detection file {}'.format(detfile))
        boxes_partial = det_partial['all_boxes']
        for i in range(num_classes):
            all_boxes[i].extend(boxes_partial[i])
        logger.info('Done processing file {}'.format(detfile))
    return all_boxes
