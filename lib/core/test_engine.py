from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import os
import yaml
import cv2
import datetime
import logging
import torch
import numpy as np

from core.config import cfg
from core.test import im_detect_all
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from datasets.roidb import get_roidb_and_dataset
from modeling import model_builder
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.net as net_utils
import utils.vis as vis_utils
from utils.io import save_object
from utils.timer import Timer
np.random.seed(666)
logger = logging.getLogger(__name__)


def test_net(
        args,
        test_range=None,
        check_expected_results=False):

    def result_getter():
        if test_range is None:
            # Parent case: Test on the entire dataset
            dataset_name = cfg.TEST.DATASETS[0]
            output_dir = args.output_dir
            results = test_net_all(args, dataset_name, output_dir)
            return results
        else:
            # Child case: Test on range of dataset
            dataset_name = cfg.TEST.DATASETS[0]
            output_dir = args.output_dir
            return test_net_range(args, dataset_name, output_dir, test_range,)

    all_results = result_getter()
    if check_expected_results and test_range is None:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)

    return all_results


def test_net_all(
        args,
        dataset_name,
        output_dir):
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    all_boxes = test_net_range(
        args, dataset_name, output_dir  # stephane on 4/30 10:09PM
    )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    results = task_evaluation.evaluate_all(dataset, all_boxes, output_dir)
    return results


def test_net_range(
        args,
        dataset_name,
        output_dir,
        test_range=None):
    """
    Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    full_roidb, dataset, start_ind, end_ind, total_num_images, total_num_cls, support_dict = get_roidb_and_dataset(dataset_name, test_range)
    model = initialize_model_from_cfg(args)

    base_real_index = full_roidb[start_ind]['real_index']
    roidb = full_roidb[start_ind : end_ind]

    index_ls = []
    for item in roidb:
        index_ls.append(item['real_index'])
    num_annotations = len(roidb)
    num_images = len(list(set(index_ls)))
    num_classes = total_num_cls
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    print('part:', num_images)

    timers = defaultdict(Timer)

    for i, entry in enumerate(roidb):
        assert len(list(set(entry['gt_classes']))) == 1
        query_cls = list(set(entry['gt_classes']))[0]

        all_cls = support_dict[query_cls]['all_cls'] 

        support_way = 5
        support_shot = 5
        support_data_all = np.zeros((support_way * support_shot, 3, 320, 320), dtype = np.float32)
        support_box_all = np.zeros((support_way * support_shot, 4), dtype = np.float32)
        support_cls_ls = []

        for cls_id, cls in enumerate(all_cls):
            begin = cls_id * support_shot
            end = (cls_id + 1) * support_shot
            support_data_all[begin:end] = support_dict[cls]['img']
            support_box_all[begin:end] = support_dict[cls]['box']
            support_cls_ls.append(cls)            

        save_path = './vis'
        im = cv2.imread(entry['image'])
        cls_boxes_i = im_detect_all(model, im, support_data_all, support_box_all, support_cls_ls, save_path, timers)

        real_index = entry['real_index'] - base_real_index
        cls_boxes_i = cls_boxes_i[1]
        for cls in support_cls_ls:
            all_boxes[cls][real_index] = cls_boxes_i[cls_boxes_i[:,5] == cls][:, :5]

        if i % 10 == 0:  # Write log summary
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_annotations - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                timers['im_detect_bbox'].average_time +
                timers['im_detect_mask'].average_time +
                timers['im_detect_keypoints'].average_time)
            misc_time = (
                timers['misc_bbox'].average_time +
                timers['misc_mask'].average_time +
                timers['misc_keypoints'].average_time)
            logger.info(
                (
                    'im_detect: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_annotations, det_time, misc_time, eta
                )
            )

        if cfg.VIS:  # visualize images
            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            vis_utils.vis_one_image(
                im[:, :, ::-1],
                '{:d}_{:s}'.format(i, im_name),
                os.path.join(output_dir, 'vis'),
                cls_boxes_i,
                thresh=cfg.VIS_TH,
                box_alpha=0.8,
                dataset=dataset,
                show_class=True
            )

    cfg_yaml = yaml.dump(cfg)
    if test_range is not None:
        det_name = 'detection_range_%s_%s.pkl' % tuple(test_range)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(
            all_boxes=all_boxes,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return all_boxes


def initialize_model_from_cfg(args):
    """
    Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = model_builder.Generalized_RCNN()
    model.eval()
    model.cuda()

    if args.load_ckpt:
        logger.info("Loading checkpoint %s", args.load_ckpt)
        checkpoint = torch.load(args.load_ckpt, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    if args.load_detectron:
        logger.info("Loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model
