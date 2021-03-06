from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import logging
import numpy as np
import pandas as pd

import utils.boxes as box_utils
import utils.keypoints as keypoint_utils
from core.config import cfg
from .json_dataset import JsonDataset
from datasets.json_dataset import JsonDataset
from .crop import crop_support
logger = logging.getLogger(__name__)


def combined_roidb_for_training(dataset_names, proposal_files):
    """Load and concatenate roidbs for one or more datasets, along with optional
    object proposals. The roidb entries are then prepared for use in training,
    which involves caching certain types of metadata for each roidb entry.
    """
    def get_roidb(dataset_name, proposal_file):
        ds = JsonDataset(dataset_name)
        roidb = ds.get_roidb(
            gt=True,
            proposal_file=proposal_file,
            crowd_filter_thresh=cfg.TRAIN.CROWD_FILTER_THRESH
        )
        if cfg.TRAIN.USE_FLIPPED:
            logger.info('Appending horizontally-flipped training examples...')
            extend_with_flipped_entries(roidb, ds)
        logger.info('Loaded dataset: {:s}'.format(ds.name))
        return roidb

    if isinstance(dataset_names, six.string_types):
        dataset_names = (dataset_names, )
    if isinstance(proposal_files, six.string_types):
        proposal_files = (proposal_files, )
    if len(proposal_files) == 0:
        proposal_files = (None, ) * len(dataset_names)
    assert len(dataset_names) == len(proposal_files)
    roidbs = [get_roidb(*args) for args in zip(dataset_names, proposal_files)]
    original_roidb = roidbs[0]
   
    # new dataset split according to class 
    roidb = []
    for item in original_roidb:
        gt_classes = list(set(item['gt_classes']))  # distinct gt classes
        all_cls = np.array(item['gt_classes'])

        for cls in gt_classes:
            item_new = item.copy()
            target_idx = np.where(all_cls == cls)[0]  # array([ 0,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12]), first element in tuple
            #item_new['id'] = item_new['id'] * 1000 + int(cls)
            item_new['target_cls'] = int(cls)
            item_new['boxes'] = item_new['boxes'][target_idx]
            item_new['max_classes'] = item_new['max_classes'][target_idx]
            item_new['gt_classes'] = item_new['gt_classes'][target_idx]
            item_new['is_crowd'] = item_new['is_crowd'][target_idx]
            item_new['segms'] = item_new['segms'][:target_idx.shape[0]]
            item_new['seg_areas'] = item_new['seg_areas'][target_idx]
            item_new['max_overlaps'] = item_new['max_overlaps'][target_idx]
            item_new['box_to_gt_ind_map'] = np.array(range(item_new['gt_classes'].shape[0]))
            item_new['gt_overlaps'] = item_new['gt_overlaps'][target_idx]
            roidb.append(item_new)

    for r in roidbs[1:]:
        roidb.extend(r)
    roidb = filter_for_training(roidb)

    if cfg.TRAIN.ASPECT_GROUPING or cfg.TRAIN.ASPECT_CROPPING:
        logger.info('Computing image aspect ratios and ordering the ratios...')
        ratio_list, ratio_index, cls_list, id_list = rank_for_training(roidb)
        logger.info('done')
    else:
        ratio_list, ratio_index, cls_list, id_list = None, None, None, None

    logger.info('Computing bounding-box regression targets...')
    add_bbox_regression_targets(roidb)
    logger.info('done')

    _compute_and_log_stats(roidb)

    print(len(roidb))
    return roidb, ratio_list, ratio_index, cls_list, id_list


def extend_with_flipped_entries(roidb, dataset):
    """Flip each entry in the given roidb and return a new roidb that is the
    concatenation of the original roidb and the flipped entries.

    "Flipping" an entry means that that image and associated metadata (e.g.,
    ground truth boxes and object proposals) are horizontally flipped.
    """
    flipped_roidb = []
    for entry in roidb:
        width = entry['width']
        boxes = entry['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        flipped_entry = {}
        dont_copy = ('boxes', 'segms', 'gt_keypoints', 'flipped')
        for k, v in entry.items():
            if k not in dont_copy:
                flipped_entry[k] = v
        flipped_entry['boxes'] = boxes
        #flipped_entry['segms'] = segm_utils.flip_segms(
        #    entry['segms'], entry['height'], entry['width']
        #)
        flipped_entry['segms'] = entry['segms']
        if dataset.keypoints is not None:
            flipped_entry['gt_keypoints'] = keypoint_utils.flip_keypoints(
                dataset.keypoints, dataset.keypoint_flip_map,
                entry['gt_keypoints'], entry['width']
            )
        flipped_entry['flipped'] = True
        flipped_roidb.append(flipped_entry)
    roidb.extend(flipped_roidb)


def filter_for_training(roidb):
    """Remove roidb entries that have no usable RoIs based on config settings.
    """
    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        if cfg.MODEL.KEYPOINTS_ON:
            # If we're training for keypoints, exclude images with no keypoints
            valid = valid and entry['has_visible_keypoints']
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    logger.info('Filtered {} roidb entries: {} -> {}'.
                format(num - num_after, num, num_after))
    return filtered_roidb


def rank_for_training(roidb):
    """Rank the roidb entries according to image aspect ration and mark for cropping
    for efficient batching if image is too long.

    Returns:
        ratio_list: ndarray, list of aspect ratios from small to large
        ratio_index: ndarray, list of roidb entry indices correspond to the ratios
    """
    RATIO_HI = cfg.TRAIN.ASPECT_HI  # largest ratio to preserve.
    RATIO_LO = cfg.TRAIN.ASPECT_LO  # smallest ratio to preserve.

    need_crop_cnt = 0

    ratio_list = []
    cls_list = []
    id_list = []
    for entry in roidb:
        width = entry['width']
        height = entry['height']
        ratio = width / float(height)
        target_cls = entry['target_cls']
        img_id = entry['id'] #int(str(entry['id'])[:-3])

        if cfg.TRAIN.ASPECT_CROPPING:
            if ratio > RATIO_HI:
                entry['need_crop'] = True
                ratio = RATIO_HI
                need_crop_cnt += 1
            elif ratio < RATIO_LO:
                entry['need_crop'] = True
                ratio = RATIO_LO
                need_crop_cnt += 1
            else:
                entry['need_crop'] = False
        else:
            entry['need_crop'] = False

        ratio_list.append(ratio)
        cls_list.append(target_cls)
        id_list.append(img_id)

    if cfg.TRAIN.ASPECT_CROPPING:
        logging.info('Number of entries that need to be cropped: %d. Ratio bound: [%.2f, %.2f]',
                     need_crop_cnt, RATIO_LO, RATIO_HI)
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    cls_list = np.array(cls_list)
    id_list = np.array(id_list)
    return ratio_list[ratio_index], ratio_index, cls_list, id_list

def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    for entry in roidb:
        entry['bbox_targets'] = _compute_targets(entry)


def _compute_targets(entry):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    rois = entry['boxes']
    overlaps = entry['max_overlaps']
    labels = entry['max_classes']
    gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    # Targets has format (class, tx, ty, tw, th)
    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return targets

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = box_utils.bbox_overlaps(
        rois[ex_inds, :].astype(dtype=np.float32, copy=False),
        rois[gt_inds, :].astype(dtype=np.float32, copy=False))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]
    # Use class "1" for all boxes if using class_agnostic_bbox_reg
    targets[ex_inds, 0] = (
        1 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else labels[ex_inds])
    targets[ex_inds, 1:] = box_utils.bbox_transform_inv(
        ex_rois, gt_rois, cfg.MODEL.BBOX_REG_WEIGHTS)
    return targets


def _compute_and_log_stats(roidb):
    classes = roidb[0]['dataset'].classes
    char_len = np.max([len(c) for c in classes])
    hist_bins = np.arange(len(classes) + 1)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((len(classes)), dtype=np.int)
    for entry in roidb:
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_classes = entry['gt_classes'][gt_inds]
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    logger.debug('Ground-truth class histogram:')
    for i, v in enumerate(gt_hist):
        logger.debug(
            '{:d}{:s}: {:d}'.format(
                i, classes[i].rjust(char_len), v))
    logger.debug('-' * char_len)
    logger.debug(
        '{:s}: {:d}'.format(
            'total'.rjust(char_len), np.sum(gt_hist)))


def get_roidb_and_dataset(dataset_name, test_range):
    """
    Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if test_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    original_roidb, roidb = dataset.get_roidb(gt=True, test_flag=True)

    # construct support image crops with bounding box
    support_roidb = []
    cnt = 0
    for item_id, item in enumerate(original_roidb):
        gt_classes = list(set(item['gt_classes']))
        for cls in gt_classes:
            item_new = item.copy()
            item_new['target_cls'] = int(cls)
            all_cls = item['gt_classes']
            target_idx = np.where(all_cls == cls)[0] 
            item_new['boxes'] = item['boxes'][target_idx]
            item_new['gt_classes'] = item['gt_classes'][target_idx]
            item_new['index'] = cnt
            item_new['real_index'] = item_id
            cnt += 1
            support_roidb.append(item_new)
    print('support annotation number: ', len(support_roidb))
    roidb_img = []
    roidb_cls = []
    roidb_index = []
    for item_id, item in enumerate(support_roidb):
        roidb_img.append(item['image'])
        roidb_cls.append(item['target_cls'])
        roidb_index.append(item['index'])
        assert item_id == item['index']
    data_dict = {'img_ls': roidb_img, 'cls_ls': roidb_cls, 'index': roidb_index}
    # construct dataframe for picking support images
    support_df = pd.DataFrame.from_dict(data_dict)
    # query image summary for each episode for picking support images, 10 query 5 support
    episode_num = 400  #600 #500
    query_way_num = 5
    query_shot_num = 10
    total_num_cls = episode_num * query_way_num + 1
    # support_way_num = 5
    support_shot_num = 5

    support_dict = {}
    for ep in range(episode_num):
        query_img = []
        query_cls = []
        query_index = []
        used_img_ls = []
        query_real_cls = []
        for way in range(query_way_num):
            for shot in range(query_shot_num):
                roidb_id = ep * 50 + way * 10 + shot
                current_roidb = roidb[roidb_id]
                query_img.append(current_roidb['image'])
                query_index.append(current_roidb['index'])
                used_img_ls.append(current_roidb['image'])
            real_cls = current_roidb['target_cls']
            query_real_cls.append(real_cls)
            query_cls.append(list(set(current_roidb['gt_classes']))[0])
        assert len(query_cls) == len(query_real_cls) == query_way_num

        for cls_id, cls in enumerate(query_cls):
            support_dict[cls] = {}
            support_dict[cls]['img'] = np.zeros((support_shot_num, 3, 320, 320), dtype = np.float32)
            support_dict[cls]['box'] = np.zeros((support_shot_num, 4), dtype = np.float32)
            support_dict[cls]['all_cls'] = query_cls
            support_real_cls = query_real_cls[cls_id]
            for shot in range(support_shot_num):
                random_id = ep * 25 + cls_id * 5 + shot
                support_index = support_df.loc[(support_df['cls_ls'] == support_real_cls) & (~support_df['img_ls'].isin(used_img_ls)), 'index'].sample(random_state=random_id).tolist()[0]
                current_support = support_roidb[support_index]
                img_name = current_support['image']
                support_img, support_box = crop_support(current_support)
                support_dict[cls]['img'][shot] = support_img
                support_dict[cls]['box'][shot] = support_box[0]
                used_img_ls.append(img_name)

    if test_range is not None:
        total_num_images = len(roidb) # it is the query roidb, 14152
        start, end = test_range
        print(f"!!!!!!!!!!!!!!!!start: {start}, end: {end}")  # stephane on 4/30 5:35PM
    else:
        start = 0
        end = len(roidb)
        total_num_images = end
    return roidb, dataset, start, end, total_num_images, total_num_cls, support_dict
