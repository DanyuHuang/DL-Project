from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils
import numpy as np

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class SQCorrelationModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.SQ_l1 = nn.Linear(1024, 1024) #???input_dimention, output_dimention???
        self.SQ_l2 = nn.Linear(1024, 1024)
        self.SQ_l3 = nn.Linear(1024, 1024)
        self.SQ_l4 = nn.Linear(1024, 1024)
        self.SQ_pool1 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.SQ_pool2 = torch.nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, input_q, input_s):
        model_q = self.SQ_pool1(input_q).view(-1, input_q.shape[1]) # the size -1 is inferred from other dimensions
        model_q=self.SQ_l1(model_q)
        model_q = F.relu(model_q)  # activation function input and output are the same size
        model_q=self.SQ_l2(model_q)
        model_q = F.sigmoid(model_q) # activation function input and output are the same size

        model_s = self.SQ_pool2(input_s).view(-1, input_s.shape[1])
        model_s=self.SQ_l3(model_s)
        model_s = F.relu(model_s)
        model_s=self.SQ_l4(model_s)
        model_s = F.sigmoid(model_s)

        cor_qs = model_q * model_s #?????????????????? #a???b???*????????????????????????a???b???size??????????????????????????????a???b?????????????????????????????????a???b???size?????????????????????a???b???element-wise????????????

        cor_qs = cor_qs.unsqueeze(-1).unsqueeze(-1) # increase 2 dimension to 4
        #???expand??????-1????????????????????????????????????????????????????????????????????????
        output_q = input_q * cor_qs.expand(-1, -1, input_q.shape[-2], input_q.shape[-1]) #shape[-1] ?????? shape[-2] ??????
        output_s = input_s * cor_qs.expand(-1, -1, input_s.shape[-2], input_s.shape[-1])

        output_q = torch.mean(output_q, 0).unsqueeze(0) #torch.mean(tensor, dim=0) dim ???????????? ?????? dim ???????????? ???????????????????????????????????????unsqueeze????????????
        return output_q, output_s

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'SQ_l1.weight': 'fc_q1_w',
            'SQ_l1.bias': 'fc_q1_b',
            'SQ_l2.weight': 'fc_q2_w',
            'SQ_l2.bias': 'fc_q2_b',
            'SQ_l3.weight': 'fc_s1_w',
            'SQ_l3.bias': 'fc_s1_b',
            'SQ_l4.weight': 'fc_s2_w',
            'SQ_l4.bias': 'fc_s2_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out)

        # Mask Branch
        if cfg.MODEL.MASK_ON:
            self.Mask_Head = get_func(cfg.MRCNN.ROI_MASK_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                self.Mask_Head.share_res5_module(self.Box_Head.res5)
            self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out)

        # Keypoints Branch
        if cfg.MODEL.KEYPOINTS_ON:
            self.Keypoint_Head = get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                self.Keypoint_Head.share_res5_module(self.Box_Head.res5)
            self.Keypoint_Outs = keypoint_rcnn_heads.keypoint_outputs(self.Keypoint_Head.dim_out)

        self.avgpool = nn.AvgPool2d(14)

        self.avgpool_fc = nn.AvgPool2d(7)
        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)
            # Check if shared weights are equaled
            if cfg.MODEL.MASK_ON and getattr(self.Mask_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Mask_Head.res5.state_dict(), self.Box_Head.res5.state_dict())
            if cfg.MODEL.KEYPOINTS_ON and getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Keypoint_Head.res5.state_dict(), self.Box_Head.res5.state_dict())

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, support_data, im_info, roidb=None, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, support_data, im_info, roidb, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, support_data, im_info, roidb, **rpn_kwargs)

    def _forward(self, data, support_data, im_info, roidb=None, **rpn_kwargs):
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))
        device_id = im_data.get_device()
        return_dict = {}  # A dict to collect return variables
        original_blob_conv = self.Conv_Body(im_data)

        if not self.training: 
            all_cls = roidb[0]['support_cls']
            test_way = len(all_cls)
            test_shot = int(roidb[0]['support_boxes'].shape[0] / test_way)
        else:
            train_way = 2 #2
            train_shot = 5 #5
        support_blob_conv = self.Conv_Body(support_data.squeeze(0))

        ########################
        # sq_Correlation network
        original_blob_conv, support_blob_conv = self.sqc(original_blob_conv, support_blob_conv)

        img_num = int(support_blob_conv.shape[0])
        img_channel = int(original_blob_conv.shape[1])
        # Construct support rpn_ret
        support_rpn_ret = {'rois': np.insert(roidb[0]['support_boxes'][0], 0, 0.)[np.newaxis, :]}
        if img_num > 1:
            for i in range(img_num-1):
                support_rpn_ret['rois'] = np.concatenate((support_rpn_ret['rois'], np.insert(roidb[0]['support_boxes'][i+1], 0, float(i+1))[np.newaxis, :]), axis=0) 
        # Get support pooled feature
        support_feature = self.roi_feature_transform(
            support_blob_conv, support_rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.Conv_Body.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )

        blob_conv = original_blob_conv
        rpn_ret = self.RPN(blob_conv, im_info, roidb)
        # if self.training:
        #     # can be used to infer fg/bg ratio
        #     return_dict['rois_label'] = rpn_ret['labels_int32']

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        assert not cfg.MODEL.RPN_ONLY

        support_box_feat = self.Box_Head(support_blob_conv, support_rpn_ret)
        if self.training:
            support_feature_mean_0 = support_feature[:5].mean(0, True)
            support_pool_feature_0 = self.avgpool(support_feature_mean_0)
            correlation_0 = F.conv2d(original_blob_conv, support_pool_feature_0.permute(1,0,2,3), groups=1024)
            rpn_ret_0 = self.RPN(correlation_0, im_info, roidb)
            box_feat_0 = self.Box_Head(blob_conv, rpn_ret_0)
            support_0 = support_box_feat[:5].mean(0, True) # simple average few shot support features

            support_feature_mean_1 = support_feature[5:10].mean(0, True)
            support_pool_feature_1 = self.avgpool(support_feature_mean_1)
            correlation_1 = F.conv2d(original_blob_conv, support_pool_feature_1.permute(1,0,2,3), groups=1024)
            rpn_ret_1 = self.RPN(correlation_1, im_info, roidb)
            box_feat_1 = self.Box_Head(blob_conv, rpn_ret_1)
            support_1 = support_box_feat[5:10].mean(0, True) # simple average few shot support features

            cls_score_now_0, bbox_pred_now_0 = self.Box_Outs(box_feat_0, support_0)
            cls_score_now_1, bbox_pred_now_1 = self.Box_Outs(box_feat_1, support_1)
 
            cls_score = torch.cat([cls_score_now_0, cls_score_now_1], dim=0)
            bbox_pred = torch.cat([bbox_pred_now_0, bbox_pred_now_1], dim=0)
            rpn_ret = {}
            for key in rpn_ret_0.keys():
                if key != 'rpn_cls_logits' and key != 'rpn_bbox_pred':
                    rpn_ret[key] = rpn_ret_0[key] #np.concatenate((rpn_ret_0[key], rpn_ret_1[key]), axis=0)
                else:
                    rpn_ret[key] = torch.cat([rpn_ret_0[key], rpn_ret_1[key]], dim=0)

        else:
            for way_id in range(test_way):
                begin = way_id * test_shot
                end = (way_id + 1) * test_shot

                support_feature_mean = support_feature[begin:end].mean(0, True)
                support_pool_feature = self.avgpool(support_feature_mean)
                correlation = F.conv2d(original_blob_conv, support_pool_feature.permute(1,0,2,3), groups=1024)
                rpn_ret = self.RPN(correlation, im_info, roidb)

                if not cfg.MODEL.RPN_ONLY:
                    if cfg.MODEL.SHARE_RES5 and self.training:
                        box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret)
                    else:
                        box_feat = self.Box_Head(blob_conv, rpn_ret)

                    support = support_box_feat[begin:end].mean(0, True) # simple average few shot support features

                    cls_score_now, bbox_pred_now = self.Box_Outs(box_feat, support)
                    cls_now = cls_score_now.new_full((cls_score_now.shape[0], 1), int(roidb[0]['support_cls'][way_id]))
                    cls_score_now = torch.cat([cls_score_now, cls_now], dim=1)
                    rois = rpn_ret['rois']
                    if way_id == 0:
                        cls_score = cls_score_now
                        bbox_pred = bbox_pred_now
                        rois_all = rois
                    else:
                        cls_score = torch.cat([cls_score, cls_score_now], dim=0)
                        bbox_pred = torch.cat([bbox_pred, bbox_pred_now], dim=0)
                        rois_all = np.concatenate((rois_all, rois), axis=0)
                else:
                    # TODO: complete the returns for RPN only situation
                    pass 
            rpn_ret['rois'] = rois_all
        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            # rpn loss
            rpn_kwargs.update(dict(
                (k, rpn_ret[k]) for k in rpn_ret.keys()
                if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
            ))
            target_cls = roidb[0]['target_cls']
            rpn_ret['labels_obj'] = np.array([int(i==target_cls) for i in rpn_ret['labels_int32']])

            # filter other class bbox targets, only supervise the target cls bbox, because the bbox_pred is from the compare feature.
            bg_idx = np.where(rpn_ret['labels_int32'] != target_cls)[0]
            rpn_ret['bbox_targets'][bg_idx] = np.full_like(rpn_ret['bbox_targets'][bg_idx], 0.)
            rpn_ret['bbox_inside_weights'][bg_idx] = np.full_like(rpn_ret['bbox_inside_weights'][bg_idx], 0.)
            rpn_ret['bbox_outside_weights'][bg_idx] = np.full_like(rpn_ret['bbox_outside_weights'][bg_idx], 0.)
            
            neg_labels_obj = np.full_like(rpn_ret['labels_obj'], 0.)
            neg_bbox_targets = np.full_like(rpn_ret['bbox_targets'], 0.)
            neg_bbox_inside_weights = np.full_like(rpn_ret['bbox_inside_weights'], 0.)
            neg_bbox_outside_weights = np.full_like(rpn_ret['bbox_outside_weights'], 0.)

            rpn_ret['labels_obj'] = np.concatenate([rpn_ret['labels_obj'], neg_labels_obj], axis=0)
            rpn_ret['bbox_targets'] = np.concatenate([rpn_ret['bbox_targets'], neg_bbox_targets], axis=0)
            rpn_ret['bbox_inside_weights'] = np.concatenate([rpn_ret['bbox_inside_weights'], neg_bbox_inside_weights], axis=0)
            rpn_ret['bbox_outside_weights'] = np.concatenate([rpn_ret['bbox_outside_weights'], neg_bbox_outside_weights], axis=0)
            
            loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
            if cfg.FPN.FPN_ON:
                for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                    return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                    return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
            else:
                return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

            # bbox loss
            loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                cls_score, bbox_pred, rpn_ret['labels_obj'], rpn_ret['bbox_targets'],
                rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
            return_dict['losses']['loss_cls'] = 1 * loss_cls
            return_dict['losses']['loss_bbox'] = 1 * loss_bbox
            return_dict['metrics']['accuracy_cls'] = accuracy_cls


            if cfg.MODEL.MASK_ON:
                if getattr(self.Mask_Head, 'SHARE_RES5', False):
                    mask_feat = self.Mask_Head(res5_feat, rpn_ret,
                                               roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
                else:
                    mask_feat = self.Mask_Head(blob_conv, rpn_ret)
                mask_pred = self.Mask_Outs(mask_feat)
                # return_dict['mask_pred'] = mask_pred
                # mask loss
                loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rpn_ret['masks_int32'])
                return_dict['losses']['loss_mask'] = loss_mask

            if cfg.MODEL.KEYPOINTS_ON:
                if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                    # No corresponding keypoint head implemented yet (Neither in Detectron)
                    # Also, rpn need to generate the label 'roi_has_keypoints_int32'
                    kps_feat = self.Keypoint_Head(res5_feat, rpn_ret,
                                                  roi_has_keypoints_int32=rpn_ret['roi_has_keypoint_int32'])
                else:
                    kps_feat = self.Keypoint_Head(blob_conv, rpn_ret)
                kps_pred = self.Keypoint_Outs(kps_feat)
                # return_dict['keypoints_pred'] = kps_pred
                # keypoints loss
                if cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'])
                else:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'],
                        rpn_ret['keypoint_loss_normalizer'])
                return_dict['losses']['loss_kps'] = loss_keypoints

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)

        else:
            # Testing
            return_dict['rois'] = rpn_ret['rois']
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred

        return return_dict

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @check_inference
    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        mask_feat = self.Mask_Head(blob_conv, rpn_blob)
        mask_pred = self.Mask_Outs(mask_feat)
        return mask_pred

    @check_inference
    def keypoint_net(self, blob_conv, rpn_blob):
        """For inference"""
        kps_feat = self.Keypoint_Head(blob_conv, rpn_blob)
        kps_pred = self.Keypoint_Outs(kps_feat)
        return kps_pred

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
