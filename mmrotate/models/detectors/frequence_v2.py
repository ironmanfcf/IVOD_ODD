import warnings
import math
import time
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np

from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward, DWTInverse
"""""""""

这里我们尝试一下双分支结构是否会带来增益，对于原始的加法操作和频率调制操作的作用。

"""""""""

@ROTATED_DETECTORS.register_module()
class FrequenceDetV2(RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FrequenceDetV2, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone_ir = build_backbone(backbone)
        self.backbone_rgb = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # 初始化 DWTTransformerattentionlayer 模块
        self.dwt_transformer_attention = DWTTransformerattentionlayer(channels=neck.out_channels)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None



    def extract_feat(self, img):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).
            batch_inputs_ir (Tensor): Input images of shape (N, C, H, W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        OD_RGB, OD_IR = img
        x_rgb = self.backbone_rgb(OD_RGB)
        x_ir = self.backbone_ir(OD_IR)

        if self.with_neck:
            x_rgb = self.neck(x_rgb)
            x_ir = self.neck(x_ir)

        # 对每个层级的特征进行处理
        processed_feats = list()
        
        # for feat, feat_ir in zip(x_rgb, x_ir):
           
        #     # 使用 DWTTransformerattentionlayer 进行特征处理
        #     processed_feat = self.dwt_transformer_attention(feat, feat_ir)       
        #     processed_feats.append(processed_feat)
            
        for feat, feat_ir in zip(x_rgb, x_ir):
            processed_feat = feat + feat_ir            
            processed_feats.append(processed_feat)

        return processed_feats
    
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat((img, img))
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale) 

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                if isinstance(img, list) or isinstance(img, tuple):
                    img_meta[img_id]['batch_input_shape'] = tuple(img[0].size()[-2:])
                else:
                    img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            # print(self.aug_test(imgs, img_metas, **kwargs)) 
            return self.aug_test(imgs, img_metas, **kwargs)
 
    
class DWTTransformerattentionlayer(nn.Module):

    def __init__(self, channels):
        super(DWTTransformerattentionlayer, self).__init__()
        self.trans_LL_HH = CBAM(in_channels=channels)
        self.trans_HH_LL = CBAM(in_channels=channels)
        self.trans_LH_HL = CBAM(in_channels=channels)
        self.trans_HL_LH = CBAM(in_channels=channels)
        self.wavelet = WaveletTransform(wavelet='haar')

    def forward(self, x_vis, x_ir):
        # 对可见光特征进行小波变换
        yl_vis, yh_vis = self.wavelet.dwt2_transform(x_vis)
        LL_vis, LH_vis, HL_vis, HH_vis = (
            yl_vis,
            yh_vis[0][:, :, 0, :, :],
            yh_vis[0][:, :, 1, :, :],
            yh_vis[0][:, :, 2, :, :]
        )

        # 对红外特征进行小波变换
        yl_ir, yh_ir = self.wavelet.dwt2_transform(x_ir)
        LL_ir, LH_ir, HL_ir, HH_ir = (
            yl_ir,
            yh_ir[0][:, :, 0, :, :],
            yh_ir[0][:, :, 1, :, :],
            yh_ir[0][:, :, 2, :, :]
        )

        # 检查并调整特征形状
        def adjust_shape(tensor_a, tensor_b):
            if tensor_a.shape != tensor_b.shape:
                min_h = min(tensor_a.shape[-2], tensor_b.shape[-2])
                min_w = min(tensor_a.shape[-1], tensor_b.shape[-1])
                tensor_a = tensor_a[..., :min_h, :min_w]
                tensor_b = tensor_b[..., :min_h, :min_w]
            return tensor_a, tensor_b

        LL_vis, HH_ir = adjust_shape(LL_vis, HH_ir)
        HH_vis, LL_ir = adjust_shape(HH_vis, LL_ir)
        LH_vis, HL_ir = adjust_shape(LH_vis, HL_ir)
        HL_vis, LH_ir = adjust_shape(HL_vis, LH_ir)

        # 保存原始分量用于残差连接
        LL_vis_orig, HH_ir_orig = LL_vis.clone(), HH_ir.clone()
        HH_vis_orig, LL_ir_orig = HH_vis.clone(), LL_ir.clone()
        LH_vis_orig, HL_ir_orig = LH_vis.clone(), HL_ir.clone()
        HL_vis_orig, LH_ir_orig = HL_vis.clone(), LH_ir.clone()

        # 对不同模态间的分量进行 CBAM 调制
        LL_vis, HH_ir = self.trans_LL_HH(LL_vis, HH_ir)
        HH_vis, LL_ir = self.trans_HH_LL(HH_vis, LL_ir)
        LH_vis, HL_ir = self.trans_LH_HL(LH_vis, HL_ir)
        HL_vis, LH_ir = self.trans_HL_LH(HL_vis, LH_ir)

        # 添加残差连接
        LL_vis += LL_vis_orig
        HH_ir += HH_ir_orig
        HH_vis += HH_vis_orig
        LL_ir += LL_ir_orig
        LH_vis += LH_vis_orig
        HL_ir += HL_ir_orig
        HL_vis += HL_vis_orig
        LH_ir += LH_ir_orig

        # 重建处理过后的分量
        yl_vis = LL_vis
        new_yh_vis = torch.stack([LH_vis, HL_vis, HH_vis], dim=2)
        yh_vis[0] = new_yh_vis

        yl_ir = LL_ir
        new_yh_ir = torch.stack([LH_ir, HL_ir, HH_ir], dim=2)
        yh_ir[0] = new_yh_ir

        # 检查并调整逆小波变换前的系数形状
        if yl_vis.shape != yl_ir.shape:
            min_h = min(yl_vis.shape[-2], yl_ir.shape[-2])
            min_w = min(yl_vis.shape[-1], yl_ir.shape[-1])
            yl_vis = yl_vis[..., :min_h, :min_w]
            yl_ir = yl_ir[..., :min_h, :min_w]

        if yh_vis[0].shape != yh_ir[0].shape:
            min_h = min(yh_vis[0].shape[-2], yh_ir[0].shape[-2])
            min_w = min(yh_vis[0].shape[-1], yh_ir[0].shape[-1])
            yh_vis[0] = yh_vis[0][..., :min_h, :min_w]
            yh_ir[0] = yh_ir[0][..., :min_h, :min_w]

        # 逆小波变换重建特征
        processed_feat_vis = self.wavelet.idwt2_transform(yl_vis, yh_vis)
        processed_feat_ir = self.wavelet.idwt2_transform(yl_ir, yh_ir)

        # 检查并调整重建后的特征形状
        if processed_feat_vis.shape != processed_feat_ir.shape:
            min_h = min(processed_feat_vis.shape[-2], processed_feat_ir.shape[-2])
            min_w = min(processed_feat_vis.shape[-1], processed_feat_ir.shape[-1])
            processed_feat_vis = processed_feat_vis[..., :min_h, :min_w]
            processed_feat_ir = processed_feat_ir[..., :min_h, :min_w]

        # 添加全局残差连接
        output = (processed_feat_vis + processed_feat_ir) / 2

        if output.shape != x_vis.shape or output.shape != x_ir.shape:
            min_h = min(output.shape[-2], x_vis.shape[-2], x_ir.shape[-2])
            min_w = min(output.shape[-1], x_vis.shape[-1], x_ir.shape[-1])
            output = output[..., :min_h, :min_w]
            x_vis = x_vis[..., :min_h, :min_w]
            x_ir = x_ir[..., :min_h, :min_w]
        output += (x_vis + x_ir) / 2

        return output
    
class WaveletTransform(nn.Module):
    def __init__(self, wavelet='haar'):
        """
        初始化小波变换类。

        参数：
        wavelet: 使用的小波函数类型，默认为 'haar'
        """
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.dwt = DWTForward(J=1, wave=wavelet, mode='zero').cuda()  # 使用GPU加速的DWT
        self.idwt = DWTInverse(wave=wavelet, mode='zero').cuda()  # 使用GPU加速的IDWT

    def dwt2_transform(self, x):
        """
        对形状为 [b, c, h, w] 的张量进行离散小波变换，并将结果存储为四个张量。

        参数：
        x: 输入张量，形状为 [b, c, h, w]

        返回：
        四个张量，分别为逼近系数、水平细节系数、垂直细节系数和对角线细节系数
        """
        # 直接在 GPU 上进行小波变换
        yl, yh = self.dwt(x)

        return yl, yh

    def idwt2_transform(self, yl, yh):
        """
        对离散小波变换系数进行反变换，恢复到原始空间域。

        参数：
        cA_tensor: 逼近系数张量
        cH_tensor: 水平细节系数张量
        cV_tensor: 垂直细节系数张量
        cD_tensor: 对角线细节系数张量

        返回：
        一个张量，形状为 [b, c, h * 2, w * 2]
        """
        img_recon = self.idwt((yl, yh))  # 进行逆小波变换

        return img_recon


# 通道注意力模块 (Channel Attention Module, CAM)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)  # 第一个全连接层
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)  # 第二个全连接层

        self.sigmoid = nn.Sigmoid()

    def forward(self, A):
        # 全局平均池化和最大池化，分别得到不同的特征表示
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(A))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(A))))

        # 将两者相加并经过 Sigmoid 函数生成通道注意力权重
        out = self.sigmoid(avg_out + max_out)

        return out


# 空间注意力模块 (Spatial Attention Module, SAM)
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)  # 7x7 卷积生成空间注意力

        self.sigmoid = nn.Sigmoid()

    def forward(self, A):
        # 通过通道维度的平均池化和最大池化来获得空间特征
        avg_out = torch.mean(A, dim=1, keepdim=True)
        max_out, _ = torch.max(A, dim=1, keepdim=True)

        # 将两者拼接 (B, 2, H, W) 作为输入，通过 7x7 卷积生成空间权重
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))

        return out


# CBAM 模块，将通道和空间注意力结合
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention1 = ChannelAttention(in_channels, reduction)
        # self.spatial_attention1 = SpatialAttention()
        self.channel_attention2 = ChannelAttention(in_channels, reduction)
        # self.spatial_attention2 = SpatialAttention()

    def forward(self, A, B):
        channel_attention_a = self.channel_attention1(A)
        # spatial_attention_a = self.spatial_attention1(A)
        channel_attention_b = self.channel_attention2(B)
        # spatial_attention_b = self.spatial_attention2(B)

        # channel_attention_a = self.channel_attention1(A)
        # channel_attention_A = channel_attention_a * A
        # spatial_attention_a = self.spatial_attention1(channel_attention_A)
        # channel_attention_b = self.channel_attention2(B)
        # channel_attention_B = channel_attention_b * B
        # spatial_attention_b = self.spatial_attention2(channel_attention_B)

        A = A * channel_attention_b
        # A = A * spatial_attention_b
        B = B * channel_attention_a  # 对每个通道进行加权
        # B = B * spatial_attention_a  # 对每个空间位置进行加权

        return A, B


