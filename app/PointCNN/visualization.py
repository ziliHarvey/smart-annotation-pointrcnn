from pyntcloud.io import bin as io_bin
from pythreejs import *
import os
import numpy as np
from lib.utils import kitti_utils

def createBBox(bounding_box,C1,C2,C3,C4,C5,C6,C7,C8,color="yellow"):
    bounding_box.append(
        {
            "color":color,
            "vertices":[C1,C2,C3,C4,C1]
        })
    bounding_box.append(
        {
            "color":color,
            "vertices":[C1,C4,C8,C5,C1]
        })
    bounding_box.append(
        {
            "color":color,
            "vertices":[C1,C2,C6,C5,C1]
        })
    bounding_box.append(
        {
            "color":color,
            "vertices":[C2,C6,C7,C3,C2]
        })
    bounding_box.append(
        {
            "color":color,
            "vertices":[C3,C7,C8,C4,C3]
        })
    return bounding_box

def get_rpn_sample(argo, index):
        sample_id = index
        pts_lidar = argo.get_lidar(sample_id)
        mode = 'TRAIN'
        pts_rect = pts_lidar[:,0:3]
        pts_intensity = np.arange(pts_lidar.shape[0])
        npoints = 70000
        random_select =True
        # generate inputs
        if mode == 'TRAIN' or random_select:
            if npoints < len(pts_rect):
                # Set pts depth to Depth in Lidar frame
                pts_depth = pts_rect[:, 2] 
                # Boolean Mask for pts where depth/X axis is less than 40.0
                pts_near_flag = pts_depth < 40.0
                # Far points > 40
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                # Near points or points within < 40
                near_idxs = np.where(pts_near_flag == 1)[0]
                # Randomly choosing from near points, Total of - npoints(16834 - len(far_points))
                near_idxs_choice = np.random.choice(near_idxs, npoints - len(far_idxs_choice), replace=False)
                
                # Concatenate with far points if far points > 0
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                if npoints > len(pts_rect):
                    extra_choice = np.random.choice(choice, npoints - len(pts_rect), replace=True)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            ret_pts_rect = pts_rect[choice, :]
            ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]
        else:
            ret_pts_rect = pts_rect
            ret_pts_intensity = pts_intensity - 0.5

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        sample_info = {'sample_id': sample_id, 'random_select': random_select}

        if mode == 'TEST':
            if cfg.RPN.USE_INTENSITY:
                pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
            else:
                pts_input = ret_pts_rect
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = ret_pts_rect
            sample_info['pts_features'] = ret_pts_features
            return sample_info

        gt_obj_list = argo.get_label(sample_id)
        gt_boxes3d = objs_to_boxes3d(gt_obj_list)

        gt_alpha = np.zeros((gt_obj_list.__len__()), dtype=np.float32)
        for k, obj in enumerate(gt_obj_list):
            gt_alpha[k] = obj.alpha

        # data augmentation
        aug_pts_rect = ret_pts_rect.copy()
        aug_gt_boxes3d = gt_boxes3d.copy()
         # prepare input
        if False:
            pts_input = np.concatenate((aug_pts_rect, ret_pts_features), axis=1)  # (N, C)
        else:
            pts_input = aug_pts_rect

        if False:
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = aug_pts_rect
            sample_info['pts_features'] = ret_pts_features
            sample_info['gt_boxes3d'] = aug_gt_boxes3d
            return sample_info

        # generate training labels
        rpn_cls_label, rpn_reg_label = generate_rpn_training_labels(aug_pts_rect, aug_gt_boxes3d)
        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = aug_pts_rect
        sample_info['pts_features'] = ret_pts_features
        sample_info['rpn_cls_label'] = rpn_cls_label
        sample_info['rpn_reg_label'] = rpn_reg_label
        sample_info['gt_boxes3d'] = aug_gt_boxes3d
        
        return sample_info

def generate_rpn_training_labels(pts_rect, gt_boxes3d):
    cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
    reg_label = np.zeros((pts_rect.shape[0], 7), dtype=np.float32)  # dx, dy, dz, ry, h, w, l
    gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
    extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
    extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)

    for k in range(gt_boxes3d.shape[0]):
        box_corners = gt_corners[k]
        fg_pt_flag = in_hull(pts_rect, box_corners)
        fg_pts_rect = pts_rect[fg_pt_flag]
        cls_label[fg_pt_flag] = 1

        # enlarge the bbox3d, ignore nearby points
        extend_box_corners = extend_gt_corners[k]
        fg_enlarge_flag = in_hull(pts_rect, extend_box_corners)
        ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
        cls_label[ignore_flag] = -1

        # pixel offset of object center
        center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
        center3d[1] -= gt_boxes3d[k][3] / 2
        reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect  # Now y is the true center of 3d box 20180928

        # size and angle encoding
        reg_label[fg_pt_flag, 3] = gt_boxes3d[k][3]  # h
        reg_label[fg_pt_flag, 4] = gt_boxes3d[k][4]  # w
        reg_label[fg_pt_flag, 5] = gt_boxes3d[k][5]  # l
        reg_label[fg_pt_flag, 6] = gt_boxes3d[k][6]  # ry

    return cls_label, reg_label


