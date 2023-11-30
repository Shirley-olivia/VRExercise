# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import List, Dict

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import is_list_of

from .bbox.transforms import get_warp_matrix
from .pose_data_sample import PoseDataSample


def merge_data_samples(data_samples: List[PoseDataSample]) -> PoseDataSample:
    """Merge the given data samples into a single data sample.

    This function can be used to merge the top-down predictions with
    bboxes from the same image. The merged data sample will contain all
    instances from the input data samples, and the identical metainfo with
    the first input data sample.

    Args:
        data_samples (List[:obj:`PoseDataSample`]): The data samples to
            merge

    Returns:
        PoseDataSample: The merged data sample.
    """

    if not is_list_of(data_samples, PoseDataSample):
        raise ValueError('Invalid input type, should be a list of '
                         ':obj:`PoseDataSample`')

    if len(data_samples) == 0:
        warnings.warn('Try to merge an empty list of data samples.')
        return PoseDataSample()

    merged = PoseDataSample(metainfo=data_samples[0].metainfo)

    if 'gt_instances' in data_samples[0]:
        merged.gt_instances = InstanceData.cat(
            [d.gt_instances for d in data_samples])

    if 'pred_instances' in data_samples[0]:
        merged.pred_instances = InstanceData.cat(
            [d.pred_instances for d in data_samples])

    if 'pred_fields' in data_samples[0] and 'heatmaps' in data_samples[
            0].pred_fields:
        reverted_heatmaps = [
            revert_heatmap(data_sample.pred_fields.heatmaps,
                           data_sample.input_center, data_sample.input_scale,
                           data_sample.ori_shape)
            for data_sample in data_samples
        ]

        merged_heatmaps = np.max(reverted_heatmaps, axis=0)
        pred_fields = PixelData()
        pred_fields.set_data(dict(heatmaps=merged_heatmaps))
        merged.pred_fields = pred_fields

    if 'gt_fields' in data_samples[0] and 'heatmaps' in data_samples[
            0].gt_fields:
        reverted_heatmaps = [
            revert_heatmap(data_sample.gt_fields.heatmaps,
                           data_sample.input_center, data_sample.input_scale,
                           data_sample.ori_shape)
            for data_sample in data_samples
        ]

        merged_heatmaps = np.max(reverted_heatmaps, axis=0)
        gt_fields = PixelData()
        gt_fields.set_data(dict(heatmaps=merged_heatmaps))
        merged.gt_fields = gt_fields

    return merged


def revert_heatmap(heatmap, input_center, input_scale, img_shape):
    """Revert predicted heatmap on the original image.

    Args:
        heatmap (np.ndarray or torch.tensor): predicted heatmap.
        input_center (np.ndarray): bounding box center coordinate.
        input_scale (np.ndarray): bounding box scale.
        img_shape (tuple or list): size of original image.
    """
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().detach().numpy()

    ndim = heatmap.ndim
    # [K, H, W] -> [H, W, K]
    if ndim == 3:
        heatmap = heatmap.transpose(1, 2, 0)

    hm_h, hm_w = heatmap.shape[:2]
    img_h, img_w = img_shape
    warp_mat = get_warp_matrix(
        input_center.reshape((2, )),
        input_scale.reshape((2, )),
        rot=0,
        output_size=(hm_w, hm_h),
        inv=True)

    heatmap = cv2.warpAffine(
        heatmap, warp_mat, (img_w, img_h), flags=cv2.INTER_LINEAR)

    # [H, W, K] -> [K, H, W]
    if ndim == 3:
        heatmap = heatmap.transpose(2, 0, 1)

    return heatmap


def split_instances(instances: InstanceData) -> List[InstanceData]:
    """Convert instances into a list where each element is a dict that contains
    information about one instance."""
    results = []

    # return an empty list if there is no instance detected by the model
    if instances is None:
        return results

    for i in range(len(instances.keypoints)):
        result = dict(
            keypoints=instances.keypoints[i].tolist(),
            keypoint_scores=instances.keypoint_scores[i].tolist(),
        )
        if 'bboxes' in instances:
            result['bbox'] = instances.bboxes[i].tolist(),
            if 'bbox_scores' in instances:
                result['bbox_score'] = instances.bbox_scores[i]
        results.append(result)

    return results


def compute_angle(point1, center, point2):
    vector1 = point1 - center
    vector2 = point2 - center
    dot = np.dot(vector1, vector2)
    norm = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    cosine_similarity = np.clip(dot / norm, -1.0, 1.0)
    angle_radians = np.arccos(cosine_similarity)
    angle = np.degrees(angle_radians)

    return angle


def JointAngle(instances: InstanceData) -> List[InstanceData]:
    results = {}

    joint_connections = [
        (9, 11, 12),  # left_shoulder
        (11, 12, 13),  # left_elbow
        (9, 14, 15),  # right_shoulder
        (14, 15, 16),  # right_elbow
        (4, 5, 6),  # left_knee
        (0, 4, 5),  # left_hip
        (1, 2, 3),  # right_knee
        (0, 1, 2),  # right_hip
    ]

    if instances is None:
        return results

    for connection in joint_connections:
        point1, center, point2 = connection
        keypoints = [point[:3] for point in instances.keypoints[0][:17]]
        # print(len(keypoints))
        angle = compute_angle(keypoints[point1], keypoints[center], keypoints[point2])
        results[f"{point1}_{center}_{point2}_angle"] = angle

    return results

def compute_joint_angle_velocity(current_angle, prev_angle, fps):
    time_interval = 1 / fps

    # 计算关节角度随时间的变化率（关节角速度）
    joint_angle_velocity = (current_angle - prev_angle) / time_interval

    return joint_angle_velocity

def JointAngleVelocities(joint_angles: List[InstanceData], last_angles: List[InstanceData]) -> List[InstanceData]:
    results = {}

    if last_angles is None:
        return results

    for (key1, angle_value1), (key2, angle_value2) in zip(joint_angles.items(), last_angles.items()):
        velocity = compute_joint_angle_velocity(angle_value1, angle_value2, 30)
        # print(f"The angle for {key1} in results1 is: {angle_value1}")
        # print(f"The angle for {key2} in results2 is: {angle_value2}")
        results[key1] = velocity

    return results

def compute_joint_angle_acceleration(current_velocity, prev_velocity, fps):
    time_interval = 1 / fps

    # 计算关节角速度随时间的变化率（关节角加速度）
    joint_angle_acceleration = (current_velocity - prev_velocity) / time_interval

    return joint_angle_acceleration

def JointAngleAcceleration(joint_angle_velocities: List[InstanceData], last_velocities: List[InstanceData]) -> List[InstanceData]:
    results = {}

    if last_velocities is None:
        return results

    for (key1, velocity_value1), (key2, velocity_value2) in zip(joint_angle_velocities.items(), last_velocities.items()):
        acceleration = compute_joint_angle_acceleration(velocity_value1, velocity_value2, 30)
        # print(f"The angle for {key1} in results1 is: {angle_value1}")
        # print(f"The angle for {key2} in results2 is: {angle_value2}")
        results[key1] = acceleration

    return results


def compute_paw_angle(point1, center, point2):
    # Calculate relative directions
    direction_1 = point1 - center
    direction_2 = center - point2

    # Calculate yaw angle(point1相对于center的偏转角）
    yaw_angle = math.atan2(direction_1[1], direction_2[0])

    # Convert angle to degrees
    yaw_angle_degrees = math.degrees(yaw_angle)

    return yaw_angle_degrees


def YawAngle(instances: InstanceData) -> List[InstanceData]:
    results = {}

    joint_connections = [
        (1, 4, 5),  # right_hip, left_hip, spine
        (4, 8, 11),  # left_hip, thorax, left_shoulder
        (1, 8, 14),  # right_hip, thorax, right_shoulder
        (8, 9, 10),  # thorax, neck_base, head
        (11, 12, 13),  # left_shoulder, left_elbow, left_wrist
        (14, 15, 16)  # right_shoulder, right_elbow, right_wrist
    ]

    if instances is None:
        return results

    for connection in joint_connections:
        point1, center, point2 = connection
        keypoints = [point[:3] for point in instances.keypoints[0][:17]]
        # print(len(keypoints))
        angle = compute_paw_angle(keypoints[point1], keypoints[center], keypoints[point2])
        results[f"{point1}_{center}_paw_angle"] = angle

    return results

