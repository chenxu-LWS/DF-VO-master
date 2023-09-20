import numpy as np
import torch

def angle_error(t_R1, t_R2):
    ret = torch.empty((t_R1.shape[0]), dtype=t_R1.dtype, device=t_R1.device)
    rotation_offset = torch.matmul(t_R1.transpose(1,2), t_R2)
    tr_R = torch.sum(rotation_offset.view(-1,9)[:,::4], axis=1) # batch trace
    cos_angle = (tr_R - 1) / 2
    if torch.any(cos_angle < -1.1) or torch.any(cos_angle > 1.1):
        raise ValueError("angle out of range, input probably not proper rotation matrices")
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    return angle*(180/np.pi)

# def evaluate_pose(tvecs, rotmats, predicted_translations, predicted_rotations):
#     assert(rotmats.shape[0] == predicted_rotations.shape[0]), \
#         'batch size of ground truth and prediction must be equal'
#
#     angle_errors = angle_error(rotmats, predicted_rotations)
#     translation_errors = torch.norm((predicted_translations - tvecs), axis=1)
#     return angle_errors, translation_errors

def evaluate_pose(tvecs, predicted_translations):
    assert(tvecs.shape[0] == predicted_translations.shape[0]), \
        'batch size of ground truth and prediction must be equal'

    translation_errors = torch.norm((predicted_translations - tvecs), axis=1)
    return translation_errors

def compute_pose_histogram(thresholds, predicted_translations, predicted_rotations, gts):
    hist = np.zeros(len(thresholds))
    gts = np.asarray(gts)
    tvecs = gts[:,:3]
    rotmats = gts[:,3:]
    predicted_rotations = np.asarray(predicted_rotations)
    predicted_translations = np.asarray(predicted_translations)

    rotation_errors, translation_errors = evaluate_pose(tvecs, rotmats, predicted_translations, predicted_rotations)

    for i in range(len(rotation_errors)):
        for j in range(len(thresholds)):
            rotation_threshold, translation_threshold = thresholds[j]
            if (rotation_errors <= rotation_threshold) and (translation_errors <= translation_threshold):
                hist[j] += 1

    return hist / gts.shape[0]





