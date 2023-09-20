import torch
from typing import Tuple, List, Iterable, Optional
from pathlib import Path
import numpy as np
from .utils import quat_to_rotmat    #from vapor-main
import os.path as osp
import cv2

def read_poses(
    poses_path: Path,
    dataset: str = "AmbiguousReloc",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    with open(poses_path) as f:
        content = f.readlines()
    parser_poses = np.array(
        [[entry for entry in line.strip().split()] for line in content[3:]],
        dtype=str,
    )
    file_paths = parser_poses[:, 0]
    seq_ids, img_ids = np.array(
        [file_path.split("/") for file_path in file_paths]
    ).T
    try:
        seq_ids = np.array([seq_id[4:] for seq_id in seq_ids], dtype=int)
        img_ids = np.array([img_id.split(".")[0][6:] for img_id in img_ids], dtype=int)
    except:
        seq_ids = None
        img_ids = None
    tvecs = parser_poses[:, 1:4].astype(float)
    rotmats = quat_to_rotmat(parser_poses[:, 4:].astype(float))
    return seq_ids, img_ids, tvecs, rotmats, file_paths


def read_depth(path, scale, target_size=None):
    depth = cv2.imread(path, -1) / scale
    if target_size is not None:
        img_h, img_w = target_size
        depth = cv2.resize(depth,
                           (img_w, img_h),
                           interpolation=cv2.INTER_NEAREST
                           )
    return depth

def get_depth(path, seq_id, img_id):
    scale_factor = 500
    img_h = 640
    img_w = 480
    img_path = "seq-0{:01d}/frame-{:06d}.depth.png".format(seq_id, img_id)
    # eg. /seq-01/frame-000999.depth.png
    img_paths = osp.join(path, img_path)
    depth = read_depth(img_paths, scale_factor, [img_h,img_w])
    return depth

def get_depth_path(path, seq_id, img_id):
    scale_factor = 500
    img_h = 640
    img_w = 480
    img_path = "seq-0{:01d}/frame-{:06d}.depth.png".format(seq_id, img_id)
    # eg. /seq-01/frame-000999.depth.png
    img_paths = osp.join(path, img_path)
    # depth = read_depth(img_paths, scale_factor, [img_h, img_w])
    return img_paths


