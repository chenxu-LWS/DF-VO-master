import cv2
import copy
from glob import glob
import math
from matplotlib import pyplot as plt
import numpy as np
import os
from time import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from libs.datasets.SevenScene import SevenScenes
from libs.geometry.camera_modules import SE3
import libs.datasets as Dataset
from libs.deep_models.my_deep_models import DeepModel
from libs.general.frame_drawer import FrameDrawer
from libs.general.timer import Timer
from libs.matching.keypoint_sampler import KeypointSampler
from libs.matching.depth_consistency import DepthConsistency
from libs.tracker import EssTracker, PnpTracker
from libs.general.utils import *
from tools import eval as ev


class localization():
    def __init__(self, cfg):

        # configuration
        self.cfg = cfg

        # # predicted global poses
        # self.global_poses = {0: SE3()}

        # predicted poses  (relative/absolute)
        self.pose = {}
        # self.pose['Rotation'] = []
        # self.pose['translations'] = []

        # reference and current data
        self.initialize_data()

        self.setup()

    def setup(self):
        """Reading configuration and setup, including

            - Timer
            - Dataset
            - Tracking method
            - Keypoint Sampler
            - Deep networks
            - Deep layers
            - Visualizer
        """
        self.timers = Timer()

        kwargs = {
            "image_size": 224,
            "mode": "ceiling_train",
            "half_image": "store_true",
            "crop": None,
            "gauss_kernel": 3,
            "gauss_sigma": (0.05, 5.0),
            "jitter_brightness": 0.7,
            "jitter_contrast": 0.7,
            "jitter_saturation": 0.7,
            "jitter_hue": 0.5,
        }

        train_set = SevenScenes(self.cfg.directory.gt_pose_dir, **kwargs)
        self.train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

        self.dataset = Dataset.datasets[self.cfg.dataset](self.cfg)

        self.tracking_method = self.cfg.tracking_method
        self.initialize_tracker()

        self.kp_sampler = KeypointSampler(self.cfg)

        self.deep_models = DeepModel(self.cfg)
        self.deep_models.initialize_models()
        if self.cfg.online_finetune.enable:
            self.deep_models.setup_train()

        # visualization interface
        self.drawer = FrameDrawer(self.cfg.visualization)

    def initialize_data(self):

        self.ref_data = {}
        self.cur_data = {}

    def initialize_tracker(self):
        if self.tracking_method == 'PnP':
            self.pnp_tracker = PnpTracker(self.cfg, self.dataset.cam_intrinsics)

    # 不一定用
    def update_global_pose(self, new_pose, scale=1.):
        """update estimated poses w.r.t global coordinate system

        Args:
            new_pose (SE3): new pose
            scale (float): scaling factor
        """
        self.cur_data['pose'].t = self.cur_data['pose'].R @ new_pose.t * scale \
                            + self.cur_data['pose'].t
        self.cur_data['pose'].R = self.cur_data['pose'].R @ new_pose.R
        self.global_poses[self.cur_data['id']] = copy.deepcopy(self.cur_data['pose'])

    def update_pose(self, new_pose, scale=1.):
        self.ref_data['pred_t'].t = self.cur_data['pose'].R @ new_pose.t * scale \
                            + self.cur_data['pose'].t

    def evaluate(self):
        translation_error = ev.evaluate_pose(tvecs=self.ref_data['gt_pose'][:,:3],
                                              predicted_translations=self.ref_data['pred_t']
                                              )
        print(("translation error %f, %f, %f") % (torch.median(translation_error)))



    def PnP_tracking(self):

        self.ref_data['pred_t'] = SE3()
        self.ref_data['motion'] = SE3()

        if self.tracking_method in ['hybrid', 'PnP']:
            self.timers.start('kp_sel','tracking')
            kp_sel_outputs = self.kp_sampler.kp_selection(self.cur_data, self.ref_data)
            if kp_sel_outputs['goood_kp_found']:
                self.kp_sampler.update_kp_data(self.cur_data, self.ref_data, kp_sel_outputs)
            self.timers.end('kp_sel')

        ''' Pose estimation using PnP '''
        # Initialize pnp pose
        hybrid_pose = SE3()

        # if not(kp_sel_outputs['good_kp_found']):
        #     print("No enough good keypoints, constant motion will be used!")
        #     pose = self.ref_data['motion']
        #     self.update_global_pose(pose, 1)
        #     return

        ''' PnP-tracker '''
        self.timers.start('pnp','tracking')
        pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
            self.ref_data[self.cfg.pnp_tracker.kp_src],
            self.cur_data[self.cfg.pnp_tracker.kp_src],
            self.ref_data['depth'],
            not (self.cfg.pnp_tracker.iterative_kp.enable)
        )  # pose: from cur->ref

        if self.cfg.pnp_tracker.iterative_kp.enable:
            self.pnp_tracker.compute_rigid_flow_kp(self.cur_data, self.ref_data, pnp_outputs['pose'])
            pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                self.ref_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                self.cur_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                self.ref_data['depth'],
                True
            )  # pose: from cur->ref

        self.timers.end('pnp')

        hybrid_pose = pnp_outputs['pose']
        # self.tracking_mode = "PnP"

        ''' Summarize data '''
        # update pose
        self.ref_data['pose'] = copy.deepcopy(hybrid_pose)
        pose = self.ref_data['pose']
        # self.update_global_pose(pose, 1)
        self.update_pose(pose, 1)

    def update_data(self, ref_data, cur_data):
        """Update data

        Args:
            ref_data (dict): reference data
            cur_data (dict): current data

        Returns:
            ref_data (dict): updated reference data
            cur_data (dict): updated current data
        """
        # for key in cur_data:
        #     if key == "id":
        #         ref_data['id'] = cur_data['id']
        #     else:
        #         if ref_data.get(key, -1) is -1:
        #             ref_data[key] = {}
        #         ref_data[key] = cur_data[key]

        # Delete unused flow to avoid data leakage
        ref_data['flow'] = None
        cur_data['flow'] = None
        ref_data['flow_diff'] = None
        return ref_data, cur_data

    def load_raw_data(self):
        # Reading image
        self.cur_data['img'] = self.dataset.get_image(self.cur_data['timestamp'])

        # Reading/Predicting depth
        if self.dataset.data_dir['depth_src'] is not None:
            self.cur_data['raw_depth'] = self.dataset.get_depth(self.cur_data['timestamp'])

    def load_data(self, batch):
        # load batch
        self.ref_data['id'] = batch[5][0]
        self.cur_data['id'] = batch[5][1]

        self.ref_data['img'] = batch[0]
        self.cur_data['img'] = batch[1]

        self.ref_data['gt_pose'] = batch[4][0]
        self.cur_data['gt_pose'] = batch[4][1]

        self.ref_data['raw_depth'] = batch[2]  # ref depth
        self.cur_data['raw_depth'] = batch[3]

        self.ref_data['depth'] = preprocess_depth(self.ref_data['raw_depth'], self.cfg.crop.depth_crop, [self.cfg.depth.min_depth, self.cfg.depth.max_depth])
        self.cur_data['depth'] = preprocess_depth(self.cur_data['raw_depth'], self.cfg.crop.depth_crop, [self.cfg.depth.min_depth, self.cfg.depth.max_depth])


    def deep_model_inference(self):

        # matrix fisher
        # self.timers.start('fisher', 'deep inference')
        # rotation = self.deep_models.forward_fisher(
        #                 [self.ref_data['img'], self.cur_data['img']]
        #                 )
        # self.ref_data['deep_rotation'] = rotation
        # self.timers.end('fisher')


        if self.tracking_method in ['hybrid', 'pnp']:
            # 深度直接用数据集的

            # Two-view flow
            self.timers.start('flow_cnn', 'deep inference')
            flows = self.deep_models.forward_flow(
                self.cur_data,
                self.ref_data,
                forward_backward=self.cfg.deep_flow.forward_backward)

            # Store flow
            self.ref_data['flow'] = flows[(self.ref_data['id'], self.cur_data['id'])].copy()
            if self.cfg.deep_flow.forward_backward:
                self.cur_data['flow'] = flows[(self.cur_data['id'], self.ref_data['id'])].copy()
                self.ref_data['flow_diff'] = flows[(self.ref_data['id'], self.cur_data['id'], "diff")].copy()

            self.timers.end('flow_cnn')




    def main(self):
        """Main program
        """
        print("==> Start Camera Relocalization")

        for epoch in range(self.cfg.num_epochs):
            for _, batch in enumerate(self.train_loader):

                """ Data reading """
                # Initialize ids and timestamps
                # self.cur_data['id'] = img_id
                # self.cur_data['timestamp'] = self.dataset.get_timestamp(img_id)

                # Read image data and (optional) precomputed depth data
                self.timers.start('data_loading')
                # self.load_raw_data()
                self.load_data(batch)
                self.timers.end('data_loading')

                # Deep model inferences
                self.timers.start('deep_inference')
                self.deep_model_inference()
                self.timers.end('deep_inference')

                """ Visual odometry """
                self.timers.start('tracking')
                self.PnP_tracking()
                self.timers.end('tracking')

                """ Online Finetuning """
                if self.cfg.online_finetune.enable:
                    self.deep_models.finetune(self.ref_data['img'], self.cur_data['img'],
                                              self.ref_data['pose'].pose,
                                              self.dataset.cam_intrinsics.mat,
                                              self.dataset.cam_intrinsics.inv_mat)

                """ evaluate """
                self.timers.start('evaluation')
                self.evaluate()
                self.timers.end('evaluation')

                """ Update reference and current data """
                self.ref_data, self.cur_data = self.update_data(
                    self.ref_data,
                    self.cur_data,
                )

                # self.tracking_stage += 1
                print("=> Finish!")

        # save finetuned model
        if self.cfg.online_finetune.enable and self.cfg.online_finetune.save_model:
            self.deep_models.save_model()

        # Output experiement information
        self.timers.time_analysis()