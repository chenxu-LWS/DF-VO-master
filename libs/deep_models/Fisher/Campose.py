import cv2
import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from .deep_fisher import DeepFisher
from loss import vmf_loss, gauss_log_likelihood, batch_torch_A_to_R


class CamPose(DeepFisher):
    def __init__(self, *args, **kwargs):
        super(CamPose).__init__(*args, **kwargs)
        self.enable_finetune = False

        # image size
        self.feed_width = 224
        self.feed_height = 224

    def initialize_network_model(self, weight_path, finetune):

        device = torch.device('cuda')

        # Initialize network
        self.model = CamPoseNet(base='ResNet', pretrained=True).cuda()

        if weight_path is not None:
            print("==> Initialize CamPoseNet with [{}}".format(weight_path))

            checkpoint = torch.load(weight_path)
            self.model.load_state_dict(checkpoint)
            self.model.to(device)

        if finetune:
            self.model.train()
        else:
            self.model.eval()

        # image size
        self.feed_height = 192
        self.feed_width = 640

    def inference(self, imgs):
        """Pose prediction

        Args:
            imgs (tensor, Nx2CxHxW): concatenated image pair

        Returns:
            pose (tensor, [Nx3x3]): relative pose from img2 to img1
        """
        out = self.model(imgs)
        R_est = batch_torch_A_to_R(out.view(-1, 3, 3))

        return R_est

    def inference_pose(self, img):
        """Pose prediction

        Args:
            imgs (tensor, Nx2CxHxW): concatenated image pair

        Returns:
            pose (tensor, [Nx3x3]): relative rotation from img2 to img1
        """
        if self.enable_finetune:
            R_est = self.inference(img)
        else:
            R_est = self.inference_no_grad(img)
        self.pred_pose = R_est

        # summarize pose predictions
        pose = self.pred_pose[:1].clone() # 即第一帧的相机姿态
        return pose


