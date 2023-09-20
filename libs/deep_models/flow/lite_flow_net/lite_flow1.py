import cv2
import math
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

from .lite_flow_net import LiteFlowNet
from ..deep_flow import import DeepFlow

class LiteFlow(DeepFlow):
    def __init__(self, *args, **kwargs):
        super(LiteFlow, self).__init__(*args, **kwargs)

        self.half_flow = False

    def initialize_network_model(self, weight_path, finetune):

        if weight_path is not None:
            print("==> Initialize LiteFlowNet with[{}]".format(weight_path))

            self.model = LiteFlowNet.cuda()

            checkpoint = torch.load(weight_path)
            self.model.load_state_dict(checkpoint)

            if finetune:
                self.model.train()
            else:
                self.model.eval()
        else:
            assert False, "No LiteFlowNet pretrained model is provided."

    def inference(self, img1, img2):

        _, _, h, w =img1.shape
        th, tw = self.get_target_size(h, w)

        # forward pass
        flow_inputs = [img1, img2]
        resize_img_list = [
                           F.interpolate(
                            img, (th, tw), mode='bilinear', align_corners=True)
                           for img in flow_inputs
        ]
        output = self.model(resize_img_list)

        # Post-process output
        flows = {}
        for s in self.flow_scales:
            flows[s] = self.resize_dense_flow(
                                output[s],
                                h, w)
            if self.half_flow:
                flows[s] /= 2.
        return flows

    def inference_flow(self,
                       img1, img2,
                       forward_backward=False,
                       dataset='kitti'):

    # flow net inference to get flows
    if forward_backward:
        input_img1 = torch.cat((img1, img2), dim=0)
        input_img2 = torch.cat((img2, img1), dim=0)
    else:
        input_img1 = img1
        input_img2 = img2

    if self.enable_finetune:
        combined_flow_data = self.inference(input_img1, input_img2)
    else:
        combined_flow_data = self.inference_no_grad(input_img1, input_img2)

    self.forward_flow = {}
    self.backward_flow = {}
    self.flow_diff = {}
    self.px1on2 = {}
    for s in self.flow_scales:
        self.forward_flow[s] = combined_flow_data[s][0:1]

        if forward_backward:
            self.backward_flow[s] = combined_flow_data[s][1:2]

            # sampled flow
            # Get sampling pixel coordinates
            self.px1on2[s] = self.flow_to_pix(self.forward_flow[s])

            # Forward-Backward flow consistency check
            if forward_backward:

                self.flow_diff[s] = self.forward_backward_consistency(
                                    flow1=self.forward_flow[s],
                                    flow2=self.backward_flow[s],
                                    px1on2=self.px1on2[s])

        # summarize flow data and flow difference for DF-VO
        flows = {}
        flows['forward'] = self.forward_flow[1].clone()
        if forward_backward:
            flows['backward'] = self.backward_flow[1].clone()
            flows['flow_diff'] = self.flow_diff[1].clone()
        return flows



