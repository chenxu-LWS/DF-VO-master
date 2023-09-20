import numpy as np
import os
import PIL.Image as pil
import torch
import torch.optim as optim
from torchvision import transforms

from .depth.monodepth2.monodepth2 import Monodepth2DepthNet
from .flow.lite_flow_net.lite_flow import LiteFlow
from .flow.hd3.hd3_flow import HD3Flow
from .pose.monodepth2.monodepth2 import Monodepth2PoseNet
from libs.deep_models.depth.monodepth2.layers import FlowToPix, PixToFlow, SSIM, get_smooth_loss
from libs.general.utils import mkdir_if_not_exists
from .Fisher import Campose

class DeepModel():

    def __init__(self, cfg):
        self.cfg = cfg
        self.finetune_cfg = self.cfg.online_finetune
        self.device = torch.device('cuda')

    def initialize_models(self):
        ''' optical flow '''
        self.flow = self.initalize_deep_flow_model()

        ''' fisher '''
        self.fisher = self.initialize_deep_fisher_model()

    def initialize_deep_flow_model(self):
        if self.cfg.deep_flow.network == 'liteflow':
            flow_net = LiteFlow(self.cfg.image.height, self.cfg.image.width)
            enable_finetune = self.finetune_cfg.enable and self.finetune_cfg.flow.enable
            flow_net.initialize_network_model(
                    weight_path=self.cfg.deep_flow.flow_net_weight,
                    finetune=enable_finetune,
                    )
        elif self.cfg.deep_flow.network == 'hd3':
            flow_net = HD3Flow(self.cfg.image.height, self.cfg.image.width)
            enable_finetune = self.finetune_cfg.enable and self.finetune_cfg.flow.enable
            flow_net.initialize_network_model(
                    weight_path=self.cfg.deep_flow.flow_net_weight,
                    finetune=enable_finetune,
                    )
        else:
            assert False, "Invalid flow network [{}] is provided.".format(
                                self.cfg.deep_flow.network
                                )
        return flow_net

    def initialize_deep_fisher_model(self):
        fisher_net = CamPose()
        enable_finetune = self.finetune_cfg and self.finetune_cfg.fisher.enable
        fisher_net.initialize_network_model(
                weight_path=self.cfg.deep_fisher.pretrained_model,
                finetune=enable_finetune
        )
        return fisher_net

    def setup_train(self):
        # Basic configuration
        self.img_cnt = 0
        self.frame_ids = [0, 1]

        # Optimization
        self.learning_rate = self.finetune_cfg.lr
        self.parameters_to_train = []

        # flow train setup
        if self.finetune_cfg.flow.enable:
            self.flow.setup_train(self, self.finetune_cfg.flow)

        # fisher train setup
        if self.finetune_cfg.fisher.enable:
            self.flow.setup_train(self,self.finetune_cfg.fisher)

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)

    def forward_flow(self, in_cur_data, in_ref_data, forward_backward):
        """Optical flow network forward interface, a forward inference.

        Args:
            in_cur_data (dict): current data
            in_ref_data (dict): reference data
            forward_backward (bool): use forward-backward consistency if True

        Returns:
            flows (dict): predicted flow data. flows[(id1, id2)] is flows from id1 to id2.

                - **flows(id1, id2)** (array, 2xHxW): flows from id1 to id2
                - **flows(id2, id1)** (array, 2xHxW): flows from id2 to id1
                - **flows(id1, id2, 'diff)** (array, 1xHxW): flow difference of id1
        """
        # Preprocess image
        # cur_imgs = np.transpose((in_cur_data['img']) / 255, (2, 0, 1))
        # ref_imgs = np.transpose((in_ref_data['img']) / 255, (2, 0, 1))

        cur_imgs = in_cur_data['img']
        ref_imgs = in_ref_data['img']
        cur_imgs = cur_imgs.float().cuda()
        ref_imgs = ref_imgs.float().cuda()

        # Forward pass
        flows = {}

        # Flow inference
        batch_flows = self.flow.inference_flow(
            img1=ref_imgs,
            img2=cur_imgs,
            forward_backward=forward_backward,
            dataset=self.cfg.dataset)

        # Save flows at current view
        src_id = in_ref_data['id']
        tgt_id = in_cur_data['id']
        flows[(src_id, tgt_id)] = batch_flows['forward'].detach().cpu().numpy()[0]
        if forward_backward:
            flows[(tgt_id, src_id)] = batch_flows['backward'].detach().cpu().numpy()[0]
            flows[(src_id, tgt_id, "diff")] = batch_flows['flow_diff'].detach().cpu().numpy()[0]
        return flows

#=================================================
    # def forward_fisher(self, imgs):
    #     # Preprocess
    #     img_tensor = []
    #     for img in imgs:
    #         input_image = pil.fromarray(img)
    #         input_image = input_image.resize((self.fisher.feed_width, self.fisher.feed_height), pil.LANCZOS)
    #         input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    #         img_tensor.append(input_image)
    #     img_tensor = torch.cat(img_tensor, 1)
    #     img_tensor = img_tensor.cuda()
    #
    #     # Prediction
    #     pred_fisher_Rotation = self.fisher.inference_pose(img_tensor)
    #     rotation = pred_fisher_Rotation.detach().cpu().numpy()[0]  #
    #     return rotation

    def finetune(self, img1, img2, K, inv_K):
        """Finetuning deep models

        Args:
            img1 (array, [HxWx3]): image 1 (reference)
            img2 (array, [HxWx3]): image 2 (current)
            pose (array, [4x4]): relative pose from view-2 to view-1 (from DF-VO)
            K (array, [3x3]): camera intrinsics
            inv_K (array, [3x3]): inverse camera intrinsics
        """
        # preprocess data
        # images
        # img1 = np.transpose((img1)/255, (2, 0, 1))
        # img2 = np.transpose((img2)/255, (2, 0, 1))
        img1 = img1.unsqueeze(0).float().cuda()
        img2 = img2.unsqueeze(0).float().cuda()

        # camera intrinsics
        K44 = np.eye(4)
        K44[:3, :3] = K.copy()
        K = torch.from_numpy(K44).unsqueeze(0).float().cuda()
        K44[:3, :3] = inv_K.copy()
        inv_K = torch.from_numpy(K44).unsqueeze(0).float().cuda()

        if self.finetune_cfg.num_frames is None or self.img_cnt < self.finetune_cfg.num_frames:
            ''' data preparation '''
            losses = {'loss': 0}
            inputs = {
                ('color', 0, 0): img1,
                ('color', 1, 0): img2,
                ('K', 0): K,
                ('inv_K', 0): inv_K,
            }
            outputs = {}

            ''' loss computation '''
            # flow
            if self.finetune_cfg.flow.enable:
                assert self.cfg.deep_flow.forward_backward, "forward-backward option has to be True for finetuning"
                for s in self.flow.flow_scales:
                    outputs.update(
                        {
                            ('flow', 0, 1, s): self.flow.forward_flow[s],
                            ('flow', 1, 0, s): self.flow.backward_flow[s],
                            ('flow_diff', 0, 1, s): self.flow.flow_diff[s]
                        }
                    )
                losses.update(self.flow.train(inputs, outputs))
                losses['loss'] += losses['flow_loss']

            # if self.finetune_cfg.fisher.enable:
            #     outputs.update(
            #         {
            #             ('fisher_rotation'): pose
            #         }
            #     )
            #======================================
            # losses.update(self.fisher.train())

            ''' backward '''
            self.model_optimizer.zero_grad()
            losses.backward()
            self.model_optimizer.step()

            self.img_cnt += 1

        else:
            # reset flow model to eval mode
            if self.finetune_cfg.flow.enable:
                self.flow.model.eval()

            # reset fisher model to eval mode
            # if self.finetune_cfg.fisher.enable:
            #     self.fisher.model.eval()

    def save_model(self):
        save_folder = os.path.join(self.cfg.directory.result_dir, "deep_models")
        mkdir_if_not_exists(save_folder)

        # Save Flow model
        model_name = "flow"
        model = self.flow.model
        ckpt_path = os.path.join(save_folder, "{}.pth".format(model_name))
        torch.save(model.state_dict(), ckpt_path)

        # Save fisher model
        # model_name = "fisher"
        # model = self.fisher.model
        # ckpt_path = os.path.join(save_folder, "{}.pth".format(model_name))
        # torch.save(model.state_dict(), ckpt_path)
